import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
from pathlib import Path

from skimage import exposure
from skimage.filters import gaussian, sato
from skimage.segmentation import morphological_chan_vese
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import (
    skeletonize, remove_small_objects, binary_opening, binary_closing,
    disk, binary_erosion, binary_dilation
)
from skimage.graph import route_through_array

plt.rcParams["figure.dpi"] = 120

ROOT_DIR = Path(__file__).resolve().parent

# ── CV2 helpers ───────────────────────────────────────────────────────────────

def make_init_levelset_ellipse(h, w, cx=None, cy=None, rx=None, ry=None):
    if cx is None: cx = w // 2
    if cy is None: cy = h // 2
    if rx is None: rx = int(0.35 * w)
    if ry is None: ry = int(0.35 * h)
    Y, X = np.ogrid[:h, :w]
    return ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0

def extract_mask_chanvese(image_path, sigma=2.5, iterations=350, smoothing=2):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    img_f = img.astype(np.float32) / 255.0
    img_eq = exposure.equalize_adapthist(img_f, clip_limit=0.02)
    img_s  = gaussian(img_eq, sigma=sigma, preserve_range=True)
    h, w = img_s.shape
    init = make_init_levelset_ellipse(h, w)
    mask = morphological_chan_vese(
        img_s, num_iter=iterations, init_level_set=init,
        smoothing=smoothing, lambda1=1.0, lambda2=1.0,
    ).astype(bool)
    return img, mask

def resample_closed_polyline(P, N):
    P = np.asarray(P, dtype=np.float64)
    Q = np.vstack([P, P[0]])
    d = np.sqrt(((Q[1:] - Q[:-1]) ** 2).sum(axis=1))
    s = np.hstack([[0.0], np.cumsum(d)])
    t = np.linspace(0, s[-1], N + 1)[:-1]
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])

def extract_outer_outline_from_mask(mask_bool, N=512):
    contours = find_contours(mask_bool.astype(np.float32), 0.5)
    c = max(contours, key=lambda a: a.shape[0])
    raw = np.column_stack([c[:, 1], c[:, 0]])
    return resample_closed_polyline(raw, N), raw

def _neighbors8_count(bin_img):
    b = bin_img.astype(np.uint8)
    p = np.pad(b, 1, mode="constant", constant_values=0)
    return (
        p[0:-2,0:-2] + p[0:-2,1:-1] + p[0:-2,2:] +
        p[1:-1,0:-2]               + p[1:-1,2:] +
        p[2:  ,0:-2] + p[2:  ,1:-1] + p[2:  ,2:]
    )

def _endpoints_of_skeleton(skel):
    nb = _neighbors8_count(skel)
    ys, xs = np.where(skel & (nb == 1))
    return np.column_stack([ys, xs])

def _farthest_pair(points_rc):
    P = points_rc.astype(float)
    dmax = -1
    a = b = None
    for i in range(len(P)):
        d = np.sum((P[i] - P)**2, axis=1)
        j = np.argmax(d)
        if d[j] > dmax:
            dmax = d[j]
            a, b = P[i], P[j]
    return tuple(a.astype(int)), tuple(b.astype(int))

def _rc_path_to_xy(path_rc):
    path_rc = np.asarray(path_rc)
    return np.column_stack([path_rc[:, 1], path_rc[:, 0]]).astype(float)

def _extract_ordered_path_from_component(comp_mask):
    endpoints = _endpoints_of_skeleton(comp_mask)
    ys, xs = np.where(comp_mask)
    pts_all = np.column_stack([ys, xs])
    if len(pts_all) < 10:
        return None
    if len(endpoints) >= 2:
        a_rc, b_rc = _farthest_pair(endpoints)
    else:
        a_rc, b_rc = _farthest_pair(pts_all)
    cost = np.ones_like(comp_mask, dtype=np.float32) * 50.0
    cost[comp_mask] = 1.0
    path_rc, _ = route_through_array(cost, start=a_rc, end=b_rc,
                                     fully_connected=True, geometric=True)
    return _rc_path_to_xy(path_rc)

def resample_open_polyline(P, N):
    P = np.asarray(P, dtype=np.float64)
    d = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(axis=1))
    s = np.hstack([[0.0], np.cumsum(d)])
    t = np.linspace(0, s[-1], N)
    return np.column_stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])

def _polyline_length(P):
    P = np.asarray(P, float)
    if len(P) < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.sum((P[1:] - P[:-1])**2, axis=1))))

def anchor_hits(curve_xy, anchor_roi, H, W):
    if curve_xy is None:
        return 0
    pts = np.round(curve_xy).astype(int)
    pts[:,0] = np.clip(pts[:,0], 0, W-1)
    pts[:,1] = np.clip(pts[:,1], 0, H-1)
    return int(np.sum(anchor_roi[pts[:,1], pts[:,0]]))

def build_ridge_map(img_u8, cell_mask,
                    clahe_clip=0.02, smooth_sigma=1.05,
                    sato_sigmas=(2,3,4,5,6,7)):
    img_f = img_u8.astype(np.float32) / 255.0
    inner = binary_erosion(cell_mask, disk(1))
    img_in = img_f.copy()
    img_in[~inner] = 0.0
    img_eq = exposure.equalize_adapthist(img_in, clip_limit=clahe_clip)
    img_s  = gaussian(img_eq, sigma=smooth_sigma, preserve_range=True)
    img_inv = (img_s.max() - img_s)
    ridge_w = sato(img_inv, sigmas=list(sato_sigmas), black_ridges=False)
    ridge_w[~cell_mask] = 0.0
    return ridge_w

def ridge_roi_to_best_curve(ridge_w, roi_mask, ridge_percentile=76,
                             open_r=2, close_r=3, min_band_size=400,
                             bridge_r=1, score_fn=None):
    vals = ridge_w[roi_mask]
    if vals.size < 50:
        return None, {}
    thr = np.percentile(vals, ridge_percentile)
    band = (ridge_w >= thr) & roi_mask
    band = binary_opening(band, disk(open_r))
    band = binary_closing(band, disk(close_r))
    band = remove_small_objects(band, min_size=int(min_band_size))
    skel = skeletonize(band)
    if bridge_r and bridge_r > 0:
        skel = skeletonize(binary_dilation(skel, disk(bridge_r)))
    lab = label(skel)
    if lab.max() < 1:
        return None, dict(thr=thr, band=band, skel=skel)
    regs = regionprops(lab)
    best_label = None
    best_score = -1e18
    for r in regs:
        comp = (lab == r.label)
        if comp.sum() < 50:
            continue
        score = float(score_fn(comp)) if score_fn is not None else float(comp.sum())
        if score > best_score:
            best_score = score
            best_label = r.label
    comp = (lab == best_label)
    return _extract_ordered_path_from_component(comp), dict(thr=thr, band=band, skel=skel, comp=comp, best_score=best_score)

def _nearest_outline_index(outline_xy, p_xy):
    d2 = np.sum((outline_xy - p_xy)**2, axis=1)
    return int(np.argmin(d2))

def _arc_indices(iA, iB, N):
    if iA <= iB:
        return np.arange(iA, iB + 1)
    return np.concatenate([np.arange(iA, N), np.arange(0, iB + 1)])

def _get_two_arcs_oriented_iA_to_iB(outline_xy, iA, iB):
    N = len(outline_xy)
    arc_fwd = outline_xy[_arc_indices(iA, iB, N)]
    arc_bwd = outline_xy[_arc_indices(iB, iA, N)][::-1]
    return arc_fwd, arc_bwd

def _choose_arc_by_side(arc1, arc2, x_center, prefer="right"):
    m1 = float(np.mean(arc1[:, 0]))
    m2 = float(np.mean(arc2[:, 0]))
    if prefer == "right":
        return arc1 if m1 >= m2 else arc2
    return arc1 if m1 <= m2 else arc2

def _drop_consecutive_duplicates(P, tol2=1e-9):
    P = np.asarray(P, float)
    if len(P) <= 1:
        return P
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = np.sum((P[1:] - P[:-1])**2, axis=1) > tol2
    return P[keep]

def _ensure_implicit_closed(loop_xy, tol2=1e-9):
    loop_xy = _drop_consecutive_duplicates(loop_xy, tol2=tol2)
    if np.sum((loop_xy[0] - loop_xy[-1])**2) < tol2:
        loop_xy = loop_xy[:-1]
    return loop_xy

def _ensure_closed(P, tol2=1e-9):
    P = np.asarray(P, float)
    if np.sum((P[0] - P[-1])**2) > tol2:
        P = np.vstack([P, P[0]])
    return P

def _drop_last_duplicate(P, tol2=1e-9):
    P = np.asarray(P, float)
    if len(P) >= 2 and np.sum((P[0]-P[-1])**2) < tol2:
        return P[:-1]
    return P

def smooth_closed_curve_moving_average(P, win=9, passes=2):
    P = _drop_last_duplicate(_ensure_closed(P))
    if win < 3:
        return P.copy()
    if win % 2 == 0:
        win += 1
    half = win // 2
    Q = P.copy()
    N = len(Q)
    for _ in range(max(1, int(passes))):
        Q2 = np.zeros_like(Q)
        for i in range(N):
            idx = (np.arange(i-half, i+half+1) % N).astype(int)
            Q2[i] = np.mean(Q[idx], axis=0)
        Q = Q2
    return Q

def resample_closed_polyline_uniform_arclength(P, N_assign):
    P = np.asarray(P, float)
    Q = _ensure_closed(P)
    seg = Q[1:] - Q[:-1]
    d = np.sqrt(np.sum(seg**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = float(s[-1])
    t = np.linspace(0.0, L, int(N_assign) + 1)[:-1]
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])

def smooth_then_upsample_closed_loop(combined_loop_xy, N_smooth=256, N_samples=2048,
                                      movavg_win=13, movavg_passes=2,
                                      post_win=7, post_passes=1):
    P = np.asarray(combined_loop_xy, float)
    Ps = smooth_closed_curve_moving_average(P, win=movavg_win, passes=movavg_passes)
    P_coarse = resample_closed_polyline_uniform_arclength(Ps, int(N_smooth))
    P_coarse = smooth_closed_curve_moving_average(P_coarse, win=post_win, passes=post_passes)
    return resample_closed_polyline_uniform_arclength(P_coarse, int(N_samples)), P_coarse

def normalize_and_scale_to_physical(loop_xy_px, Lx_um, Ly_um):
    P = np.asarray(loop_xy_px, float)
    x, y = P[:, 0], P[:, 1]
    xc, yc = x - x.mean(), y - y.mean()
    Lx_px = xc.max() - xc.min()
    Ly_px = yc.max() - yc.min()
    return np.column_stack([xc / Lx_px * Lx_um, yc / Ly_px * Ly_um])


# ── Alignment helpers ─────────────────────────────────────────────────────────

def add_theta_and_sort_plane(arr4, plane="yz"):
    arr4 = np.asarray(arr4, dtype=float)
    ID, x, y, z = arr4[:,0], arr4[:,1], arr4[:,2], arr4[:,3]
    if plane == "yz":
        theta = np.degrees(np.arctan2(z, y))
    elif plane == "xz":
        theta = np.degrees(np.arctan2(z, x))
    else:
        raise ValueError("plane must be 'yz' or 'xz'.")
    theta = np.mod(theta, 360.0)
    arr5 = np.column_stack([ID, x, y, z, theta])
    return arr5[np.argsort(arr5[:, 4])]

def _center(P):
    return P - P.mean(axis=0, keepdims=True)

def _rotate_2d(P, phi_deg):
    phi = np.deg2rad(phi_deg)
    c, s = np.cos(phi), np.sin(phi)
    return P @ np.array([[c, -s], [s, c]], dtype=float).T

def _chamfer_sym(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.inf
    dA, _ = cKDTree(B).query(A, k=1)
    dB, _ = cKDTree(A).query(B, k=1)
    return float(dA.mean() + dB.mean())

def align_exp_to_ref_by_rotation_2d(exp_2d, ref_2d,
                                     coarse_step=5.0, fine_window=6.0,
                                     fine_step=0.25, center=True):
    exp0 = _center(np.asarray(exp_2d, float)) if center else np.asarray(exp_2d, float).copy()
    ref0 = _center(np.asarray(ref_2d, float)) if center else np.asarray(ref_2d, float).copy()
    best_phi, best_score = 0.0, np.inf
    for phi in np.arange(0.0, 360.0, coarse_step):
        sc = _chamfer_sym(_rotate_2d(exp0, phi), ref0)
        if sc < best_score:
            best_score, best_phi = sc, float(phi)
    for phi in np.arange(best_phi - fine_window, best_phi + fine_window + 1e-12, fine_step):
        phi_w = float(np.mod(phi, 360.0))
        sc = _chamfer_sym(_rotate_2d(exp0, phi_w), ref0)
        if sc < best_score:
            best_score, best_phi = sc, phi_w
    return _rotate_2d(exp0, best_phi), best_phi, best_score

def _order_boundary_knn(points2d, k=10, lam=0.15, start_mode="max_first"):
    P = np.asarray(points2d, float)
    N = P.shape[0]
    tree = cKDTree(P)
    k = int(np.clip(k, 4, min(30, N-1)))
    if start_mode == "max_first":
        start = int(np.argmax(P[:, 0]))
    elif start_mode == "min_first":
        start = int(np.argmin(P[:, 0]))
    elif start_mode == "max_r":
        start = int(np.argmax(np.sum(P**2, axis=1)))
    else:
        start = 0
    visited = np.zeros(N, dtype=bool)
    order = [start]
    visited[start] = True
    _, nn = tree.query(P[start], k=k+1)
    second = int(nn[1])
    order.append(second)
    visited[second] = True
    prev, cur = start, second
    prev_dir = P[cur] - P[prev]
    prev_dir /= (np.linalg.norm(prev_dir) + 1e-15)
    for _ in range(N - 2):
        _, nn = tree.query(P[cur], k=k+1)
        candidates = [int(j) for j in nn[1:] if not visited[int(j)]]
        if not candidates:
            remaining = np.where(~visited)[0]
            if remaining.size == 0:
                break
            nxt = int(remaining[np.argmin(np.linalg.norm(P[remaining] - P[cur], axis=1))])
        else:
            best_score, nxt = -np.inf, None
            for j in candidates:
                v = P[j] - P[cur]
                dist = np.linalg.norm(v)
                vhat = v / (dist + 1e-15)
                score = float(np.dot(prev_dir, vhat)) - lam * dist
                if score > best_score:
                    best_score, nxt = score, j
        order.append(nxt)
        visited[nxt] = True
        v = P[nxt] - P[cur]
        prev_dir = v / (np.linalg.norm(v) + 1e-15)
        prev, cur = cur, nxt
    return P[np.array(order)]

def _resample_closed_curve_arclength(ordered2d, M):
    P = np.asarray(ordered2d, float)
    if np.linalg.norm(P[0] - P[-1]) > 1e-12:
        P = np.vstack([P, P[0]])
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    L = s[-1]
    st = np.linspace(0.0, L, M+1)[:-1]
    out = np.zeros((M, 2), float)
    j = 0
    for i, si in enumerate(st):
        while j < len(s)-2 and s[j+1] < si:
            j += 1
        denom = (s[j+1] - s[j]) if (s[j+1] - s[j]) != 0 else 1e-15
        out[i] = (1 - (si - s[j]) / denom) * P[j] + ((si - s[j]) / denom) * P[j+1]
    return out

def _best_cyclic_shift(A, B, allow_reverse=True):
    A, B = np.asarray(A, float), np.asarray(B, float)
    M = A.shape[0]
    def rmse(X):
        return np.sqrt(np.mean(np.sum((X - B)**2, axis=1)))
    best_rmse, best = np.inf, None
    for k in range(M):
        A2 = np.roll(A, k, axis=0)
        e = rmse(A2)
        if e < best_rmse:
            best_rmse, best = e, (A2, k, False, e)
    if allow_reverse:
        Arev = A[::-1].copy()
        for k in range(M):
            A2 = np.roll(Arev, k, axis=0)
            e = rmse(A2)
            if e < best_rmse:
                best_rmse, best = e, (A2, k, True, e)
    return best

def resample_exp_to_ref_arclength_plane(sec_expr_2d, sec_ref_5, plane="yz",
                                         knn_k=10, knn_lam=0.15,
                                         start_mode="max_first", allow_reverse=True):
    ref = np.asarray(sec_ref_5, float)
    exp = np.asarray(sec_expr_2d, float)
    M = ref.shape[0]
    ID_ref, theta_ref = ref[:, 0], ref[:, 4]
    if plane == "yz":
        ref_2d = ref[:, [2, 3]]
        def pack_back(e):
            return np.column_stack([ID_ref, ref[:, 1], e[:, 0], e[:, 1], theta_ref])
    elif plane == "xz":
        ref_2d = ref[:, [1, 3]]
        def pack_back(e):
            return np.column_stack([ID_ref, e[:, 0], ref[:, 2], e[:, 1], theta_ref])
    else:
        raise ValueError("plane must be 'yz' or 'xz'.")
    exp_ordered = _order_boundary_knn(exp, k=knn_k, lam=knn_lam, start_mode=start_mode)
    exp_rs = _resample_closed_curve_arclength(exp_ordered, M)
    exp_best, _, _, _ = _best_cyclic_shift(exp_rs, ref_2d, allow_reverse=allow_reverse)
    return pack_back(exp_best)

def sort_by_id_drop_theta(arr5):
    arr5 = np.asarray(arr5, float)
    return arr5[np.argsort(arr5[:, 0])][:, :4]

def merge_by_id_average(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    def compress(arr):
        ids = arr[:, 0].astype(np.int64)
        uniq, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
        if np.all(cnt == 1):
            return arr
        out = np.zeros((len(uniq), 4), float)
        out[:, 0] = uniq
        for c in range(1, 4):
            out[:, c] = np.bincount(inv, weights=arr[:, c]) / cnt
        return out
    A2, B2 = compress(A), compress(B)
    idA, idB = A2[:, 0].astype(np.int64), B2[:, 0].astype(np.int64)
    mapA = {int(i): A2[k, 1:4] for k, i in enumerate(idA)}
    mapB = {int(i): B2[k, 1:4] for k, i in enumerate(idB)}
    overlap = np.intersect1d(idA, idB)
    onlyA   = np.setdiff1d(idA, idB)
    onlyB   = np.setdiff1d(idB, idA)
    rows = []
    for i in overlap:
        rows.append([int(i), *(0.5 * (mapA[int(i)] + mapB[int(i)]))])
    for i in onlyA:
        rows.append([int(i), *mapA[int(i)]])
    for i in onlyB:
        rows.append([int(i), *mapB[int(i)]])
    merged = np.array(rows, float)
    return merged[np.argsort(merged[:, 0])]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams["font.family"] = "Times New Roman"
    image_path = ROOT_DIR / "In_image" / "Image_Disco.png"
    N_outline  = 2048
    Lx_um      = 8.78160
    Ly_um      = 2.73829
    fs         = 25

    STEPS = [
        "Part 1  cell segmentation (Chan-Vese)",
        "Part 2  concave ridge detection",
        "Part 3  build combined loop",
        "Part 4  smooth & upsample",
        "Save    D_p1234.png",
        "YZ      rotation + resample",
        "XZ      rotation + resample",
        "Merge   D_shown.png + npz files",
    ]

    with tqdm(total=len(STEPS), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
              ncols=72) as pbar:

        # ── Part 1 ──────────────────────────────────────────────────────────
        pbar.set_description(STEPS[0])
        img_u8, cell_mask = extract_mask_chanvese(
            image_path, sigma=2.0, iterations=200, smoothing=4)
        outline_ptsN, _ = extract_outer_outline_from_mask(cell_mask, N=N_outline)
        pbar.update(1)

        # ── Part 2 ──────────────────────────────────────────────────────────
        pbar.set_description(STEPS[1])
        H, W = cell_mask.shape
        yy, _ = np.where(cell_mask)
        y_c = int(np.round(yy.mean()))

        ridge_w = build_ridge_map(img_u8, cell_mask)

        avoid_boundary_r = 6
        core_mask = binary_erosion(cell_mask, disk(avoid_boundary_r))
        pad = 40

        upper_roi = core_mask.copy()
        upper_roi[min(H, y_c + pad):, :] = False
        upper_roi[H - int(0.20 * H):, :] = False

        upper_xy, _ = ridge_roi_to_best_curve(
            ridge_w, upper_roi, ridge_percentile=76, min_band_size=500, bridge_r=1)

        lower_roi = core_mask.copy()
        lower_roi[:max(0, y_c - pad), :] = False

        anchor_roi = lower_roi.copy()
        anchor_roi[:, :int(0.60 * W)] = False
        anchor_roi[:int(0.55 * H), :] = False

        def score_lower(comp_mask):
            return 1000.0 * float((comp_mask & anchor_roi).sum()) + float(comp_mask.sum())

        best_lower  = None
        best_metric = -1e18

        grow_percs = [76, 74, 72, 70, 68, 66]
        for perc in tqdm(grow_percs, desc="  lower grow", leave=False, ncols=52):
            lower_xy_try, _ = ridge_roi_to_best_curve(
                ridge_w, lower_roi, ridge_percentile=perc,
                min_band_size=250, bridge_r=2, score_fn=score_lower)
            if lower_xy_try is None:
                continue
            hits   = anchor_hits(lower_xy_try, anchor_roi, H, W)
            L      = _polyline_length(lower_xy_try)
            metric = 10000.0 * hits + L
            if metric > best_metric:
                best_metric = metric
                best_lower  = lower_xy_try

        lower_xy = best_lower

        if upper_xy is None:
            raise RuntimeError("Upper concave not found.")
        if lower_xy is None:
            raise RuntimeError("Lower concave not found.")

        N_concave  = 1024
        upper_ptsN = resample_open_polyline(upper_xy, N_concave)
        lower_ptsN = resample_open_polyline(lower_xy, N_concave)
        pbar.update(1)

        # ── Part 3 ──────────────────────────────────────────────────────────
        pbar.set_description(STEPS[2])
        outline_xy = np.asarray(outline_ptsN, float)
        upper_xy   = np.asarray(upper_ptsN, float)
        lower_xy   = np.asarray(lower_ptsN, float)

        if upper_xy[0,0] > upper_xy[-1,0]:
            upper_xy = upper_xy[::-1]
        if lower_xy[0,0] > lower_xy[-1,0]:
            lower_xy = lower_xy[::-1]

        uL, uR = upper_xy[0], upper_xy[-1]
        lL, lR = lower_xy[0], lower_xy[-1]

        i_uR = _nearest_outline_index(outline_xy, uR)
        i_lR = _nearest_outline_index(outline_xy, lR)
        arc1, arc2 = _get_two_arcs_oriented_iA_to_iB(outline_xy, i_uR, i_lR)
        x_center = float(np.mean(outline_xy[:,0]))
        right_arc = _choose_arc_by_side(arc1, arc2, x_center, prefer="right")

        d_left = float(np.linalg.norm(uL - lL))
        if d_left <= 40.0:
            left_conn = np.vstack([lL, uL])
        else:
            i_lL = _nearest_outline_index(outline_xy, lL)
            i_uL = _nearest_outline_index(outline_xy, uL)
            a1, a2 = _get_two_arcs_oriented_iA_to_iB(outline_xy, i_lL, i_uL)
            left_conn = _choose_arc_by_side(a1, a2, x_center, prefer="left")

        combined_loop_xy = _ensure_implicit_closed(np.vstack([
            upper_xy, right_arc[1:], lower_xy[::-1], left_conn[1:]
        ]))
        pbar.update(1)

        # ── Part 4 ──────────────────────────────────────────────────────────
        pbar.set_description(STEPS[3])
        loop_xy_samples, _ = smooth_then_upsample_closed_loop(
            combined_loop_xy, N_smooth=128, N_samples=2048,
            movavg_win=13, movavg_passes=2, post_win=7, post_passes=1)
        loop_phys_um = normalize_and_scale_to_physical(loop_xy_samples, Lx_um=Lx_um, Ly_um=Ly_um)
        pbar.update(1)

        # ── Save D_p1234.png ─────────────────────────────────────────────────
        pbar.set_description(STEPS[4])
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.imshow(img_u8, cmap="gray", origin="upper")
        ax.plot(outline_ptsN[:, 0], outline_ptsN[:, 1], lw=1.8, label="outer outline", c='b')
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.legend(prop={"family": "Times New Roman", "size": 16})

        ax = axes[0, 1]
        ax.imshow(img_u8, cmap="gray", origin="upper")
        ax.plot(outline_ptsN[:,0], outline_ptsN[:,1], lw=1.6, c="b",     label="outer outline")
        ax.plot(upper_ptsN[:,0],   upper_ptsN[:,1],   lw=2.2, c="green", label="concave 1")
        ax.plot(lower_ptsN[:,0],   lower_ptsN[:,1],   lw=2.2, c="lime",  label="concave 2")
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(prop={"family": "Times New Roman", "size": 16})

        ax = axes[1, 0]
        ax.imshow(img_u8, cmap="gray", origin="upper")
        ax.plot(combined_loop_xy[:,0], combined_loop_xy[:,1], lw=3.0, c="red", label="combined loop")
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(prop={"family": "Times New Roman", "size": 16})

        ax = axes[1, 1]
        ax.imshow(img_u8, cmap="gray", origin="upper")
        ax.plot(loop_xy_samples[:,0], loop_xy_samples[:,1], lw=2.6, label="smoothed loop", c="orange")
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(prop={"family": "Times New Roman", "size": 16})

        plt.tight_layout()
        plt.savefig(ROOT_DIR / "Out_graph" / "D_p1234.png", dpi=300, bbox_inches="tight")
        plt.close()
        pbar.update(1)

        # shared alignment inputs
        ref_yz_info = np.load(ROOT_DIR / "In_DPDref" / "D_A1.05_hf_partial_x.npz")
        ref_xz_info = np.load(ROOT_DIR / "In_DPDref" / "D_A1.05_hf_partial_y.npz")
        image_xy = loop_phys_um

        # ── YZ ───────────────────────────────────────────────────────────────
        pbar.set_description(STEPS[5])
        sec_ref_yz = add_theta_and_sort_plane(ref_yz_info["rbc_p"], plane="yz")
        sec_expr_yz, _, _ = align_exp_to_ref_by_rotation_2d(
            image_xy[:, [0,1]], sec_ref_yz[:, [2,3]])
        sec_exp2_yz = resample_exp_to_ref_arclength_plane(
            sec_expr_yz, sec_ref_yz, plane="yz")
        sec_exp2_yz_final = sort_by_id_drop_theta(sec_exp2_yz)
        np.savez(ROOT_DIR / "Out_data4MFNN" / "D_exp_partial_yz.npz",
                 sphere_p=ref_yz_info["sphere_p"], rbc_p=sec_exp2_yz_final,
                 sphere_r=ref_yz_info["sphere_r"], rbc_r=ref_yz_info["rbc_r"])
        pbar.update(1)

        # ── XZ ───────────────────────────────────────────────────────────────
        pbar.set_description(STEPS[6])
        sec_ref_xz = add_theta_and_sort_plane(ref_xz_info["rbc_p"], plane="xz")
        sec_expr_xz, _, _ = align_exp_to_ref_by_rotation_2d(
            image_xy[:, [0,1]], sec_ref_xz[:, [1,3]])
        sec_exp2_xz = resample_exp_to_ref_arclength_plane(
            sec_expr_xz, sec_ref_xz, plane="xz")
        sec_exp2_xz_final = sort_by_id_drop_theta(sec_exp2_xz)
        np.savez(ROOT_DIR / "Out_data4MFNN" / "D_exp_partial_xz.npz",
                 sphere_p=ref_xz_info["sphere_p"], rbc_p=sec_exp2_xz_final,
                 sphere_r=ref_xz_info["sphere_r"], rbc_r=ref_xz_info["rbc_r"])
        pbar.update(1)

        # ── Merge + D_shown.png ───────────────────────────────────────────────
        pbar.set_description(STEPS[7])
        exp_merged = merge_by_id_average(sec_exp2_yz_final, sec_exp2_xz_final)
        r0_merged  = merge_by_id_average(ref_yz_info["sphere_p"], ref_xz_info["sphere_p"])
        np.savez(ROOT_DIR / "Out_data4MFNN" / "D_exp_partial_xz+yz.npz",
                 sphere_p=r0_merged, rbc_p=exp_merged,
                 sphere_r=ref_yz_info["sphere_r"], rbc_r=ref_yz_info["rbc_r"])

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        ax = axes[0]
        yz_closed = np.vstack([sec_expr_yz, sec_expr_yz[0]])
        ax.plot(yz_closed[:,0], yz_closed[:,1], c="orange", linewidth=2.5,
                label="extracted smoothed loop", zorder=-1)
        ax.scatter(sec_ref_yz[:,2], sec_ref_yz[:,3], s=15, c="blue",
                   label="simualtion reference samples")
        ax.scatter(sec_exp2_yz[:,2], sec_exp2_yz[:,3], s=15, c="red",
                   label="pair-wised samples")
        for i in range(sec_ref_yz.shape[0]):
            ax.plot([sec_ref_yz[i,2], sec_exp2_yz[i,2]],
                    [sec_ref_yz[i,3], sec_exp2_yz[i,3]], 'k-', lw=1.0, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(prop={"family": "Times New Roman", "size": fs})
        ax.set_xlabel("y (um)", fontsize=fs)
        ax.set_ylabel("z (um)", fontsize=fs)
        ax.tick_params(labelsize=fs)

        ax = axes[1]
        xz_closed = np.vstack([sec_expr_xz, sec_expr_xz[0]])
        ax.plot(xz_closed[:,0], xz_closed[:,1], c="orange", linewidth=2.5,
                label="extracted smoothed loop", zorder=-1)
        ax.scatter(sec_ref_xz[:,1], sec_ref_xz[:,3], s=15, c="blue",
                   label="simualtion reference samples")
        ax.scatter(sec_exp2_xz[:,1], sec_exp2_xz[:,3], s=15, c="red",
                   label="pair-wised samples")
        for i in range(sec_ref_xz.shape[0]):
            ax.plot([sec_ref_xz[i,1], sec_exp2_xz[i,1]],
                    [sec_ref_xz[i,3], sec_exp2_xz[i,3]], 'k-', lw=1.0, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(prop={"family": "Times New Roman", "size": fs})
        ax.set_xlabel("x (um)", fontsize=fs)
        ax.set_ylabel("z (um)", fontsize=fs)
        ax.tick_params(labelsize=fs)

        plt.tight_layout()
        plt.savefig(ROOT_DIR / "Out_graph" / "D_shown.png", dpi=300, bbox_inches="tight")
        plt.close()
        pbar.update(1)


if __name__ == "__main__":
    main()
