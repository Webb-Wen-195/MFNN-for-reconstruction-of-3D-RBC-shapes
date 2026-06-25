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
    skeletonize, remove_small_objects, remove_small_holes,
    binary_opening, binary_closing, binary_dilation, disk
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

def extract_concave_loop_from_ridge_v2(
    img_u8, cell_mask,
    use_equalize=True, sigma_pre=1.0,
    ridge_sigmas=(2,4,6,8,10,12),
    alpha=2.5, boundary_exclude_px=18,
    high_pct=97, low_pct=90,
    close_r=6, open_r=1, min_obj=600, hole_area=2500,
    prefer_deep=True
):
    img_f = img_u8.astype(np.float32) / 255.0
    if use_equalize:
        img_f = exposure.equalize_adapthist(img_f, clip_limit=0.02)
    if sigma_pre and sigma_pre > 0:
        img_f = gaussian(img_f, sigma=sigma_pre, preserve_range=True)

    ridge = sato(img_f, sigmas=list(ridge_sigmas), black_ridges=True)
    ridge[~cell_mask] = 0.0

    from scipy.ndimage import distance_transform_edt
    dist_in = distance_transform_edt(cell_mask)
    dist_norm = dist_in / (dist_in.max() + 1e-12)
    ridge_w = ridge * (dist_norm ** alpha)
    ridge_w[~cell_mask] = 0.0

    inner_roi = cell_mask & (dist_in >= boundary_exclude_px)
    if inner_roi.sum() < 2000:
        inner_roi = cell_mask & (dist_in >= max(5, boundary_exclude_px // 2))
    if inner_roi.sum() == 0:
        return None, dict(ridge=ridge, ridge_w=ridge_w, dist_in=dist_in)

    vals = ridge_w[inner_roi]
    hi = np.percentile(vals, high_pct)
    lo = np.percentile(vals, low_pct)

    seed = (ridge_w >= hi) & inner_roi
    cand = (ridge_w >= lo) & inner_roi
    lab_cand = label(cand)
    seed_labels = np.unique(lab_cand[seed])
    seed_labels = seed_labels[seed_labels != 0]
    if len(seed_labels) == 0:
        return None, dict(ridge=ridge, ridge_w=ridge_w, dist_in=dist_in,
                          inner_roi=inner_roi, seed=seed, band=None)

    band = np.isin(lab_cand, seed_labels)
    if open_r and open_r > 0:
        band = binary_opening(band, disk(open_r))
    if close_r and close_r > 0:
        band = binary_closing(band, disk(close_r))
    band = remove_small_objects(band, min_size=min_obj)
    band = remove_small_holes(band, area_threshold=hole_area)

    lab = label(band)
    if lab.max() == 0:
        return None, dict(ridge=ridge, ridge_w=ridge_w, dist_in=dist_in,
                          inner_roi=inner_roi, seed=seed, band=band)

    props = regionprops(lab, intensity_image=ridge_w)
    best_label = None
    best_score = -1
    for r in props:
        comp = (lab == r.label)
        q90 = float(np.percentile(dist_in[comp], 90))
        strength = float(r.mean_intensity)
        area = float(r.area)
        score = (q90 ** 2.0) * (strength + 1e-12) * (area ** 0.25) if prefer_deep else strength * area
        if score > best_score:
            best_score = score
            best_label = r.label

    comp = (lab == best_label)
    contours = find_contours(comp.astype(np.float32), 0.5)
    if not contours:
        return None, dict(ridge=ridge, ridge_w=ridge_w, dist_in=dist_in,
                          inner_roi=inner_roi, seed=seed, band=band, comp=comp)
    c = max(contours, key=lambda a: a.shape[0])
    loop_xy = np.column_stack([c[:,1], c[:,0]])
    return loop_xy, dict(ridge=ridge, ridge_w=ridge_w, dist_in=dist_in,
                         inner_roi=inner_roi, seed=seed, band=band, comp=comp,
                         thresholds=(lo, hi), alpha=alpha)

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

def _prune_spurs(skel, iters=15):
    sk = skel.copy()
    for _ in range(iters):
        ep = _endpoints_of_skeleton(sk)
        if len(ep) == 0:
            break
        sk[ep[:,0], ep[:,1]] = False
    return sk

def extract_single_centerline_from_band(band_mask, ridge_weighted=None,
                                         spur_prune_iters=18, bridge_dilate_r=0, eps=1e-6):
    skel = skeletonize(band_mask)
    if bridge_dilate_r and bridge_dilate_r > 0:
        skel = skeletonize(binary_dilation(skel, disk(bridge_dilate_r)))

    sk_pruned = _prune_spurs(skel, iters=spur_prune_iters)
    if sk_pruned.sum() < 30:
        sk_pruned = skel

    lab = label(sk_pruned)
    if lab.max() > 1:
        regs = regionprops(lab)
        keep = max(regs, key=lambda r: r.area).label
        sk_pruned = (lab == keep)

    endpoints = _endpoints_of_skeleton(sk_pruned)
    if len(endpoints) < 2:
        ys, xs = np.where(sk_pruned)
        pts = np.column_stack([ys, xs])
        if len(pts) < 10:
            return None, {}
        a_rc, b_rc = _farthest_pair(pts)
    else:
        a_rc, b_rc = _farthest_pair(endpoints)

    if ridge_weighted is None:
        cost = np.ones_like(sk_pruned, dtype=np.float32) * 10.0
        cost[sk_pruned] = 1.0
    else:
        cost = 1.0 / (ridge_weighted + eps)
        cost += (~sk_pruned).astype(np.float32) * 8.0

    path_rc, _ = route_through_array(cost, start=a_rc, end=b_rc, fully_connected=True, geometric=True)
    path_rc = np.array(path_rc)
    return np.column_stack([path_rc[:,1], path_rc[:,0]]), {}

def _xy_to_rc(pt_xy):
    return (int(round(pt_xy[1])), int(round(pt_xy[0])))

def _rc_path_to_xy(path_rc):
    path_rc = np.asarray(path_rc)
    return np.column_stack([path_rc[:, 1], path_rc[:, 0]]).astype(float)

def _nearest_outline_index(outline_xy, p_xy):
    return int(np.argmin(np.sum((outline_xy - p_xy)**2, axis=1)))

def _arc_indices(iA, iB, N):
    if iA <= iB:
        return np.arange(iA, iB + 1)
    return np.concatenate([np.arange(iA, N), np.arange(0, iB + 1)])

def _arclen(P):
    P = np.asarray(P, float)
    if len(P) < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.sum((P[1:] - P[:-1])**2, axis=1))))

def _drop_consecutive_duplicates(P, tol2=1e-9):
    P = np.asarray(P, float)
    if len(P) <= 1:
        return P
    keep = np.ones(len(P), dtype=bool)
    keep[1:] = np.sum((P[1:] - P[:-1])**2, axis=1) > tol2
    return P[keep]

def build_closed_loop_with_given_attach_indices(outline_xy, concave_ext_xy, i0, i1, arc_mode="longer"):
    outline_xy = np.asarray(outline_xy, float)
    N = outline_xy.shape[0]
    arc_fwd = outline_xy[_arc_indices(i0, i1, N)]
    arc_bwd = outline_xy[_arc_indices(i1, i0, N)][::-1]
    Lf, Lb = _arclen(arc_fwd), _arclen(arc_bwd)
    outer_arc = arc_fwd if (Lf >= Lb) == (arc_mode == "longer") else arc_bwd
    loop_xy = _drop_consecutive_duplicates(np.vstack([outer_arc, np.asarray(concave_ext_xy, float)[::-1]]))
    if np.sum((loop_xy[0] - loop_xy[-1])**2) < 1e-9:
        loop_xy = loop_xy[:-1]
    return loop_xy

def choose_attach_indices_controllable(outline_xy, concave_xy,
                                        search_window=140, left_shift=60,
                                        right_shift=60, prefer_upper=False):
    outline_xy = np.asarray(outline_xy, float)
    concave_xy = np.asarray(concave_xy, float)
    N = len(outline_xy)
    y_med = np.median(outline_xy[:, 1])

    def local_best(endpoint_xy):
        j0 = _nearest_outline_index(outline_xy, endpoint_xy)
        best, best_d2 = j0, 1e18
        for dk in range(-search_window, search_window + 1):
            j = (j0 + dk) % N
            if prefer_upper and outline_xy[j, 1] > y_med:
                continue
            d2 = np.sum((outline_xy[j] - endpoint_xy)**2)
            if d2 < best_d2:
                best_d2 = d2
                best = j
        return best

    jL = local_best(concave_xy[0])
    jR = local_best(concave_xy[-1])
    if outline_xy[jL, 0] > outline_xy[jR, 0]:
        jL, jR = jR, jL
    i0 = (jL - int(abs(left_shift))) % N
    i1 = (jR + int(abs(right_shift))) % N
    if outline_xy[i0, 0] > outline_xy[i1, 0]:
        i0, i1 = i1, i0
    return int(i0), int(i1), dict(jL=jL, jR=jR)

def extend_concave_to_outline_given_indices(concave_xy, outline_xy, i0, i1,
                                             cell_mask, ridge_w,
                                             corridor_mask=None, penalty_outside=400.0, eps=1e-6):
    concave_xy = np.asarray(concave_xy, float)
    outline_xy = np.asarray(outline_xy, float)
    p0_rc = _xy_to_rc(concave_xy[0])
    p1_rc = _xy_to_rc(concave_xy[-1])
    a0_rc = _xy_to_rc(outline_xy[i0])
    a1_rc = _xy_to_rc(outline_xy[i1])

    cost = (1.0 / (ridge_w + eps)).astype(np.float32)
    cost[~cell_mask] = 1e6
    if corridor_mask is not None:
        cost = cost + (~corridor_mask).astype(np.float32) * float(penalty_outside)

    path0_rc, _ = route_through_array(cost, start=p0_rc, end=a0_rc, fully_connected=True, geometric=True)
    path1_rc, _ = route_through_array(cost, start=p1_rc, end=a1_rc, fully_connected=True, geometric=True)

    conn0_xy = _rc_path_to_xy(path0_rc)
    conn1_xy = _rc_path_to_xy(path1_rc)
    concave_ext_xy = _drop_consecutive_duplicates(
        np.vstack([conn0_xy[::-1], concave_xy[1:-1], conn1_xy]))
    return concave_ext_xy, dict(ok=True, method="outline", i0=i0, i1=i1,
                                attach0_xy=outline_xy[i0], attach1_xy=outline_xy[i1])

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
    t = np.linspace(0.0, s[-1], int(N_assign) + 1)[:-1]
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


# ── Redense helpers ───────────────────────────────────────────────────────────

def resample_closed_loop_xy(xy, n):
    xy = np.asarray(xy, dtype=float)
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0]]
    if xy.shape[0] < 3:
        raise ValueError("Need at least 3 unique points.")
    xyc = np.vstack([xy, xy[0]])
    seglen = np.linalg.norm(np.diff(xyc, axis=0), axis=1)
    L = float(seglen.sum())
    if L <= 0:
        raise ValueError("Loop length is zero.")
    s = np.r_[0.0, np.cumsum(seglen)]
    good = np.r_[True, np.diff(s) > 0]
    s, xyc = s[good], xyc[good]
    s_new = np.linspace(0.0, L, n + 1)[:-1]
    return np.column_stack([np.interp(s_new, s, xyc[:, 0]),
                            np.interp(s_new, s, xyc[:, 1])])

def _contiguous_true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs = []
    start = prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append((start, prev + 1))
            start = prev = k
    runs.append((start, prev + 1))
    if runs and runs[0][0] == 0 and runs[-1][1] == n and mask[0] and mask[-1]:
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return runs

def _collapse_duplicate_x(x, y, xbin=None, mode="median"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if xbin is None:
        xbin = max((np.max(x) - np.min(x)) / 300.0, 1e-6)
    xb = np.round(x / xbin) * xbin
    order = np.argsort(xb)
    xb, y = xb[order], y[order]
    uniq = np.unique(xb)
    yu = np.array([np.median(y[xb == xu]) if mode == "median"
                   else np.mean(y[xb == xu]) for xu in uniq])
    return uniq, yu

def _resample_poly_by_arclength(poly, x0, x1, n, endpoint_points=None, oversample=4000):
    xd = np.linspace(float(x0), float(x1), oversample)
    yd = np.polyval(poly, xd)
    dx, dy = np.diff(xd), np.diff(yd)
    s = np.concatenate([[0.0], np.cumsum(np.sqrt(dx*dx + dy*dy))])
    L = s[-1]
    if L <= 0:
        xs = np.linspace(x0, x1, n)
        pts = np.column_stack([xs, np.polyval(poly, xs)])
    else:
        s_new = np.linspace(0.0, L, n)
        xs = np.interp(s_new, s, xd)
        pts = np.column_stack([xs, np.polyval(poly, xs)])
    if endpoint_points is not None:
        pts[0], pts[-1] = endpoint_points
    return pts

def fix_overlap_region_by_poly(xy, x_min=-1.5, x_max=1.6, y_max=0.3,
                                deg=5, collapse_mode="median",
                                keep_n="same", xbin=None, choose_run="largest"):
    xy = np.asarray(xy, float)
    N = xy.shape[0]
    x, y = xy[:, 0], xy[:, 1]
    region_mask = (x > x_min) & (x < x_max) & (y < y_max)
    runs = _contiguous_true_runs(region_mask)
    if not runs:
        return xy.copy()
    if choose_run == "largest":
        run = max(runs, key=lambda ab: (ab[1]-ab[0]) if ab[0] < ab[1] else (N - ab[0] + ab[1]))
    else:
        run = runs[0]
    a, b = run
    seg_idx = np.arange(a, b) if a < b else np.concatenate([np.arange(a, N), np.arange(0, b)])
    seg = xy[seg_idx]
    xs, ys = seg[:, 0], seg[:, 1]
    i_min, i_max = int(np.argmin(xs)), int(np.argmax(xs))
    p_min, p_max = seg[i_min].copy(), seg[i_max].copy()
    x_u, y_u = _collapse_duplicate_x(xs, ys, xbin=xbin, mode=collapse_mode)
    if x_u.size < 3:
        return xy.copy()
    poly = np.polyfit(x_u, y_u, deg=min(int(deg), x_u.size - 1))
    n_seg = max(seg.shape[0] if keep_n == "same" else int(keep_n), 3)
    x0, x1 = p_min[0], p_max[0]
    if x1 < x0:
        x0, x1, p_min, p_max = x1, x0, p_max, p_min
    seg_new = _resample_poly_by_arclength(poly, x0, x1, n_seg, endpoint_points=(p_min, p_max))
    if np.linalg.norm(seg[0] - seg_new[0]) > np.linalg.norm(seg[0] - seg_new[-1]):
        seg_new = seg_new[::-1]
    xy_new = np.delete(xy.copy(), seg_idx, axis=0)
    return np.insert(xy_new, a, seg_new, axis=0)


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

def _resample_closed_curve_arclength(ordered2d, M):
    P = np.asarray(ordered2d, float)
    if np.linalg.norm(P[0] - P[-1]) > 1e-12:
        P = np.vstack([P, P[0]])
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    L = s[-1]
    if not np.isfinite(L) or L <= 0:
        raise ValueError("Curve length is invalid.")
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

def resample_exp_to_ref_arclength_plane(exp_2d, sec_ref_5, plane="yz", allow_reverse=True):
    ref = np.asarray(sec_ref_5, float)
    exp = np.asarray(exp_2d, float)
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
    exp_rs = _resample_closed_curve_arclength(exp, M)
    exp_best, _, _, _ = _best_cyclic_shift(exp_rs, ref_2d, allow_reverse=allow_reverse)
    return pack_back(exp_best)

def sort_by_id_drop_theta(arr5):
    arr5 = np.asarray(arr5, float)
    return arr5[np.argsort(arr5[:, 0])][:, :4]


# ── 2-opt loop ordering ───────────────────────────────────────────────────────

def _segments_intersect(a, b, c, d):
    def ccw(p, q, r):
        return (r[1]-p[1]) * (q[0]-p[0]) > (q[1]-p[1]) * (r[0]-p[0])
    return ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d)

def _two_opt(route):
    P = route.copy()
    N = len(P)
    improved = True
    while improved:
        improved = False
        for i in range(N - 1):
            a, b = P[i], P[(i+1) % N]
            for j in range(i+2, N-1):
                c, d = P[j], P[(j+1) % N]
                if _segments_intersect(a, b, c, d):
                    P[i+1:j+1] = P[i+1:j+1][::-1]
                    improved = True
    return P

def order_loop_nn_2opt(points, start_idx=None):
    P = np.asarray(points, float)
    N = len(P)
    if start_idx is None:
        start_idx = int(np.lexsort((P[:,1], P[:,0]))[0])
    used = np.zeros(N, dtype=bool)
    order = [start_idx]
    used[start_idx] = True
    for _ in range(N - 1):
        last = P[order[-1]]
        candidates = np.where(~used)[0]
        d2 = np.sum((P[candidates] - last)**2, axis=1)
        order.append(int(candidates[np.argmin(d2)]))
        used[order[-1]] = True
    return _two_opt(P[order])

def polygon_signed_area(loop):
    P = np.asarray(loop, float)
    x, y = P[:,0], P[:,1]
    return 0.5 * np.sum(x * np.roll(y,-1) - y * np.roll(x,-1))

def ensure_same_orientation(ref_loop, exp_loop):
    if np.sign(polygon_signed_area(ref_loop)) != np.sign(polygon_signed_area(exp_loop)):
        exp_loop = exp_loop[::-1]
    return exp_loop

def anchor_bottom(points):
    P = np.asarray(points, float)
    idx = int(np.lexsort((P[:,0], P[:,1]))[0])
    return np.roll(P, -idx, axis=0)

def best_shift_exp_to_ref(ref, exp):
    ref, exp = np.asarray(ref, float), np.asarray(exp, float)
    N = len(ref)
    best_k, best_sse = 0, np.inf
    for k in range(N):
        e = np.roll(exp, -k, axis=0)
        sse = float(np.sum((ref - e)**2))
        if sse < best_sse:
            best_sse, best_k = sse, k
    return np.roll(exp, -best_k, axis=0), best_k


# ── Merge ─────────────────────────────────────────────────────────────────────

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
    rows = []
    for i in np.intersect1d(idA, idB):
        rows.append([int(i), *(0.5 * (mapA[int(i)] + mapB[int(i)]))])
    for i in np.setdiff1d(idA, idB):
        rows.append([int(i), *mapA[int(i)]])
    for i in np.setdiff1d(idB, idA):
        rows.append([int(i), *mapB[int(i)]])
    merged = np.array(rows, float)
    return merged[np.argsort(merged[:, 0])]


# ── ID-lookup helpers ─────────────────────────────────────────────────────────

def _match_ids_yz(ref_ord, exp_paired, sec_ref_yz):
    sec_exp3 = []
    for refline, exp_line in zip(ref_ord, exp_paired):
        ref_y, ref_z = refline[0], refline[1]
        exp_y, exp_z = exp_line[0], exp_line[1]
        for infoline in sec_ref_yz:
            ID, x, y, z, theta = infoline
            if np.round(ref_y, 8) == np.round(y, 8) and np.round(ref_z, 8) == np.round(z, 8):
                sec_exp3.append([ID, x, exp_y, exp_z, theta])
                break
    sec_exp3 = np.array(sec_exp3)
    return sec_exp3[np.argsort(sec_exp3[:, 0].astype(np.int64))]

def _match_ids_xz(ref_ord, exp_paired, sec_ref_xz):
    sec_exp3 = []
    for refline, exp_line in zip(ref_ord, exp_paired):
        ref_x, ref_z = refline[0], refline[1]
        exp_x, exp_z = exp_line[0], exp_line[1]
        for infoline in sec_ref_xz:
            ID, x, y, z, theta = infoline
            if np.round(ref_x, 8) == np.round(x, 8) and np.round(ref_z, 8) == np.round(z, 8):
                sec_exp3.append([ID, exp_x, y, exp_z, theta])
                break
    sec_exp3 = np.array(sec_exp3)
    return sec_exp3[np.argsort(sec_exp3[:, 0].astype(np.int64))]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams["font.family"] = "Times New Roman"
    image_path = ROOT_DIR / "In_image" / "Image_Stomato.png"
    N_outline  = 2048
    Lx_um      = 7.87860
    Ly_um      = 4.14557
    fs         = 25
    N_re       = 512

    STEPS = [
        "Part 1  cell segmentation (Chan-Vese)",
        "Part 2  ridge concave extraction",
        "Part 3  build combined loop",
        "Part 4  smooth & upsample",
        "Save    S_p1234.png",
        "Redense + YZ  alignment → S_exp_partial_yz.npz",
        "XZ      alignment → S_exp_partial_xz.npz",
        "Merge   S_shown.png + S_exp_partial_xz+yz.npz",
    ]

    with tqdm(total=len(STEPS), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
              ncols=72) as pbar:

        # ── Part 1 ──────────────────────────────────────────────────────────
        pbar.set_description(STEPS[0])
        img_u8, cell_mask = extract_mask_chanvese(
            image_path, sigma=2.0, iterations=400, smoothing=4)
        outline_ptsN, _ = extract_outer_outline_from_mask(cell_mask, N=N_outline)
        pbar.update(1)

        # ── Part 2 ──────────────────────────────────────────────────────────
        pbar.set_description(STEPS[1])
        _, dbg = extract_concave_loop_from_ridge_v2(img_u8, cell_mask)
        band    = dbg["comp"]
        ridge_w = dbg["ridge_w"]

        center_xy, _ = extract_single_centerline_from_band(
            band, ridge_weighted=ridge_w, spur_prune_iters=18, bridge_dilate_r=1)
        pbar.update(1)

        # ── Part 3 ──────────────────────────────────────────────────────────
        pbar.set_description(STEPS[2])
        i0, i1, _ = choose_attach_indices_controllable(
            outline_ptsN, center_xy,
            search_window=160, left_shift=40, right_shift=60, prefer_upper=False)

        H, W = cell_mask.shape
        conc_mask = np.zeros((H, W), dtype=bool)
        cc = np.clip(np.round(center_xy).astype(int), [0, 0], [W-1, H-1])
        conc_mask[cc[:,1], cc[:,0]] = True
        corridor = (binary_dilation(band, disk(22)) | binary_dilation(conc_mask, disk(18))) & cell_mask

        concave_ext_xy, attach = extend_concave_to_outline_given_indices(
            concave_xy=center_xy, outline_xy=outline_ptsN,
            i0=i0, i1=i1, cell_mask=cell_mask, ridge_w=ridge_w,
            corridor_mask=corridor, penalty_outside=600.0)

        combined_loop_xy = build_closed_loop_with_given_attach_indices(
            outline_ptsN, concave_ext_xy, attach["i0"], attach["i1"], arc_mode="longer")
        pbar.update(1)

        # ── Part 4 ──────────────────────────────────────────────────────────
        pbar.set_description(STEPS[3])
        loop_xy_samples, _ = smooth_then_upsample_closed_loop(
            combined_loop_xy, N_smooth=128, N_samples=2048,
            movavg_win=13, movavg_passes=2, post_win=7, post_passes=1)
        loop_phys_um = normalize_and_scale_to_physical(loop_xy_samples, Lx_um=Lx_um, Ly_um=Ly_um)
        pbar.update(1)

        # ── Save S_p1234.png ─────────────────────────────────────────────────
        pbar.set_description(STEPS[4])
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.imshow(img_u8, cmap="gray", origin="upper")
        ax.plot(outline_ptsN[:, 0], outline_ptsN[:, 1], lw=1.8, label="outer outline", c='b')
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.legend(prop={"family": "Times New Roman", "size": 16})

        ax = axes[0, 1]
        ax.imshow(img_u8, cmap="gray", origin="upper")
        ax.plot(outline_ptsN[:,0], outline_ptsN[:,1], lw=1.8, c="b",   label="outer outline")
        ax.plot(center_xy[:,0],    center_xy[:,1],    lw=2.4, c="g",   label="concave")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.legend(prop={"family": "Times New Roman", "size": 16})

        ax = axes[1, 0]
        ax.imshow(img_u8, cmap="gray", origin="upper")
        ax.plot(combined_loop_xy[:,0], combined_loop_xy[:,1], lw=2.8, c="r", label="combined loop")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.legend(prop={"family": "Times New Roman", "size": 16})

        ax = axes[1, 1]
        ax.imshow(img_u8, cmap="gray", origin="upper")
        ax.plot(loop_xy_samples[:,0], loop_xy_samples[:,1], lw=2.5, c="orange", label="smoothed loop")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.legend(prop={"family": "Times New Roman", "size": 16})

        plt.tight_layout()
        plt.savefig(ROOT_DIR / "Out_graph" / "S_p1234.png", dpi=300, bbox_inches="tight")
        plt.close()
        pbar.update(1)

        # shared alignment inputs
        ref_yz_info = np.load(ROOT_DIR / "In_DPDref" / "S_mu200_hf_partial_x.npz")
        ref_xz_info = np.load(ROOT_DIR / "In_DPDref" / "S_mu200_hf_partial_y.npz")

        # ── Redense + YZ ─────────────────────────────────────────────────────
        pbar.set_description(STEPS[5])
        xy_fixed   = fix_overlap_region_by_poly(
            loop_phys_um, x_min=-1.5, x_max=1.6, y_max=0.3,
            deg=5, collapse_mode="median", keep_n="same")
        xy_uniform = resample_closed_loop_xy(xy_fixed, n=N_re)

        sec_ref_yz = add_theta_and_sort_plane(ref_yz_info["rbc_p"], plane="yz")
        sec_expr_yz, _, _ = align_exp_to_ref_by_rotation_2d(
            xy_uniform[:, [0,1]], sec_ref_yz[:, [2,3]])
        sec_exp2_yz = resample_exp_to_ref_arclength_plane(
            sec_expr_yz, sec_ref_yz, plane="yz")

        reference_yz  = sec_ref_yz[:, [2,3]]
        experiment_yz = sec_exp2_yz[:, [2,3]]
        ref_ord_yz = anchor_bottom(order_loop_nn_2opt(reference_yz))
        exp_ord_yz = anchor_bottom(order_loop_nn_2opt(experiment_yz))
        exp_ord_yz = ensure_same_orientation(ref_ord_yz, exp_ord_yz)
        exp_yz_new, _ = best_shift_exp_to_ref(ref_ord_yz, exp_ord_yz)

        sec_exp4_yz = _match_ids_yz(ref_ord_yz, exp_yz_new, sec_ref_yz)
        sec_exp2_yz_final = sort_by_id_drop_theta(sec_exp4_yz)
        np.savez(ROOT_DIR / "Out_data4MFNN" / "S_exp_partial_yz.npz",
                 sphere_p=ref_yz_info["sphere_p"], rbc_p=sec_exp2_yz_final,
                 sphere_r=ref_yz_info["sphere_r"], rbc_r=ref_yz_info["rbc_r"])
        pbar.update(1)

        # ── XZ ───────────────────────────────────────────────────────────────
        pbar.set_description(STEPS[6])
        sec_ref_xz = add_theta_and_sort_plane(ref_xz_info["rbc_p"], plane="xz")
        sec_expr_xz, _, _ = align_exp_to_ref_by_rotation_2d(
            xy_uniform[:, [0,1]], sec_ref_xz[:, [1,3]])
        sec_exp2_xz = resample_exp_to_ref_arclength_plane(
            sec_expr_xz, sec_ref_xz, plane="xz")

        reference_xz  = sec_ref_xz[:, [1,3]]
        experiment_xz = sec_exp2_xz[:, [1,3]]
        ref_ord_xz = anchor_bottom(order_loop_nn_2opt(reference_xz))
        exp_ord_xz = anchor_bottom(order_loop_nn_2opt(experiment_xz))
        exp_ord_xz = ensure_same_orientation(ref_ord_xz, exp_ord_xz)
        exp_xz_new, _ = best_shift_exp_to_ref(ref_ord_xz, exp_ord_xz)

        sec_exp4_xz = _match_ids_xz(ref_ord_xz, exp_xz_new, sec_ref_xz)
        sec_exp2_xz_final = sort_by_id_drop_theta(sec_exp4_xz)
        np.savez(ROOT_DIR / "Out_data4MFNN" / "S_exp_partial_xz.npz",
                 sphere_p=ref_xz_info["sphere_p"], rbc_p=sec_exp2_xz_final,
                 sphere_r=ref_xz_info["sphere_r"], rbc_r=ref_xz_info["rbc_r"])
        pbar.update(1)

        # ── Merge + S_shown.png ───────────────────────────────────────────────
        pbar.set_description(STEPS[7])
        exp_merged = merge_by_id_average(sec_exp2_yz_final, sec_exp2_xz_final)
        r0_merged  = merge_by_id_average(ref_yz_info["sphere_p"], ref_xz_info["sphere_p"])
        np.savez(ROOT_DIR / "Out_data4MFNN" / "S_exp_partial_xz+yz.npz",
                 sphere_p=r0_merged, rbc_p=exp_merged,
                 sphere_r=ref_yz_info["sphere_r"], rbc_r=ref_yz_info["rbc_r"])

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        ax = axes[0]
        yz_closed = np.vstack([sec_expr_yz, sec_expr_yz[0]])
        ax.plot(yz_closed[:,0], yz_closed[:,1], c="orange", linewidth=2.5,
                label="extracted smoothed loop", zorder=-1)
        ax.scatter(ref_ord_yz[:,0], ref_ord_yz[:,1], s=15, c="blue",
                   label="simualtion reference samples")
        ax.scatter(exp_yz_new[:,0], exp_yz_new[:,1], s=15, c="red",
                   label="pair-wised samples")
        for i in range(ref_ord_yz.shape[0]):
            ax.plot([ref_ord_yz[i,0], exp_yz_new[i,0]],
                    [ref_ord_yz[i,1], exp_yz_new[i,1]], 'k-', lw=0.5, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(prop={"family": "Times New Roman", "size": fs})
        ax.set_xlabel("y (um)", fontsize=fs)
        ax.set_ylabel("z (um)", fontsize=fs)
        ax.tick_params(labelsize=fs)

        ax = axes[1]
        xz_closed = np.vstack([sec_expr_xz, sec_expr_xz[0]])
        ax.plot(xz_closed[:,0], xz_closed[:,1], c="orange", linewidth=2.5,
                label="extracted smoothed loop", zorder=-1)
        ax.scatter(ref_ord_xz[:,0], ref_ord_xz[:,1], s=15, c="blue",
                   label="simualtion reference samples")
        ax.scatter(exp_xz_new[:,0], exp_xz_new[:,1], s=15, c="red",
                   label="pair-wised samples")
        for i in range(ref_ord_xz.shape[0]):
            ax.plot([ref_ord_xz[i,0], exp_xz_new[i,0]],
                    [ref_ord_xz[i,1], exp_xz_new[i,1]], 'k-', lw=0.5, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(prop={"family": "Times New Roman", "size": fs})
        ax.set_xlabel("x (um)", fontsize=fs)
        ax.set_ylabel("z (um)", fontsize=fs)
        ax.tick_params(labelsize=fs)

        plt.tight_layout()
        plt.savefig(ROOT_DIR / "Out_graph" / "S_shown.png", dpi=300, bbox_inches="tight")
        plt.close()
        pbar.update(1)


if __name__ == "__main__":
    main()
