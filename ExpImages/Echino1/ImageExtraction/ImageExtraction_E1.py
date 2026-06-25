import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from scipy.ndimage import gaussian_filter1d
from skimage import exposure
from skimage.filters import gaussian, sato, threshold_otsu
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import (
    binary_closing, binary_opening, binary_dilation,
    disk, remove_small_objects, remove_small_holes
)

plt.rcParams["figure.dpi"] = 120

ROOT_DIR = Path(__file__).resolve().parent

# ── CV2 helpers ───────────────────────────────────────────────────────────────

def resample_closed_polyline(P, N):
    P = np.asarray(P, dtype=np.float64)
    Q = np.vstack([P, P[0]])
    d = np.sqrt(((Q[1:] - Q[:-1]) ** 2).sum(axis=1))
    s = np.hstack([[0.0], np.cumsum(d)])
    t = np.linspace(0, s[-1], N + 1)[:-1]
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])

def keep_largest_component(mask_bool):
    lab = label(mask_bool)
    if lab.max() == 0:
        return mask_bool
    props = regionprops(lab)
    k = np.argmax([p.area for p in props])
    return lab == props[k].label

def extract_band_mask(
    image_path,
    sigma_img=2.0,
    sato_sigmas=(2, 3, 4, 5, 6, 7, 8),
    thr_mode="percentile",
    thr_percentile=82,
    open_r=1,
    close_r_big=10,
    close_r_small=6,
    dilate_r=2,
    min_obj=2000,
    hole_area=40000
):
    img_u8 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(image_path)
    img_f  = img_u8.astype(np.float32) / 255.0
    img_eq = exposure.equalize_adapthist(img_f, clip_limit=0.03)
    img_s  = gaussian(img_eq, sigma=sigma_img, preserve_range=True)
    inv    = 1.0 - img_s

    ridge = sato(inv, sigmas=list(sato_sigmas), black_ridges=False).astype(np.float32)
    ridge = (ridge - ridge.min()) / (ridge.max() - ridge.min() + 1e-12)

    t = threshold_otsu(ridge) if thr_mode == "otsu" else np.percentile(ridge, thr_percentile)
    band = ridge > t

    if open_r and open_r > 0:
        band = binary_opening(band, disk(open_r))
    band = binary_closing(band, disk(close_r_big))
    band = binary_closing(band, disk(close_r_small))
    if dilate_r and dilate_r > 0:
        band = binary_dilation(band, disk(dilate_r))
    band = remove_small_objects(band, min_size=min_obj)
    band_filled = remove_small_holes(band, area_threshold=hole_area)
    band_filled = keep_largest_component(band_filled)

    return img_u8, band_filled.astype(bool), ridge, band.astype(bool)

def extract_outer_outline_from_mask(mask_bool, N=2048):
    contours = find_contours(mask_bool.astype(np.float32), 0.5)
    if not contours:
        raise RuntimeError("No contours found from mask.")
    best, best_area = None, -1.0
    for c in contours:
        xy = np.column_stack([c[:, 1], c[:, 0]])
        x, y = xy[:, 0], xy[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area > best_area:
            best_area = area
            best = xy
    return resample_closed_polyline(best, N), best

def smooth_closed_loop_xy(loop_xy, sigma=3.0):
    P = np.asarray(loop_xy, dtype=np.float64)
    x = gaussian_filter1d(P[:, 0], sigma=sigma, mode="wrap")
    y = gaussian_filter1d(P[:, 1], sigma=sigma, mode="wrap")
    return np.column_stack([x, y])

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
    theta = np.degrees(np.arctan2(z, y)) if plane == "yz" else np.degrees(np.arctan2(z, x))
    theta = np.mod(theta, 360.0)
    arr5 = np.column_stack([ID, x, y, z, theta])
    return arr5[np.argsort(arr5[:, 4])]

def _center(P):
    return P - P.mean(axis=0, keepdims=True)

def _rotate_2d(P, phi_deg):
    phi = np.deg2rad(phi_deg)
    c, s = np.cos(phi), np.sin(phi)
    return P @ np.array([[c, -s], [s, c]], dtype=float).T

def rotation_2d(exp_2d, ref_2d, rotate_angle=0.0, center=True):
    exp0 = _center(np.asarray(exp_2d, float)) if center else np.asarray(exp_2d, float).copy()
    phi = float(np.mod(rotate_angle, 360.0))
    return _rotate_2d(exp0, phi), phi

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

def resample_exp_to_ref_arclength_plane(exp_2d, sec_ref_5, allow_reverse=True):
    ref = np.asarray(sec_ref_5, float)
    exp = np.asarray(exp_2d, float)
    M = ref.shape[0]
    ID_ref, theta_ref = ref[:, 0], ref[:, 4]
    ref_2d = ref[:, [2, 3]]
    exp_rs = _resample_closed_curve_arclength(exp, M)
    exp_best, _, _, _ = _best_cyclic_shift(exp_rs, ref_2d, allow_reverse=allow_reverse)
    return np.column_stack([ID_ref, ref[:, 1], exp_best[:, 0], exp_best[:, 1], theta_ref])

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams["font.family"] = "Times New Roman"
    image_path = ROOT_DIR / "In_image" / "Image_Echino1.png"
    N_outline  = 2048
    Lx_um      = 7.5
    fs         = 25
    Rot_Angle  = 3.0

    STEPS = [
        "Part 1  band extraction + outer outline",
        "Part 2  smooth & resample → physical coords",
        "Save    E1_p1234.png",
        "YZ      rotation + resample + match",
        "Save    E1_exp_partial_yz.npz + E1_shown.png",
    ]

    with tqdm(total=len(STEPS), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
              ncols=72) as pbar:

        # ── Part 1 ───────────────────────────────────────────────────────────
        pbar.set_description(STEPS[0])
        img_u8, band_filled, ridge, _ = extract_band_mask(
            image_path,
            sigma_img=2.0, sato_sigmas=(2, 3, 4, 5, 6, 7, 8),
            thr_mode="percentile", thr_percentile=82,
            open_r=1, close_r_big=10, close_r_small=6,
            dilate_r=2, min_obj=2000, hole_area=40000)
        outline_ptsN, _ = extract_outer_outline_from_mask(band_filled, N=N_outline)
        img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        pbar.update(1)

        # ── Part 2 ───────────────────────────────────────────────────────────
        pbar.set_description(STEPS[1])
        outline_smooth = smooth_closed_loop_xy(outline_ptsN, sigma=3.5)
        outline_smooth = resample_closed_polyline(outline_smooth, N_outline)
        xarr = outline_smooth[:, 0]
        yarr = outline_smooth[:, 1]
        Ly_um = Lx_um / (xarr.max() - xarr.min()) * (yarr.max() - yarr.min())
        loop_phys_um = normalize_and_scale_to_physical(outline_smooth, Lx_um=Lx_um, Ly_um=Ly_um)
        pbar.update(1)

        # ── Save E1_p1234.png ─────────────────────────────────────────────────
        pbar.set_description(STEPS[2])
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.imshow(ridge, cmap="gray", origin="upper")
        ax.axis("off")

        ax = axes[0, 1]
        ax.imshow(band_filled, cmap="gray", origin="upper")
        ax.plot(outline_ptsN[:, 0], outline_ptsN[:, 1], lw=2.0, c="b", label="outer outline")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.legend(prop={"family": "Times New Roman", "size": 16}, frameon=False, labelcolor="white")

        ax = axes[1, 0]
        ax.imshow(img_orig, cmap="gray", origin="upper", vmin=0, vmax=255)
        ax.plot(outline_ptsN[:, 0], outline_ptsN[:, 1], lw=2.0, c="b", label="outer outline")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.legend(prop={"family": "Times New Roman", "size": 16}, frameon=False)

        ax = axes[1, 1]
        ax.imshow(img_orig, cmap="gray", origin="upper", vmin=0, vmax=255)
        ax.plot(outline_smooth[:, 0], outline_smooth[:, 1], lw=2.5, c="orange", label="smoothed loop")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.legend(prop={"family": "Times New Roman", "size": 16}, frameon=False)

        plt.tight_layout()
        plt.savefig(ROOT_DIR / "Out_graph" / "E1_p1234.png", dpi=300, bbox_inches="tight")
        plt.close()
        pbar.update(1)

        # ── YZ alignment ─────────────────────────────────────────────────────
        pbar.set_description(STEPS[3])
        ref_yz_info = np.load(ROOT_DIR / "In_DPDref" / "Eii_mod_partial_x.npz")
        sec_ref_yz  = add_theta_and_sort_plane(ref_yz_info["rbc_p"], plane="yz")

        sec_expr_yz, _ = rotation_2d(loop_phys_um[:, [0,1]], sec_ref_yz[:, [2,3]],
                                     rotate_angle=Rot_Angle)
        sec_exp2_yz = resample_exp_to_ref_arclength_plane(sec_expr_yz, sec_ref_yz)

        reference_yz  = sec_ref_yz[:, [2,3]]
        experiment_yz = sec_exp2_yz[:, [2,3]]
        ref_ord = anchor_bottom(order_loop_nn_2opt(reference_yz))
        exp_ord = anchor_bottom(order_loop_nn_2opt(experiment_yz))
        exp_ord = ensure_same_orientation(ref_ord, exp_ord)
        exp_yz_new, _ = best_shift_exp_to_ref(ref_ord, exp_ord)
        pbar.update(1)

        # ── Save npz + shown.png ──────────────────────────────────────────────
        pbar.set_description(STEPS[4])
        sec_exp4_yz = _match_ids_yz(ref_ord, exp_yz_new, sec_ref_yz)
        sec_exp2_yz_final = sort_by_id_drop_theta(sec_exp4_yz)
        np.savez(ROOT_DIR / "Out_data4MFNN" / "E1_exp_partial_yz.npz",
                 sphere_p=ref_yz_info["sphere_p"], rbc_p=sec_exp2_yz_final,
                 sphere_r=ref_yz_info["sphere_r"], rbc_r=ref_yz_info["rbc_r"])

        yz_closed = np.vstack([sec_expr_yz, sec_expr_yz[0]])
        plt.figure(figsize=(10, 10))
        plt.plot(yz_closed[:, 0], yz_closed[:, 1],
                 c="orange", linewidth=2.5, label="extracted smoothed loop", zorder=-1)
        plt.scatter(ref_ord[:, 0], ref_ord[:, 1],
                    s=15, c="blue", label="simualtion reference samples")
        plt.scatter(exp_yz_new[:, 0], exp_yz_new[:, 1],
                    s=15, c="red", label="pair-wised samples")
        for i in range(ref_ord.shape[0]):
            plt.plot([ref_ord[i,0], exp_yz_new[i,0]],
                     [ref_ord[i,1], exp_yz_new[i,1]], 'k-', lw=0.5, alpha=0.5)
        plt.axis("equal")
        plt.legend(prop={"family": "Times New Roman", "size": fs})
        plt.xlabel("y (um)", fontsize=fs)
        plt.ylabel("z (um)", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.savefig(ROOT_DIR / "Out_graph" / "E1_shown.png", dpi=300, bbox_inches="tight")
        plt.close()
        pbar.update(1)


if __name__ == "__main__":
    main()
