'''
This file is for distant viewing on Caravaggio's oeuvre.

Extract lighting information from paintings and incorporate it into a JSON file.
'''

import json
import os
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import Any, Dict, List, Tuple
from scipy.ndimage import sobel
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates
from math import sqrt
from scipy.special import erf

#############################################################################################################
#############################################################################################################
# Load the master JSON
json_filepath = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json'
with open(json_filepath, 'r') as f:
    data = json.load(f)

##############################################################################
# Variables to change
painting_id = 15
url = data['paintings'][painting_id-1]['url']           

parent_dir = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/images'
img_filename = str(painting_id) + '.jpg'
# img_filename = 'obliquelight.png'
img_path = os.path.join(parent_dir, img_filename)
image = cv2.imread(img_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
utils.show_painting_summary(painting_id)

# utils.plot_painting_contours_C(painting_id, json_filepath, img_path, scale=200, plot_contours_flag=True, plot_only_contours=False, plot_all_contours=True, contour_ids=None)
# plot_painting_contours_C(painting_id, json_filepath, img_path, scale=200, plot_only_contours=False, plot_all_contours=True, contour_ids=None)

##############################################################################
# Extract lighting information from paintings
##############################################################################
# Single contour
# smoothness = 5
# N = 5
# sigma = 3
contour0 = utils.extract_contour_ui(image_rgb)
contour1 = utils.extract_contour_ui(image_rgb)
contour2 = utils.extract_contour_ui(image_rgb)
contour3 = utils.extract_contour_ui(image_rgb)

contours = [contour0]


# Multiple contour
def get_contour_info_by_id(
    data: Dict[str, Any],
    painting_id: int,
    contour_ids: List[int]
) -> Tuple[List[np.ndarray], List[float], List[int]]:
    try:
        painting = next(p for p in data["paintings"]
                      if p["painting_id"] == painting_id)
    except StopIteration:
        raise ValueError(f"painting_id={painting_id} not found")
    id2entry = {c["contour_id"]: c
                for c in painting["analysis"].get("contours", [])}
    contours: List[np.ndarray] = []
    smooths:  List[float]      = []
    Ns:       List[int]        = []
    for cid in contour_ids:
        entry = id2entry.get(cid)
        if entry is None:
            raise ValueError(f"contour_id={cid} not found")
        coords_block   = entry["contour_coordinates"]        # list of k contours
        smooth_block   = entry["smoothness"]                 # list of k floats
        N_block        = entry["Parameter_N"]                # list of k ints
        if (isinstance(coords_block, list) and
            isinstance(smooth_block, list) and
            isinstance(N_block, list) and
            len(coords_block) == len(smooth_block) == len(N_block)):
            for c, s, n in zip(coords_block, smooth_block, N_block):
                contours.append(np.asarray(c, dtype=float))
                smooths.append(float(s))
                Ns.append(int(n))
        else:  # fallback: treat entry as single contour
            contours.append(np.asarray(coords_block, dtype=float))
            smooths.append(float(smooth_block if isinstance(smooth_block, (int, float)) else smooth_block[0]))
            Ns.append(int(N_block if isinstance(N_block, (int, float)) else N_block[0]))
    return contours, smooths, Ns


utils.show_painting_summary(painting_id)
contour_ids=[1, 2, 3, 4, 5]
contours, smooth_vals, N_vals = get_contour_info_by_id(data, painting_id=painting_id, contour_ids=contour_ids)

contours = contours + [contour0, contour1]
len(contours)

def get_normals_C(contour, smoothness=10, oversample=4):
    """
    Fit a spline to *contour* and return *one* averaged normal per image pixel.
    Parameters
    ----------
    contour    : list | (N,2) array of (x, y) coordinates
    smoothness : spline smoothing parameter (same role as before)
    oversample : how many spline samples to generate *per original point*
                 (≥1 is enough; 3–5 is plenty)
    Returns
    -------
    x_px, y_px : 1-D arrays (M,)   – averaged contour coords (float)
    normals_px : 2-D array  (M,2) – outward-pointing unit normals
    """
    # --- 0. raw contour → spline -------------------------------------------------
    contour = np.asarray(contour, dtype=float)
    x_raw, y_raw = contour[:, 0], contour[:, 1]
    # cubic B-spline through data
    tck, _ = splprep([x_raw, y_raw], s=smoothness, per=False)
    n_dense = max(oversample * len(contour), len(contour))      # guarantee ≥ input size
    u_dense = np.linspace(0.0, 1.0, n_dense, endpoint=False)
    sx, sy        = splev(u_dense, tck)                         # spline points
    dxdu, dydu    = splev(u_dense, tck, der=1)                  # tangents
    # --- 1. surface normals at dense samples -------------------------------
    tangents      = np.column_stack([dxdu, dydu])
    normals_dense = np.column_stack([-dydu, dxdu])
    normals_dense /= np.linalg.norm(normals_dense, axis=1, keepdims=True)
    # --- 2. collapse samples that map to the same pixel --------------------
    ix = np.floor(sx).astype(np.int32)
    iy = np.floor(sy).astype(np.int32)
    # unique pixel id = (ix,iy) packed into one int64 for fast grouping
    pix_id = (ix.astype(np.int64) << 32) | iy.astype(np.int64)
    uniq_id, inv = np.unique(pix_id, return_inverse=True)
    M = uniq_id.size
    # accumulate sums per pixel
    sum_x  = np.bincount(inv, sx, minlength=M)
    sum_y  = np.bincount(inv, sy, minlength=M)
    sum_nx = np.bincount(inv, normals_dense[:, 0], minlength=M)
    sum_ny = np.bincount(inv, normals_dense[:, 1], minlength=M)
    counts = np.bincount(inv, minlength=M).astype(float)
    x_px = sum_x / counts
    y_px = sum_y / counts
    normals_px = np.column_stack([sum_nx, sum_ny]) / counts[:, None]
    normals_px /= np.linalg.norm(normals_px, axis=1, keepdims=True)
    # --- 3. order pixels along contour ------------------------------------
    # take the *first* appearance index inside each pixel to keep topology
    first_idx = np.zeros(M, dtype=int)
    np.minimum.at(first_idx, inv, np.arange(n_dense))
    order = np.argsort(first_idx)
    x_px, y_px, normals_px = x_px[order], y_px[order], normals_px[order]
    # --- 4. orient normals consistently outward ---------------------------
    normals_px = correct_normal_orientations(x_raw, y_raw, x_px, y_px, normals_px)
    return x_px, y_px, normals_px


smooth_vals = [15] *1 # smoothness values for each contour
N_vals = [5]*1


all_spline_x: List[Any] = []
all_spline_y: List[Any] = []
all_normals: List[Any] = []

# Loop over each (contour, smoothness) pair and call get_normals_C:
for contour, smoothness in zip(contours, smooth_vals):
    spline_x, spline_y, normals = get_normals_C(contour, smoothness)
    all_spline_x.append(spline_x)
    all_spline_y.append(spline_y)
    all_normals.append(normals)

plot_contour_and_normals(img_path, all_spline_x,all_spline_y,all_normals,arrow_scale=50, point_size=2)



def plot_contour_and_normals(
        img_path: str,
        x_in, y_in,             # either 1-D array  *or* list of arrays
        normals_in,             # (N,2) array  *or* list of arrays
        arrow_scale: float = 50,
        point_size:  int   = 2
    ):
    # ---- 1. transparently allow single arrays or lists ------------------
    if isinstance(x_in, (list, tuple)):
        x = np.concatenate(x_in).astype(np.float32)
        y = np.concatenate(y_in).astype(np.float32)
        N  = np.vstack(normals_in).astype(np.float32)
    else:
        x = np.asarray(x_in, dtype=np.float32)
        y = np.asarray(y_in, dtype=np.float32)
        N = np.asarray(normals_in, dtype=np.float32)
    # ---- 2. load & display image ---------------------------------------
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.scatter(x, y, c='red', s=point_size, label='Contour pts')
    # ---- 3. draw every normal as an arrow ------------------------------
    for xi, yi, (nx, ny) in zip(x, y, N):
        ax.arrow(xi, yi, nx*arrow_scale, ny*arrow_scale,
                 color='skyblue', head_width=3, length_includes_head=True)
    ax.axis('off')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------- #
#  GET THE TRUE CONTOUR AT SUBPIXEL LEVEL           #
# ------------------------------------------------------------------------- #
def _isodata_threshold(img: np.ndarray, *, eps: float = .5, max_iter: int = 40):
    """Return a global threshold T that separates the histogram into two modes."""
    t = img.mean()                                   # initial guess
    for _ in range(max_iter):
        g1 = img[img <= t]
        g2 = img[img >  t]
        new_t = 0.5 * (g1.mean() + g2.mean())
        if abs(new_t - t) < eps:
            break
        t = new_t
    return t

def get_subpixel_contour(
        gray_image : np.ndarray,
        x          : np.ndarray,
        y          : np.ndarray,
        normals    : np.ndarray,
        N          : int,
        *,
        delta            : float = 0.25,   # sampling pitch along the normal
        R_px             : float | None = None,
        amplitude_thresh : float = 1.0     # minimal Gaussian peak height
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    **Implements the pipeline of “Gaussian‑Based Approach to Sub‑pixel Detection
    of Blurred and Unsharp Edges”**
    1. Sobel gradient magnitude  |∇L|
    2. Single global threshold (ISODATA) → coarse edge (for diagnostics only)
    3. For every coarse contour sample:
       • 1‑D samples of |∇L| along the *outward* normal  
       • quadratic interpolation of the discrete peak (initial s0̂)  
       • Levenberg–Marquardt fit of a Gaussian peak B + A·exp(‑(s‑s0)²/2σ²)  
       • update the vertex position  p_new = p_old − s0·n
    """
    # ------------------------------------------------------------------ #
    # 0.  Sobel gradient magnitude, ISODATA threshold                    #
    # ------------------------------------------------------------------ #
    gx   = sobel(gray_image.astype(np.float32), axis=1, mode='reflect')
    gy   = sobel(gray_image.astype(np.float32), axis=0, mode='reflect')
    grad = np.hypot(gx, gy).astype(np.float32)
    # coarse mask (optional; keeps I/O identical to the paper)
    T_coarse = _isodata_threshold(grad)
    coarse_mask = grad > T_coarse
    # ------------------------------------------------------------------ #
    # 1.  two interpolators (gradient + luminance)                       #
    # ------------------------------------------------------------------ #
    grid = (np.arange(gray_image.shape[0], dtype=np.float32),   # rows (y)
            np.arange(gray_image.shape[1], dtype=np.float32))   # cols (x)
    interp_grad = RegularGridInterpolator(grid, grad,
                                          bounds_error=False, fill_value=np.nan)
    # ------------------------------------------------------------------ #
    # 2.  initialise outputs                                             #
    # ------------------------------------------------------------------ #
    new_x  = x.astype(np.float32).copy()
    new_y  = y.astype(np.float32).copy()
    # new_I  = interp_gray(np.column_stack((y, x)))               # initial guess
    if R_px is None:
        R_px = min(3.0, float(N))
    R_px = float(min(R_px, N))
    # ------------------------------------------------------------------ #
    # 3.  per‑vertex optimisation                                        #
    # ------------------------------------------------------------------ #
    for i in range(len(x)):
        p0 = np.array([x[i], y[i]], dtype=np.float32)   # (x, y)
        n  = normals[i].astype(np.float32)
        # -------------------------------------------------------------- #
        # 3a. sample |∇L| along the normal                               #
        # -------------------------------------------------------------- #
        s = np.arange(-R_px, R_px + 1e-3, delta, dtype=np.float32)
        coords = np.column_stack((p0[1] - s*n[1],   # rows  (y)
                                  p0[0] - s*n[0]))  # cols  (x)
        g_vals = interp_grad(coords)
        valid = ~np.isnan(g_vals)
        if valid.sum() < 5:
            continue                     # keep original point, no shift
        s, g_vals = s[valid], g_vals[valid]
        # -------------------------------------------------------------- #
        # 3b. coarse sub‑pixel peak via quadratic fit                    #
        # -------------------------------------------------------------- #
        k_max = np.argmax(g_vals)
        if 0 < k_max < len(s) - 1:
            num   = (g_vals[k_max - 1] - g_vals[k_max + 1]) * delta
            denom = 2*(g_vals[k_max - 1] - 2*g_vals[k_max] + g_vals[k_max + 1])
            s0_hat = s[k_max] + (num / denom) if denom != 0 else s[k_max]
        else:
            s0_hat = s[k_max]
        # -------------------------------------------------------------- #
        # 3c. Gaussian peak fit                                          #
        # -------------------------------------------------------------- #
        def gauss_peak(s_, A, s0, sigma, B):
            return B + A * np.exp(- (s_ - s0)**2 / (2*sigma**2))
        p_init = (g_vals[k_max] - np.median(g_vals),  # A (amplitude)
                  s0_hat,                             # s0
                  0.8,                                # σ  (px)
                  np.median(g_vals))                  # B (base line)
        try:
            (A, s0, sigma, B), _ = curve_fit(
                gauss_peak, s, g_vals, p0=p_init,
                bounds=([-np.inf, -R_px, 0.3, 0.0],
                        [ np.inf,  R_px, 3.0, np.inf]),
                maxfev=400
            )
        except RuntimeError:
            A, s0 = 0.0, 0.0          # very flat → ignore
        # reject weak peaks (texture edges, noise)
        if A < amplitude_thresh:
            s0 = 0.0
        # -------------------------------------------------------------- #
        # 3d. update coordinate + luminance at that coordinate           #
        # -------------------------------------------------------------- #
        p_new = p0 - s0 * n
        new_x[i], new_y[i] = p_new
        # new_I[i] = interp_gray([[p_new[1], p_new[0]]])[0]
    return new_x, new_y


all_new_points_x: List[Any] = []
all_new_points_y: List[Any] = []

for spline_x, spline_y, normals in zip(all_spline_x, all_spline_y, all_normals):
    new_x, new_y = get_subpixel_contour(
        gray_image,
        spline_x,
        spline_y,
        normals,
        N=10,
        delta=0.25,
        R_px=3.0,
        amplitude_thresh=1.0
    )
    all_new_points_x.append(new_x)
    all_new_points_y.append(new_y)


# Plot the original spline points and the new subpixel points
# org_img = cv2.imread(img_path)
# gray_image = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
# fig, ax = plt.subplots()
# ax.imshow(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))
# for spline_x, spline_y, new_x, new_y in zip(
#     all_spline_x, all_spline_y, all_new_points_x, all_new_points_y):
#     ax.scatter(spline_x, spline_y, color='blue', s=2, label='_nolegend_')
#     ax.scatter(new_x, new_y, color='red', s=2, label='_nolegend_')

# # Create a single legend entry for “old” vs. “subpixel” points
# ax.scatter([], [], color='blue', s=2, label='Old (spline)')
# ax.scatter([], [], color='red', s=2, label='Subpixel')
# ax.axis('off')
# ax.legend()
# plt.show()


def _extrapolate_I0(
    gray: np.ndarray,
    px: float, py: float,
    nx: float, ny: float,
    *,
    R_in_px     : float  = 4.0,
    delta       : float  = 0.25,
    method      : str    = "erf",
    poly_degree : int    = 2,
    amplitude_min: float = 3.0
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Sample gray[y,x] along the inward normal at (px,py) and extrapolate
    to s=0.  Returns (I0, s_valid, I_valid).
    """
    # build a fast cubic sampler once per call
    def _sampler(coords):
        # coords: (K,2) array of [row, col]
        rc = np.vstack([coords[:,0], coords[:,1]])
        return map_coordinates(gray, rc, order=3, mode='reflect')
    # inward unit normal
    n_in = np.array([-nx, -ny], dtype=np.float32)
    # sample positions s ∈ [δ, R]
    s = np.arange(delta, R_in_px+1e-3, delta, dtype=np.float32)
    coords = np.column_stack((py + s*n_in[1],  px + s*n_in[0]))  # (y, x)
    I_samp = _sampler(coords)
    # keep only valid, sufficiently‐strong edges
    valid = (~np.isnan(I_samp)) & (np.ptp(I_samp) >= amplitude_min)
    s_v, I_v = s[valid], I_samp[valid]
    if len(s_v) < 4:
        return np.nan, s_v, I_v
    # now extrapolate to s=0
    if method == "mean":
        mask = s_v >= 1.0  
        I0 = I_v[mask].mean()
    elif method == "linear":
        m = s_v <= 1.0
        coef = np.polyfit(s_v[m], I_v[m], 1)
        I0 = np.polyval(coef, 0.0)
    elif method == "robust_poly":
        # Huber‐weighted polynomial
        w = np.ones_like(s_v)
        for _ in range(5):
            V    = np.vander(s_v, poly_degree+1)
            coef, *_ = np.linalg.lstsq(V * w[:,None], I_v*w, rcond=None)
            resid = I_v - V.dot(coef)
            mad   = np.median(np.abs(resid)) + 1e-6
            w     = np.minimum(1.0, 1.5 * mad / np.abs(resid))
        I0 = np.polyval(coef, 0.0)
    elif method == "erf":
        # fixed‐sigma error‐function model
        def erf_step(s_, I_in, dI):
            σ = 0.6
            return I_in - 0.5*dI*(1 + erf((-s_)/(sqrt(2)*σ))) # -s because of the direction of normal sampling is inwards!
        try:
            p0 = np.array([I_v.max(), I_v.ptp()], dtype=float)
            lb = np.array([0, 0], dtype=float)
            ub = np.array([255, 255], dtype=float)
            eps = 1e-6
            p0 = np.minimum(np.maximum(p0, lb + eps), ub - eps)
            popt, _ = curve_fit(erf_step, s_v, I_v,
                                p0=p0,
                                bounds=([0,0], [255,255]),
                                maxfev=200)
            I0 = popt[0]
            # print(f"Extrapolated I0 = {I0:.2f} using erf method with parameters: {popt}")
        except RuntimeError:
            # print("Curve fitting failed, using mean instead.")
            I0 = I_v.mean()
    else:
        raise ValueError(f"Unknown method {method!r}")
    return float(I0), s_v, I_v


def get_subpixel_luminance(
    gray_image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    normals: np.ndarray,
    *,
    R_in_px      = 4.0,
    delta        = 0.25,
    method       = "erf",
    poly_degree  = 2,
    amplitude_min= 3.0,
    return_profile=False
):
    gray = gray_image.astype(np.float32)
    I0  = np.full_like(x, np.nan, dtype=np.float32)
    profiles = [] if return_profile else None
    for i, (px, py, (nx, ny)) in enumerate(zip(x, y, normals)):
        I0_i, s_v, I_v = _extrapolate_I0(
            gray, px, py, nx, ny,
            R_in_px=R_in_px,
            delta=delta,
            method=method,
            poly_degree=poly_degree,
            amplitude_min=amplitude_min
        )
        I0[i] = I0_i
        if return_profile:
            profiles.append((s_v, I_v))
    if return_profile:
        return I0, profiles
    else:
        return I0


all_I_true: List[Any] = []
for new_x, new_y, normals in zip(all_new_points_x, all_new_points_y, all_normals):
    I_true = get_subpixel_luminance(
        gray_image,
        new_x,
        new_y,
        normals,
        R_in_px=4.0,
        delta=0.25,
        method="erf"
    )
    all_I_true.append(I_true)


# if I_true contains nans remove that nans and save the indices to remove same indices from normals
filtered_new_points_x_list: List[np.ndarray] = []
filtered_new_points_y_list: List[np.ndarray] = []
filtered_normals_list: List[np.ndarray] = []
filtered_I_true_list: List[np.ndarray] = []
for new_x, new_y, normals, I_true in zip(
    all_new_points_x, all_new_points_y, all_normals, all_I_true
):
    # Identify valid (non-NaN) indices
    valid_indices = ~np.isnan(I_true)
    # Filter out NaNs for this contour
    filtered_new_x = new_x[valid_indices]
    filtered_new_y = new_y[valid_indices]
    filtered_normals = normals[valid_indices]
    filtered_I_true = I_true[valid_indices]
    # Append the filtered arrays to the new lists
    filtered_new_points_x_list.append(filtered_new_x)
    filtered_new_points_y_list.append(filtered_new_y)
    filtered_normals_list.append(filtered_normals)
    filtered_I_true_list.append(filtered_I_true)


def compute_light_direction_C(
        normal_list,              # list of (Ni,2) arrays
        lum_list,                 # list of 1-D arrays   (same length Ni)
        max_iter: int = 10
    ):
    # -------- 1. concatenate all patches ---------------------------------
    normals           = np.vstack(normal_list).astype(np.float32)
    ground_truth_subpixel = np.concatenate(lum_list).astype(np.float32)
    n_total_points = len(ground_truth_subpixel)
    mask = np.ones(n_total_points, dtype=bool)
    prev_mask = None
    # -------- 2. iterative positive-dot filtering ------------------------
    for _ in range(max_iter):
        current_idx = np.where(mask)[0]
        M = normals[current_idx]
        b = ground_truth_subpixel[current_idx]
        # linear LS
        vx, vy = np.linalg.lstsq(M, b, rcond=None)[0]
        L = np.array([vx, vy], dtype=np.float32)
        nrm = np.linalg.norm(L)
        if nrm > 1e-12:
            L /= nrm
            # outward‐flip safeguard
            if np.dot(np.mean(M, axis=0), L) < 0:
                L = -L
        else:
            L[:] = 0.0
        # keep only rows whose dot > 0
        new_mask = (normals @ L) > 0
        # stop if mask stabilises or too few points
        if np.array_equal(new_mask, mask) or new_mask.sum() < 3:
            mask = new_mask if new_mask.sum() >= 3 else mask
            break
        prev_mask, mask = mask, new_mask
    # -------- 3. final LS on converged mask ------------------------------
    M_fin = normals[mask]
    b_fin = ground_truth_subpixel[mask]
    vx, vy = np.linalg.lstsq(M_fin, b_fin, rcond=None)[0]
    L_final = np.array([vx, vy], dtype=np.float32)
    nrm = np.linalg.norm(L_final)
    if nrm > 1e-12:
        L_final /= nrm
        if np.dot(np.mean(M_fin, axis=0), L_final) < 0:
            L_final = -L_final
    else:
        L_final[:] = 0.0
    # -------- 4. uncertainty via cost-curvature --------------------------
    θ = np.arctan2(M_fin[:,1], M_fin[:,0])
    order = np.argsort(θ)
    θ_s   = θ[order]
    b_s   = b_fin[order]
    def cost(phi):
        return np.mean((np.maximum(0, np.cos(θ_s - phi)) - b_s)**2)
    phi_opt = np.arctan2(L_final[1], L_final[0])
    C0      = cost(phi_opt)
    δ       = 1e-4
    second  = (cost(phi_opt+δ) + cost(phi_opt-δ) - 2*C0) / δ**2
    sigma_deg = np.degrees(np.sqrt(2.0/second)) if second>0 else np.inf
    return L_final, sigma_deg


light_direction_estimation, angle_std_degs = compute_light_direction_C(filtered_normals_list, filtered_I_true_list)

empirical_stds: List[float] = []
for new_x, new_y, normals, I_true in zip(
    filtered_new_points_x_list,
    filtered_new_points_y_list,
    filtered_normals_list,
    filtered_I_true_list
):
    # Compute empirical std for this contour
    std_deg = utils.compute_empirical_std(
        img_path,
        new_x,
        new_y,
        normals,
        I_true,
        max_iter=10,
        N_max=3
    )
    empirical_stds.append(std_deg)

empirical_stds_arr = np.array(empirical_stds, dtype=np.float32)
average_std_deg = np.nanmean(empirical_stds_arr)


stacked_normals = np.vstack(filtered_normals_list)
stacked_I_true = np.concatenate(filtered_I_true_list)
# -----------------------------------------------------------------------------#
# sigma = 1
# ground_truth_subpixel_smooth = gaussian_filter1d(ground_truth_subpixel, sigma=sigma)
# empirical_std_deg = utils.compute_empirical_std(img_path, new_points_x, new_points_y, normals, I_true, max_iter=10, N_max=3)

####
# gt_light_direction_estimation = np.array([-0.89, -0.45])
# gt_light_direction_estimation = np.array([-0.7454, 0.666])
# angles_rad = np.degrees(np.mod(np.arctan2(-light_direction_estimation[1], light_direction_estimation[0]), 2*np.pi))        # [0, 2π)

# plot luminance distributions against contour index
# meas_smooth = gaussian_filter1d(I_true, sigma=1)
theoretical_light = np.clip(stacked_normals @ light_direction_estimation, 0, 1)
# ground_truth_light = gt_light_direction_estimation @ normals.T

I_true_norm = utils.normalize_to_01(stacked_I_true)
# I_meas_norm = utils.normalize_to_01(meas_smooth)
I_theo_norm = utils.normalize_to_01(theoretical_light)
# I_gt_norm   = utils.normalize_to_01(ground_truth_light)

# plot luminance distributions against angle
kx, ky = stacked_normals[:, 0], stacked_normals[:, 1]
angles_rad = np.mod(np.arctan2(-ky, kx), 2*np.pi)          # [0, 2π)
angles_deg = np.degrees(angles_rad)
order      = np.argsort(angles_deg)
θ_sorted   = angles_deg[order]
I_theo_s   = I_theo_norm[order]
# I_gt_s     = I_gt_norm[order]

plt.figure(figsize=(6, 3))
# noisy measurement → scatter
plt.scatter(angles_deg, I_true_norm, s=14, alpha=0.6, label='measured')
# theoretical & ground-truth Lambertian → lines
plt.plot(θ_sorted, I_theo_s, lw=1.4, label='ground-truth Lambertian')
plt.xlabel('surface-normal angle θ  (deg)')
plt.ylabel('luminance (normalised)')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

##

# mse, delta, meas_block, theo_block, I_meas_norm, I_theo_norm, sigma = utils.compute_shading_delta_metric(ground_truth_subpixel, light_direction_estimation, normals, sigma=sigma, max_sigma=100, sigma_step=5)
shading_metrics: List[Tuple[float, float]] = []  # each entry: (mse, delta)
for new_x, new_y, normals, I_true in zip(
    filtered_new_points_x_list,
    filtered_new_points_y_list,
    filtered_normals_list,
    filtered_I_true_list
):
    result = utils.compute_shading_delta_metric(
        I_true,
        light_direction_estimation,
        normals,
        sigma=5,
        max_sigma=100,
        sigma_step=5)
    if result is not None:
        mse, delta, *_ = result
        shading_metrics.append((mse, delta))

# Convert to arrays
shading_metrics_arr = np.array(shading_metrics, dtype=np.float32)  # shape (num_contours, 2)
avg_mse = float(np.nanmean(shading_metrics_arr[:, 0]))
deltas = shading_metrics_arr[:, 1]
abs_mean = float(np.nanmean(np.abs(deltas)))
sum_delta = np.nansum(deltas)
overall_sign = np.sign(sum_delta) if sum_delta != 0 else 0
final_delta = abs_mean * overall_sign

# level_of_noise = utils.compute_noise_metric(ground_truth_subpixel)
noise_list: List[int] = []
for I_true_i in filtered_I_true_list:
    n_i = utils.compute_noise_metric(I_true_i)
    noise_list.append(n_i)

# Convert to numpy array for aggregation
noise_arr = np.array(noise_list, dtype=np.float32)
# If you prefer length-weighted average:
lengths = np.array([len(I) for I in filtered_I_true_list], dtype=np.float32)
weighted_avg_noise = float(np.dot(noise_arr, lengths) / np.sum(lengths))


# _, _, _, _, std_mse_deg = utils.get_Lightdirection_pdf([light_direction_estimation], [I_meas_norm], [normals])
# compute ranges
norm_smoothed_luminance_distribution = stacked_I_true / stacked_I_true.max()
brightness_range = norm_smoothed_luminance_distribution.max() - norm_smoothed_luminance_distribution.min()
normal_vectors_span = utils.compute_angle_range(stacked_normals)
coefficients_reg = utils.compute_sh_coefficients(stacked_normals, norm_smoothed_luminance_distribution)
# utils.visualize_delta_metric(I_meas_norm, I_theo_norm, meas_block, theo_block, title='Shading distributions (index domain)', lo_frac=0.1, hi_frac=0.9)
utils.vis_light_estimation_C(img_path, light_direction_estimation, filtered_new_points_x_list[0], filtered_new_points_y_list[0], average_std_deg, scale = 500, visual_factor = 1.0, plot_Ld=True, plot_contours_flag=True, plot_only_contours=False)

contour_type = "fruit"
belongs_to_person = "no"

contour_data = {    
    "contour_coordinates": contours,
    "smoothness": smooth_vals,
    "Parameter_N": N_vals,
    "brightness_range": [float(round(brightness_range, 4))],
    "normal_vectors_span": [float(normal_vectors_span)],
    "contour_type": contour_type,
    "belongs_to_person": belongs_to_person,
    "light_direction_estimation": [
        float(light_direction_estimation[0]), 
        float(light_direction_estimation[1])
    ],
    "empirical_std_deg": [float(average_std_deg)],
    "estimation_std_degrees": [float(angle_std_degs)],
    "std_mse_deg": [0.0],
    "mean_squared_error": [float(avg_mse)],
    "shading_rate": [float(final_delta)],
    "level_of_noise": [float(weighted_avg_noise)],
    "spherical_harmonics_coeffs": coefficients_reg.tolist(),
}

# ADD THE CONTOUR DATA TO THE JSON FILE
utils.add_contour(painting_id, contour_data)

utils.show_painting_summary(painting_id)

# UPDATE CONTOUR DATA BY ID
utils.replace_contour(painting_id=painting_id, contour_id=2, new_contour_data = contour_data)

# # DELETE CONTOUR DATA BY ID
utils.delete_contour(painting_id=painting_id, contour_id=6)

# (OPTIONAL) 'depicted_light_direction_evidence' Cast shadows or specular highlights: light direction estiamtions with no variance
cast_shadow = utils.extract_light_direction_ui(img_path)

# depicted_light_direction_evidence = cast_shadow
depicted_light_direction_evidence = []
# For specular highlights use the python script in C:/Users/pepel/PROJECTS/FaceArt/castShadow/specular_highlights.py

depicted_light_direction_evidence = [
    {
        "type": "cast shadow",
        "light_direction": cast_shadow,
        "notes": "cast shadow on the rear wall"
    }
]

# GLOBAL CONSISTENCY METRIC (after all contours have been extracted)
# global_light_direction_estimation, global_estimation_std_degrees = compute_global_estimation(painting_id, json_filepath, gray_image)

def _collect_per_contour_dirs(painting_entry: Dict[str, Any]) -> List[np.ndarray]:
    dirs: List[np.ndarray] = []
    for item in painting_entry.get("analysis", {}).get("contours", []):
        raw_dir = item.get("light_direction_estimation")
        if raw_dir is None:
            continue
        # Some entries store a single vector, others a list around it
        if isinstance(raw_dir, list) and len(raw_dir) == 2 and all(isinstance(v, (int, float)) for v in raw_dir):
            dirs.append(np.asarray(raw_dir, dtype=np.float32))
        elif isinstance(raw_dir, list) and len(raw_dir) == 1 and isinstance(raw_dir[0], list):
            cand = raw_dir[0]
            if len(cand) == 2:
                dirs.append(np.asarray(cand, dtype=np.float32))
    return dirs

def compute_global_light_from_estimates(painting_id: int, json_filepath: str) -> Tuple[np.ndarray, float]:
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        painting = next(p for p in data["paintings"] if p["painting_id"] == painting_id)
    except StopIteration:
        raise ValueError(f"painting_id={painting_id} not found")
    dirs = _collect_per_contour_dirs(painting)
    if not dirs:
        raise ValueError("No light_direction_estimation vectors present in JSON.")
    # Normalize each vector
    normed = []
    for v in dirs:
        nrm = np.linalg.norm(v)
        if nrm > 1e-6:
            normed.append(v / nrm)
    if not normed:
        raise ValueError("All light_direction_estimation vectors have zero length.")
    # Convert to angles, compute circular mean
    angles = np.arctan2([v[1] for v in normed], [v[0] for v in normed])
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    mean_angle = np.arctan2(mean_sin, mean_cos)
    L_mean = np.array([np.cos(mean_angle), np.sin(mean_angle)], dtype=np.float32)
    # --- uncertainty ---
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    circ_std = np.sqrt(-2 * np.log(R + 1e-12))
    std_mean_deg = float(np.degrees(circ_std))
    return L_mean, std_mean_deg

global_light_direction_estimation, global_estimation_std_degrees = compute_global_light_from_estimates(painting_id, json_filepath)

# calculate the center of the image
height, width = gray_image.shape
x_coordinates = np.array([width / 2], dtype=np.float32)
y_coordinates = np.array([height / 2], dtype=np.float32)
utils.vis_global_light_estimation_C(img_path, global_light_direction_estimation, x_coordinates, y_coordinates, global_estimation_std_degrees, scale=1000, visual_factor = 1.0)

# number_of_people = 1
# utils.add_remaining_info(painting_id, depicted_light_direction_evidence, number_of_people, global_light_direction_estimation, global_estimation_std_degrees)

def add_remaining_global_info(painting_id, global_light_direction_estimation, global_estimation_std_degrees, json_filepath="C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"):
    # 1. Load the master JSON
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    # 2. Locate the target painting
    for painting in data["paintings"]:
        if painting["painting_id"] == painting_id:
            painting["analysis"]["global_light_direction_estimation"] = global_light_direction_estimation
            painting["analysis"]["global_estimation_std_degrees"] = global_estimation_std_degrees
            break
    data = convert_ndarray(data)
    safe_write_json(data, json_filepath)

add_remaining_global_info(painting_id, global_light_direction_estimation, global_estimation_std_degrees)



utils.plot_painting_contours_C(painting_id, json_filepath, img_path, scale=700, plot_contours_flag=True, plot_only_contours=False, plot_all_contours=True, contour_ids=None)






































##############################################################################
# UPDATE EXISTING CONTOURS WITH LOOPING
##############################################################################

import glob
from tqdm import tqdm

failed_contours = []
parent_dir = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/images'
sigma = 30
json_filepath = "C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"
with open(json_filepath, 'r') as f:
    data = json.load(f)

for painting in tqdm(data.get("paintings", [])):
    painting_id = painting.get("painting_id")
    contours_list = painting.get("analysis", {}).get("contours", [])
    image_files = glob.glob(os.path.join(parent_dir, f"{painting_id}.*"))
    if image_files:
        img_path = image_files[0]  # Take the first match.
        image = cv2.imread(img_path)
    else:
        print(f"No image file found for painting_id {painting_id} in {parent_dir}")
        continue
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not contours_list:
        print(f"No contours found for painting_id={painting_id}.")
        continue
    new_contours = []
    for contour in contours_list:
        contour_id = contour.get("contour_id")
        try:
            # Extract contour parameters.
            smoothness = contour.get("smoothness", None)
            N = contour.get("Parameter_N", None)
            contour_coordinates = contour.get("contour_coordinates", None)
            contour_type = contour.get("contour_type", None)
            belongs_to_person = contour.get("belongs_to_person", None)
            if not contour_coordinates or len(contour_coordinates) < 2:
                raise ValueError("Invalid contour coordinates.")
            (ground_truth_subpixel, light_direction_estimation, angle_std_degs, 
             empirical_std_deg, normals, x_coordinates, y_coordinates) = utils.process_contour_C(
                    img_path, gray_image, contour_coordinates, smoothness, N, sigma)
            result = utils.compute_shading_delta_metric(ground_truth_subpixel, light_direction_estimation, normals,sigma=sigma, max_sigma=100, sigma_step=5)
            if result is None:
                failed_contours.append((painting_id, contour_id))
                print(f"Contour_id={contour_id} in painting_id={painting_id} removed (delta metric=NONE).")
                continue
            else:
                mse, delta, meas_block, theo_block, I_meas_norm, I_theo_norm, sigma = result
            level_of_noise = utils.compute_noise_metric(ground_truth_subpixel)
            _, _, _, _, std_mse_deg = utils.get_Lightdirection_pdf([light_direction_estimation], [I_meas_norm], [normals])
            meas_smooth = gaussian_filter1d(ground_truth_subpixel, sigma=sigma)
            norm_smoothed_luminance_distribution = meas_smooth / meas_smooth.max()
            brightness_range = float(np.round(norm_smoothed_luminance_distribution.max() - norm_smoothed_luminance_distribution.min(), 4))
            normal_vectors_span = utils.compute_angle_range(normals)
            # Compute spherical harmonics coefficients.
            coefficients_reg = utils.compute_sh_coefficients(normals, norm_smoothed_luminance_distribution)
            # Prepare new contour data.
            new_contour_data = { 
                "contour_id": contour_id,   
                "contour_coordinates": contour_coordinates,
                "smoothness": smoothness,
                "Parameter_N": N,
                "brightness_range": [round(brightness_range, 4)],
                "normal_vectors_span": [normal_vectors_span],
                "contour_type": contour_type,
                "belongs_to_person": belongs_to_person,
                "light_direction_estimation": [light_direction_estimation],
                "empirical_std_deg": [empirical_std_deg],
                "estimation_std_degrees": [angle_std_degs],
                "std_mse_deg": [std_mse_deg],
                "mean_squared_error": [mse],
                "shading_rate": [delta],
                "level_of_noise": [level_of_noise],
                "spherical_harmonics_coeffs": [coefficients_reg]
            }
            new_contours.append(new_contour_data)
        except Exception as e:
            failed_contours.append((painting_id, contour.get("contour_id"), str(e)))
            print(f"Error processing painting_id={painting_id}, contour_id={contour.get('contour_id')}: {e}")
            continue
    # Update the contours list for this painting.
    painting["analysis"]["contours"] = new_contours
    # Compute global estimation for the painting and update its analysis.
    global_light_direction_estimation, global_estimation_std_degrees = utils.compute_global_estimation(painting_id, img_path, json_filepath, utils.process_contour_C, gray_image)
    painting.setdefault("analysis", {})["global_light_direction_estimation"] = (global_light_direction_estimation.tolist() if global_light_direction_estimation is not None else None)
    painting["analysis"]["global_estimation_std_degrees"] = global_estimation_std_degrees

# Once all paintings are processed, convert any numpy arrays and write the JSON file.
data = utils.convert_ndarray(data)
utils.safe_write_json(data, json_filepath)




##############################################################################
# Code to re-order contour's ids for each painting

json_filepath = "C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"
with open(json_filepath, 'r') as f:
    data = json.load(f)

paintings = data.get("paintings", [])
for painting in paintings:
    painting_id = painting.get("painting_id")
    analysis = painting.setdefault("analysis", {})
    contours_list = analysis.setdefault("contours", [])
    if not contours_list:
        continue
    for new_id, contour in enumerate(contours_list, start=1):
        contour["contour_id"] = new_id
    print(f"Painting_id={painting_id}: reassigned {len(contours_list)} contours IDs from 1 to {len(contours_list)}.")


# 3) Write the updated data back to file
with open(json_filepath, 'w') as f:
    json.dump(data, f, indent=2)



##############################################################################
# BACKGROUDS ANALYSIS: DARKNESS
##############################################################################
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress

def compute_darkness(img_path):
    """
    darkness = 255 - (average of Value channel over valid pixels).
    """
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    # Check if the image has an alpha channel (4th channel)
    has_alpha = (len(image.shape) == 3 and image.shape[2] == 4)
    # Separate channels
    if has_alpha:
        b_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        r_channel = image[:, :, 2]
        a_channel = image[:, :, 3]
    else:
        b_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        r_channel = image[:, :, 2]
        a_channel = None
    # Create a boolean mask for valid pixels: not near-white, and not transparent
    # "Near-white" threshold: R,G,B > 250
    near_white_mask = (
        (r_channel > 250) &
        (g_channel > 250) &
        (b_channel > 250)
    )
    if has_alpha:
        valid_mask = (~near_white_mask) & (a_channel > 0)
    else:
        valid_mask = ~near_white_mask
    if not np.any(valid_mask):
        return None
    # Build a 3-channel BGR image (ignore alpha) to convert to HSV
    bgr = np.dstack([b_channel, g_channel, r_channel])
    # Convert BGR -> HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Extract the Value channel (hsv[:,:,2])
    value_channel = hsv[:, :, 2]
    # Compute mean value on valid pixels
    mean_value = np.mean(value_channel[valid_mask])
    # Darkness metric in HSV: darkness = 255 - mean(Value)
    darkness = 255 - mean_value
    # or artenatively, for Y luminance
    # luminance_Y = 0.2126 * r_channel + 0.7152 * g_channel + 0.0722 * b_channel
    # # Extract valid pixels
    # valid_Y_values = luminance_Y[valid_mask]
    # # Compute perceptual darkness
    # mean_Y = np.mean(valid_Y_values)
    # darkness = 100 - (mean_Y / 255) * 100 
    return darkness


path = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/bg/52.jpg' 
compute_darkness(path)


parent_dir = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/bg_resized'
json_path = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json'

# Read JSON metadata
with open(json_path, 'r', encoding='utf-8') as f:
    metadata_list = json.load(f)

results = []
paintings_metadata = metadata_list["paintings"]
for item in paintings_metadata:
    painting_id = item['painting_id']
    date = item.get('date', None) 
    technique = item.get("technique", "")
    painting_type = item.get("type", "")
    # We can try both .jpg and .png
    jpg_path = os.path.join(parent_dir, f"{painting_id}.jpg")
    png_path = os.path.join(parent_dir, f"{painting_id}.png")
    if os.path.exists(jpg_path):
        darkness = compute_darkness(jpg_path)
    elif os.path.exists(png_path):
        darkness = compute_darkness(png_path)
    else:
        darkness = None
    results.append({
        "painting_id": painting_id,
        "date": date,
        "technique": technique,
        "type": painting_type,
        "darkness": darkness
                })


# Filter out entries with no darkness or no date
filtered_results = [r for r in results if r['darkness'] is not None]

def clean_date(date_str):
    if not date_str:
        return None  # Handle None or empty case
    # Normalize and remove common prefixes
    date_str = date_str.lower()
    date_str = re.sub(r'c[.]?\s?|ca[.]?\s?', '', date_str)  # Remove "c.", "ca.", "c ", "ca "
    # Find all numbers in the string
    years = re.findall(r'\d{4}', date_str)
    if years:
        return int(years[0])  # Only keep the first detected four-digit year
    return None  # If no valid year is found, return None

for r in filtered_results:
    r['date'] = clean_date(r['date'])

for r in filtered_results:
    print(r['date'])


# Sort by date
filtered_results.sort(key=lambda x: x['date'])

for result in filtered_results:
    print(result['painting_id'], result['date'])

# Extract arrays for plotting
years = [r['date'] for r in filtered_results]
darkness_values = [r['darkness'] for r in filtered_results]

# Plot darkness vs. year
plt.figure(figsize=(8, 5))
plt.scatter(years, darkness_values, color='blue', label='Darkness')
plt.title("Caravaggio Background Darkness Over Time")
plt.xlabel("Year")
plt.ylabel("Darkness (255 - mean grayscale)")
plt.legend()
plt.grid(True)
plt.show()


def plot_darkness_regression(dates, darkness_values):
    # Ensure inputs are NumPy arrays
    dates = np.array(dates)
    darkness_values = np.array(darkness_values)
    # Fit a linear regression line
    slope, intercept, r_value, p_value, std_err = linregress(dates, darkness_values)
    # Generate fitted values for plotting
    regression_line = slope * dates + intercept
    # Plot original data
    plt.figure(figsize=(8, 5))
    plt.scatter(dates, darkness_values, color='blue', label="Observed Darkness")
    # Plot regression line
    plt.plot(dates, regression_line, color='red', linestyle='dashed', label=f"Regression (R²={r_value**2:.2f})")
    # Titles & Labels
    plt.title("Background Darkness Over Time")
    plt.xlabel("Year")
    plt.ylabel("Darkness")
    plt.legend()
    plt.grid(True)
    # Set x-axis to only show integer years
    years_range = np.arange(int(np.min(dates)), int(np.max(dates)) + 1, 1)
    plt.xticks(years_range, rotation=45)  # Ensure only whole years appear on the axis
    # Show plot
    plt.show()

plot_darkness_regression(years, darkness_values)

from PIL import Image

input_path = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/bg/80.jpg'
output_path = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/bg_resized/80.jpg'

with Image.open(input_path) as img:
    width, height = img.size
    # Check if resizing is needed
    if width > 4000 or height > 4000:
        # Maintain aspect ratio
        if width > height:
            new_width = 4000
            new_height = int((4000 / width) * height)
        else:
            new_height = 4000
            new_width = int((4000 / height) * width)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img.save(output_path)


##############################################################################
# BACKGROUDS ANALYSIS: COLOR PALETTE
##############################################################################




##############################################################################
# PAINTING ANALYSIS: COLOR PALETTE
##############################################################################


