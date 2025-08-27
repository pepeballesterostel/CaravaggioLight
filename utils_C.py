'''
utils_C.py to store functions for Caravaggio eouvre analysis using the occluding contour algorithm.
'''


import numpy as np
import json
import os
import cv2
import tempfile
from typing import Any, Dict, List, Tuple
from math import sqrt

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
from scipy.ndimage import sobel
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import sobel
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates
from scipy.special import erf

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib import colormaps


def show_painting_summary(painting_id, json_filepath="C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"):
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    target_painting = None
    for painting in data["paintings"]:
        if painting["painting_id"] == painting_id:
            target_painting = painting
            break
    if not target_painting:
        print(f"No painting found with painting_id={painting_id}.")
        return
    print(f"Painting ID: {target_painting['painting_id']} — {target_painting['title']} ({target_painting['location']})")
    contours_list = target_painting["analysis"]["contours"]
    print(f"Number of contours: {len(contours_list)}")
    for c in contours_list:
        c_id = c["contour_id"]
        c_type = c.get("contour_type", "N/A")  # 'N/A' if none specified
        c_person = c.get("belongs_to_person", "N/A")  # 'N/A' if none specified
        print(f"  • contour_id={c_id}, contour_type={c_type}, belongs_to_person={c_person}")


def extract_contour_ui(image_rgb):
    instructions = """
    Instructions:
    1. Press 'p' to enter point selection mode.
    2. Press 'z' to enter zoom mode.
    3. Press 'f' to finalize the current contour.
    4. Press 'u' to undo the last point.
    5. Press 'r' to remove the last contour.
    6. Press 'h' for help.
    7. Press 'q' to quit.
    """
    # Global variables to store contours and mode
    contours = []  # List to store all contours
    current_contour = []  # Points in the current contour
    mode = 'zoom'  # Start in zoom mode
    xlim, ylim = None, None  # To preserve zoom state
    text_box = None  # For help instructions display
    help_visible = False  # Track if help is visible
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    ax.axis('off')  
    # Instructions for the user
    def toggle_instructions():
        nonlocal  text_box, help_visible
        if help_visible:
            if text_box:
                text_box.remove() 
            help_visible = False
        else:
            text_box = plt.text(0.05, 0.95, instructions, transform=ax.transAxes, fontsize=8,
                                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
            help_visible = True
        plt.draw()
    def onclick(event):
        nonlocal  current_contour, mode
        if mode == 'select' and event.xdata is not None and event.ydata is not None:
            ix, iy = event.xdata, event.ydata
            current_contour.append((ix, iy))
            plt.plot(ix, iy, 'ro', markersize=1)
            plt.draw()
        elif mode == 'zoom':
            pass  # In zoom mode, don't record points
    def plot_contour(contour):
        contour = np.array(contour)
        plt.scatter(contour[:, 0], contour[:, 1], color='r', s=1) 
        plt.draw()
    def redraw_image():
        nonlocal  xlim, ylim
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.clear()
        ax.imshow(image_rgb)
        ax.axis('off')
        for contour in contours:
            plot_contour(contour)
        for point in current_contour:
            plt.plot(point[0], point[1], 'ro', markersize=0.1) 
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.draw()
    def onkey(event):
        nonlocal  mode, current_contour, contours
        if event.key == 'z':  # Switch to zoom mode
            mode = 'zoom'
            print("Switched to zoom mode")
        elif event.key == 'p':  # Switch to point selection mode
            mode = 'select'
            print("Switched to point selection mode")
        elif event.key == 'f':  # Finalize the current contour
            if len(current_contour) > 2:
                contours.append(current_contour.copy())
                plot_contour(current_contour)
                current_contour = []  # Reset for a new contour
                print(f"Contour finalized. Total contours: {len(contours)}")
            else:
                print("Not enough points to finalize the contour.")
        elif event.key == 'u':  # Undo the last point
            if current_contour:
                removed_point = current_contour.pop()
                print(f"Removed point: {removed_point}")
                redraw_image()
            else:
                print("No points to undo.")
        elif event.key == 'r':  # Remove the last contour
            if contours:
                removed_contour = contours.pop()
                print(f"Removed last contour. Total contours left: {len(contours)}")
                redraw_image()
            else:
                print("No contours to remove.")
        elif event.key == 'h':  # Show help instructions
            toggle_instructions()
        elif event.key == 'q':  # Quit
            plt.close()
    # Display image for point selection
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    return contours[0]

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

def correct_normal_orientations(x, y, spline_x, spline_y, normals):
    """
    Ensure that all normals point outward by checking their direction relative to the centroid.
    """
    # Step 1: Calculate the centroid of the contour
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    # Step 2: Check each normal to see if it points inward
    for i in range(len(spline_x)):
        # Vector from centroid to the current spline point
        vec_to_centroid = np.array([centroid_x - spline_x[i], centroid_y - spline_y[i]])
        dot_product = np.dot(normals[i], vec_to_centroid)
        # If dot product is positive, normal points inward, so we flip it
        if dot_product > 0:
            normals[i] = -normals[i]
    return normals

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
    # img = cv2.imread(img_path)
    # if img is None:
    #     raise FileNotFoundError(img_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img_arr.shape
    ax.imshow(img_arr, cmap='gray', alpha=1, extent=[0, w, h, 0])
    # ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.scatter(x, y, c='red', s=point_size, label='Contour pts')
    # ---- 3. draw every normal as an arrow ------------------------------
    for xi, yi, (nx, ny) in zip(x, y, N):
        ax.arrow(xi, yi, nx*arrow_scale, ny*arrow_scale,
                 color='skyblue', head_width=3, length_includes_head=True)
    ax.axis('off')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

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
    
def compute_light_direction_C(
        normal_list,            
        lum_list,                 
        max_iter: int = 10
    ):
    # -------- 1. concatenate all patches ---------------------------------
    if isinstance(normal_list, np.ndarray):
        normal_list = [normal_list]
    if isinstance(lum_list, np.ndarray):
        lum_list = [lum_list]
    normals = np.vstack(normal_list).astype(np.float32)
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


def compute_empirical_std(normals, ground_truth_subpixel, max_iter, N_max):
    # List to store the estimated light direction for each partition
    light_directions = []
    # Total number of contour points; assumed they are ordered.
    total_points = len(ground_truth_subpixel)
    for N in range(1, N_max + 1):
        # Compute partition boundaries for the current N.
        indices = np.arange(total_points)
        partitions = np.array_split(indices, N)
        # For each partition, slice the arrays and compute the estimate.
        for part in partitions:
            # Ensure at least a minimal number of points
            if len(part) < 3:
                continue
            part_normals = normals[part]
            part_ground_truth = ground_truth_subpixel[part]
            # Compute light direction for this partition.
            L_est, _ = compute_light_direction_C(
                part_normals, 
                part_ground_truth, 
                max_iter
            )
            # L_est should be a 2D unit vector.
            light_directions.append(L_est)
    angles = np.array([np.arctan2(L[1], L[0]) for L in light_directions])
    # Compute the circular standard deviation.
    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    if R > 0:
        circ_std = np.sqrt(-2 * np.log(R))
    else:
        circ_std = float('inf')
    # Convert from radians to degrees.
    circ_std_deg = np.degrees(circ_std)
    return circ_std_deg

def normalize_to_01(arr):
    a_min, a_max = arr.min(), arr.max()
    if a_max > a_min:
        return (arr - a_min)/(a_max - a_min)
    else:
        # no variation
        return np.zeros_like(arr)
    
    
def find_01_block(intensities, lo=0.1, hi=0.9):
    """
    Given a smoothed brightness array, automatically detect 
    whether it's mostly increasing or decreasing from index 0 to -1,
    then find a single contiguous block from fraction=lo to fraction=hi.
    
    Returns (i_lo, i_hi) indices for that block, or (None, None) if not found.
    """
    N = len(intensities)
    if N < 2:
        return None, None
    first_val = intensities[0]
    last_val = intensities[-1]
    # Decide if we do "increasing" or "decreasing" detection
    # If last_val > first_val => distribution is mostly rising
    # else => mostly falling
    is_increasing = (last_val >= first_val)
    i_lo = None
    i_hi = None
    if is_increasing:
        # We want first crossing of lo, then first crossing of hi after that
        current_val = first_val
        state = 'looking_for_lo'
        for i in range(N-1):
            y0, y1 = intensities[i], intensities[i+1]
            # if we haven't found lo yet
            if state == 'looking_for_lo':
                if (y0 < lo <= y1) or (y0 > lo >= y1):  
                    # crossing lo
                    if y1 != y0:
                        alpha = (lo - y0)/(y1 - y0)
                    else:
                        alpha = 0.0
                    i_lo = i + alpha
                    state = 'looking_for_hi'
            elif state == 'looking_for_hi':
                if (y0 < hi <= y1) or (y0 > hi >= y1):
                    # crossing hi
                    if y1 != y0:
                        alpha = (hi - y0)/(y1 - y0)
                    else:
                        alpha = 0.0
                    i_hi = i + alpha
                    break
    else:
        # Decreasing approach: we want first crossing of hi, then first crossing of lo after that
        state = 'looking_for_hi'
        for i in range(N-1):
            y0, y1 = intensities[i], intensities[i+1]
            if state == 'looking_for_hi':
                if (y0 > hi >= y1) or (y0 < hi <= y1):
                    # crossing hi
                    if y1 != y0:
                        alpha = (hi - y0)/(y1 - y0)
                    else:
                        alpha = 0.0
                    i_lo = i + alpha  # reuse i_lo to store the "higher fraction"
                    state = 'looking_for_lo'
            elif state == 'looking_for_lo':
                if (y0 > lo >= y1) or (y0 < lo <= y1):
                    # crossing lo
                    if y1 != y0:
                        alpha = (lo - y0)/(y1 - y0)
                    else:
                        alpha = 0.0
                    i_hi = i + alpha
                    break
    # i_lo < i_hi if is_increasing, else i_lo > i_hi if is_decreasing
    return i_lo, i_hi

def compute_shading_delta_metric(ground_truth_subpixel, light_direction_estimation, normals, sigma=10, max_sigma=100, sigma_step=5, lo_frac=0.1, hi_frac=0.9):
    '''
    We compute the normal vector angle span in intensity w.r.t. pixel index distribution. 
    '''
    def get_index_slice(lo_idx, hi_idx):
        # If either is None, we can't proceed
        if lo_idx is None or hi_idx is None:
            return []
        if lo_idx < hi_idx:
            start = int(np.ceil(lo_idx))
            end = int(np.floor(hi_idx))
            return range(start, end+1)
        else:
            start = int(np.ceil(hi_idx))
            end = int(np.floor(lo_idx))
            return range(start, end+1)
    current_sigma = sigma
    while current_sigma <= max_sigma:
        try:
            meas_smooth = gaussian_filter1d(ground_truth_subpixel, sigma=current_sigma)
            theoretical_light = np.clip(light_direction_estimation @ normals.T, 0, 1)
            I_meas_norm = normalize_to_01(meas_smooth)
            I_theo_norm = normalize_to_01(theoretical_light)
            mse = np.mean((I_meas_norm - I_theo_norm)**2)
            i_lo_meas, i_hi_meas = find_01_block(I_meas_norm, lo_frac, hi_frac)
            i_lo_theo, i_hi_theo = find_01_block(I_theo_norm, lo_frac, hi_frac)
            meas_block = list(get_index_slice(i_lo_meas, i_hi_meas))
            theo_block = list(get_index_slice(i_lo_theo, i_hi_theo))
            meas_angle_range = compute_angle_range(normals[meas_block])
            theo_angle_range = compute_angle_range(normals[theo_block])
            delta = round(meas_angle_range - theo_angle_range, 4)
            return round(mse,4), delta, meas_block, theo_block, I_meas_norm, I_theo_norm, current_sigma
        except TypeError as te:
            # This likely means i_lo_meas or i_hi_meas was None, or something else
            print(f"TypeError encountered with sigma={current_sigma}: {te}")
        except Exception as e:
            print(f"Error encountered with sigma={current_sigma}: {e}")
        # If we failed, increase sigma by sigma_step and try again
        current_sigma += sigma_step
    print(f"Failed to compute shading delta metric even up to sigma={max_sigma}. Returning None.")
    return None

def compute_noise_metric(brightness):
    '''
    metric to compute the extent to which a contour contour is noisy or highly stylised. 
    We do this by tracking the number of sign changes in the brightness gradient.
    '''
    diff = [brightness[i+1] - brightness[i] for i in range(len(brightness)-1)]
    sign_changes = 0
    for i in range(len(diff)-1):
        if diff[i]*diff[i+1] < 0:
            sign_changes += 1
    return sign_changes

def compute_angle_range(normals):
    angles = np.arctan2(normals[:, 1], normals[:, 0])
    angles = np.degrees(angles)
    sorted_angles = np.sort(angles)
    diffs = np.diff(sorted_angles)
    wrap_diff = (sorted_angles[0] + 360.0) - sorted_angles[-1]
    all_diffs = np.append(diffs, wrap_diff)
    largest_gap = np.max(all_diffs)
    angle_range = 360.0 - largest_gap
    return round(angle_range, 2)

def sh_basis_cartesian(normals):
    '''
    Takes the normal vectors and calculates the sh basis for 5 coefficients for each vertice. Returns a (n_vertices, n_sh_coeffs) matrix.
    The first order is constant, then linear, then quadratic polynomials.
    '''
    # Define constants
    n_normals = len(normals)
    # compute SH basis
    basis = np.zeros((n_normals, 5))
    # att = np.pi*np.array([1, 2.0/3.0, 1/4.0])
    for i, normal in enumerate(normals):
        x, y = normal
        basis[i,0] = 0.282095
        basis[i,1] = 0.488603 * y
        basis[i,2] = 0.488603 * x
        basis[i,3] = 1.092548 * x * y
        basis[i,4] = 0.546274 * (x ** 2 - y ** 2)
    return basis

def compute_sh_coefficients(normals, luminance, lambda_reg = 0.1):
    basis = sh_basis_cartesian(normals)
    C = np.diag([1, 2, 2, 3, 3])
    regularized_basis = basis.T @ basis + lambda_reg * C
    regularized_inverse = np.linalg.inv(regularized_basis)
    coefficients_reg = regularized_inverse @ basis.T @ luminance
    return coefficients_reg

def convert_ndarray(obj):
    """
    Recursively convert numpy arrays in `obj` to lists
    so that it becomes JSON-serializable.
    """
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def safe_write_json(data, json_filepath):
    # Create a temp file in the same directory
    dir_name, base_name = os.path.split(json_filepath)
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False) as tmp:
        temp_path = tmp.name
        json.dump(data, tmp, indent=2)
    # If we reach here, the dump succeeded with no error
    os.replace(temp_path, json_filepath)  # Overwrite the old file atomically

def add_contour(painting_id, new_contour_data, json_filepath="C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"):
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    for painting in data["paintings"]:
        if painting["painting_id"] == painting_id:
            contours_list = painting["analysis"].get("contours", [])
            new_id = len(contours_list) + 1
            new_contour = {
                "contour_id": new_id,
                "contour_coordinates": new_contour_data["contour_coordinates"],
                "smoothness": new_contour_data["smoothness"],
                "Parameter_N": new_contour_data["Parameter_N"],
                "brightness_range": new_contour_data["brightness_range"],
                "normal_vectors_span": new_contour_data["normal_vectors_span"],
                "contour_type": new_contour_data["contour_type"],
                "belongs_to_person": new_contour_data["belongs_to_person"],
                "light_direction_estimation": new_contour_data["light_direction_estimation"],
                "empirical_std_deg": new_contour_data["empirical_std_deg"],
                "estimation_std_degrees": new_contour_data["estimation_std_degrees"],
                "std_mse_deg": new_contour_data["std_mse_deg"],
                "mean_squared_error": new_contour_data["mean_squared_error"],
                "shading_rate": new_contour_data["shading_rate"],
                "level_of_noise": new_contour_data["level_of_noise"],
                "spherical_harmonics_coeffs": new_contour_data["spherical_harmonics_coeffs"]
            }
            contours_list.append(new_contour)
            painting["analysis"]["contours"] = contours_list
            break
    data = convert_ndarray(data)
    safe_write_json(data, json_filepath)

def replace_contour(painting_id: int, contour_id: int, new_contour_data: dict, json_filepath="C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"):
    """
    Overwrite the entire contour's data with 'new_contour_data'. The only field we keep from the old contour is the 'contour_id'.
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    # Locate painting
    target_painting = None
    for painting in data["paintings"]:
        if painting["painting_id"] == painting_id:
            target_painting = painting
            break
    if target_painting is None:
        raise ValueError(f"No painting found with painting_id={painting_id}.")
    # Locate contour
    contours_list = target_painting["analysis"]["contours"]
    index = None
    for i, c in enumerate(contours_list):
        if c["contour_id"] == contour_id:
            index = i
            break
    if index is None:
        raise ValueError(f"No contour with contour_id={contour_id} found in painting_id={painting_id}.")
    # Keep the old contour_id
    new_data_to_store = {
        "contour_id": contour_id
    }
    # Merge with new_contour_data
    new_data_to_store.update(new_contour_data)
    # Overwrite the old contour
    contours_list[index] = new_data_to_store
    # Convert arrays
    data = convert_ndarray(data)
    # Safe write
    safe_write_json(data, json_filepath)
    return True

def delete_contour(painting_id: int, contour_id: int, json_filepath="C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"):
    """
    Remove the contour with the given 'contour_id' from the painting's
    'analysis["contours"]' list.
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    # Locate painting
    target_painting = None
    for painting in data["paintings"]:
        if painting["painting_id"] == painting_id:
            target_painting = painting
            break
    if target_painting is None:
        raise ValueError(f"No painting found with painting_id={painting_id}.")
    # Locate and remove the contour
    contours_list = target_painting["analysis"]["contours"]
    old_length = len(contours_list)
    new_contours = [c for c in contours_list if c["contour_id"] != contour_id]
    if len(new_contours) == old_length:
        raise ValueError(f"No contour with contour_id={contour_id} found in painting_id={painting_id}.")
    # Overwrite with the filtered list
    target_painting["analysis"]["contours"] = new_contours
    # Convert arrays
    data = convert_ndarray(data)
    # Safe write
    safe_write_json(data, json_filepath)
    return True

def extract_light_direction_ui(image_path):
    '''
    This function allows a user to interactively select two points on an image 
    and then returns the normalized vector (direction) from the first point to the second. 
    '''
    # Load the image using OpenCV
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # Display the image with matplotlib to allow zooming and point selection
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title("Press 'z' to zoom, 'p' to select points")
    points = []
    mode = 'zoom'  # Start in zoom mode by default
    def on_key(event):
        nonlocal mode
        if event.key == 'z':
            mode = 'zoom'
            plt.title("Zoom Mode: Press 'p' to switch to Point Selection Mode")
            fig.canvas.draw()
        elif event.key == 'p':
            mode = 'point'
            plt.title("Point Selection Mode: Click to select points, Press 'z' to switch to Zoom Mode")
            fig.canvas.draw()
    def onclick(event):
        if mode == 'point' and event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')  # Plot the selected point
            fig.canvas.draw()
            if len(points) == 2:
                plt.close()  # Close the plot once 2 points are selected
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    # Ensure two points have been selected
    if len(points) != 2:
        raise ValueError("Two points were not selected. Please try again.")
    # Calculate the direction vector from point 1 to point 2
    p1 = np.array(points[0])
    p2 = np.array(points[1])
    direction_vector = p2 - p1
    # Normalize the direction vector
    norm = np.linalg.norm(direction_vector)
    if norm == 0:
        raise ValueError("The two points selected are identical, resulting in a zero-length direction vector.")
    normalized_direction_vector = direction_vector / norm
    return normalized_direction_vector


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

def vis_global_light_estimation_C(img_path, direction, x_coordinates, y_coordinates, angle_std_degs=None, scale=1000, visual_factor = 2.0, plot_Ld=True, plot_contours_flag=False, plot_only_contours=False):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    org_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if org_img is None:
        raise ValueError(f"Could not read image from {img_path}")
    ax.imshow(
        org_img,
        cmap='gray',
        alpha=0.7,
        extent=[0, org_img.shape[1], org_img.shape[0], 0]
    )
    img_center = (org_img.shape[1] // 2, org_img.shape[0] // 2)
    if plot_only_contours:
        ax.plot(x_coordinates, y_coordinates, '.', markersize=4, color='green')
    else:
        # Possibly plot contour points
        if plot_contours_flag:
            ax.plot(x_coordinates, y_coordinates, '.', markersize=2, color='green')
        # 4) Light direction arrow
        if plot_Ld and direction is not None:
            center_x = img_center[0]
            center_y = img_center[1]
            dx = direction[0] * scale
            dy = direction[1] * scale
            # draw the arrow
            ax.arrow(
                center_x,
                center_y,
                dx,
                dy,
                head_width=30,
                head_length=30,
                width=6,
                fc='yellow',
                ec='yellow',
                zorder=5
            )
            angle_degs = np.degrees(np.arctan2(direction[1], direction[0]))
            if  angle_std_degs is not None:
                enlarged_std = angle_std_degs * visual_factor
                angle_min = angle_degs - enlarged_std
                angle_max = angle_degs + enlarged_std
                # Make a wedge spanning [angle_min, angle_max] in degrees
                # with radius=scale (or you can pick a different radius).
                wedge = Wedge(
                    (center_x, center_y),  # center
                    scale,                 # radius
                    angle_min,
                    angle_max,
                    color='yellow',
                    alpha=0.2,
                    zorder=4
                )
                ax.add_patch(wedge)
    # 6) Axes limits & flip y-axis to match image coords
    ax.set_xlim([-0.1 * org_img.shape[1], 1.1 * org_img.shape[1]])
    ax.set_ylim([1.1 * org_img.shape[0], -0.1 * org_img.shape[0]])  # invert y
    ax.axis('off')  # Hide axes & ticks
    plt.show()

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

def add_remaining_info(painting_id, depicted_light_direction_evidence, number_of_people, global_light_direction_estimation, global_estimation_std_degrees, json_filepath="C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"):
    # 1. Load the master JSON
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    # 2. Locate the target painting
    for painting in data["paintings"]:
        if painting["painting_id"] == painting_id:
            painting["analysis"]["depicted_light_direction_evidence"] = depicted_light_direction_evidence
            painting["analysis"]["num_people"] = number_of_people
            painting["analysis"]["global_light_direction_estimation"] = global_light_direction_estimation
            painting["analysis"]["global_estimation_std_degrees"] = global_estimation_std_degrees
            break
    data = convert_ndarray(data)
    safe_write_json(data, json_filepath)


def extract_direction(raw):
    if raw is None:
        return None
    # If someone accidentally put a dict in here:
    if isinstance(raw, dict) and "light_direction" in raw:
        raw = raw["light_direction"]
    # Drill down through nested lists/tuples:
    while isinstance(raw, (list, tuple)) and len(raw) > 0 \
          and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    # Now check final shape:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            dx = float(raw[0])
            dy = float(raw[1])
            return dx, dy
        except (TypeError, ValueError):
            return None
    return None

def vis_contour_light_estimations(data, painting_id, img_path,
                                  scale=700,
                                  uncertainty_factor=1.0, color = 'Set2',
                                  bg='white', save_fig=False):
    # 1) Find painting
    painting = next((p for p in data["paintings"]
                     if p["painting_id"] == painting_id), None)
    if painting is None:
        raise ValueError(f"painting_id={painting_id} not found")
    # 2) Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")
    h, w = img.shape
    # 3) Setup figure
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.imshow(img, cmap='gray', alpha=1, extent=[0, w, h, 0])
    # 4) Plot contours + arrows + uncertainty wedges
    contours = painting.get("analysis", {}).get("contours", [])
    cmap = colormaps[color]
    for i, c in enumerate(contours):
        raw = c.get("contour_coordinates", [])
        # flatten all segments into pts
        pts = []
        for seg in raw:
            if isinstance(seg, (list, tuple)):
                if seg and isinstance(seg[0], (list, tuple)):
                    pts.extend(seg)
                elif len(seg) == 2 and all(isinstance(v, (int, float)) for v in seg):
                    pts.append(seg)
        if not pts:
            continue
        coords_all = np.array(pts, dtype=float)
        color = cmap(i % cmap.N)
        # draw each segment
        for seg in raw:
            if isinstance(seg, (list, tuple)) and seg and isinstance(seg[0], (list, tuple)):
                coords = np.array(seg, dtype=float)
                ax.plot(coords[:,0], coords[:,1], '-', color=color, linewidth=1)
        # extract a valid (dx,dy)
        raw_ld = c.get("light_direction_estimation")
        maybe = extract_direction(raw_ld)
        if maybe is None:
            maybe = extract_direction(c.get("light_direction"))
        if maybe is None:
            continue
        dx, dy = maybe
        # place arrow at center of first segment
        first_seg = raw[0] if len(raw) > 1 else raw[0]
        if isinstance(first_seg[0], (list, tuple)):
            coords0 = np.array(first_seg, dtype=float)
        else:
            coords0 = np.array([first_seg], dtype=float)
        cx, cy = coords0.mean(axis=0)
        # draw arrow
        ax.arrow(cx, cy,
                 dx*scale, dy*scale,
                 head_width=15, head_length=15, width=2,
                 length_includes_head=True,
                 fc=color, ec=color, zorder=5)
        # draw uncertainty wedge
        std = c.get("empirical_std_deg", 0)
        if isinstance(std, (list, tuple)) and len(std) == 1:
            std = std[0]
        angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        delta = std * uncertainty_factor
        wedge = Wedge((cx, cy), scale, angle-delta, angle+delta,
                      color=color, alpha=0.2, zorder=4)
        ax.add_patch(wedge)
    # 5) Finalize
    ax.set_xlim(-0.05*w, 1.05*w)
    ax.set_ylim(1.05*h, -0.05*h)
    ax.axis('off')
    plt.show()
    if save_fig:
        # High-res PNG
        png_name = f"painting_{painting_id}_contours.png"
        fig.savefig(png_name,
                    format="png",
                    dpi=600,
                    bbox_inches="tight")
        print(f"Saved")

# Code to re-order contour's ids for each painting in the json dataset
def order_json_ids(json_filepath = "C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"):
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
    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=2)


def plot_single_contour_light(
    img,
    contour_segments,
    light_direction,
    std_deg,
    scale=700,
    uncertainty_factor=1.0,
    contour_color='green',
    arrow_color='yellow',
    bg='white',
    save_fig=False,
    fig_name='contour_light.png'
):
    # — load/prepare image —
    if isinstance(img, str):
        img_arr = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if img_arr is None:
            raise ValueError(f"Cannot load image at {img}")
    else:
        img_arr = img.copy()
        if img_arr.ndim == 3:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    h, w = img_arr.shape
    # — compute centroid over all segments —
    all_pts = np.vstack(contour_segments[0])
    cx, cy = all_pts.mean(axis=0)
    # — set up figure —
    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.imshow(img_arr, cmap='gray', alpha=1, extent=[0, w, h, 0])
    # — plot each segment in green —
    for seg in contour_segments:
        seg = np.asarray(seg, dtype=float)
        ax.plot(seg[:,0], seg[:,1], '-', color=contour_color, linewidth=3)
    # — draw the arrow for the estimated light direction —
    dx, dy = light_direction
    ax.arrow(
        cx, cy,
        dx*scale, dy*scale,            # invert dy for image coords
        head_width=30, head_length=30, width=12,
        length_includes_head=True,
        fc=arrow_color, ec=arrow_color, zorder=4
    )
    # — draw the uncertainty wedge centered at the arrow origin,
    #    aligned with the arrow’s direction —
    # use flipped dy to match arrow orientation
    angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    delta = std_deg * uncertainty_factor
    wedge = Wedge(
        (cx, cy),       # same origin as arrow
        scale,          # radius matches arrow length
        angle - delta,
        angle + delta,
        color=arrow_color,
        alpha=0.2,
        zorder=4
    )
    ax.add_patch(wedge)
    # — finalize —
    ax.set_xlim(-0.05*w, 1.05*w)
    ax.set_ylim(1.05*h, -0.05*h)
    ax.axis('off')
    plt.tight_layout()
    if save_fig:
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {fig_name}")
    plt.show()