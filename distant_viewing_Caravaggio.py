'''
This file is for distant viewing on Caravaggio's oeuvre.

Extract lighting information from paintings and incorporate it into a JSON file.
'''

import json
import os
import cv2
import utils_C
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Tuple

#############################################################################################################
#############################################################################################################
# Load the master JSON
json_filepath = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json'
with open(json_filepath, 'r') as f:
    data = json.load(f)

##############################################################################
# Variables to change
painting_id = 38
url = data['paintings'][painting_id-1]['url']           

parent_dir = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/images'
img_filename = str(painting_id) + '.jpg'
# img_filename = 'obliquelight.png'
img_path = os.path.join(parent_dir, img_filename)
image = cv2.imread(img_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
utils_C.show_painting_summary(painting_id)

##############################################################################
# EXTRACT CONTOURS
##############################################################################
# Single contour
contour0 = utils_C.extract_contour_ui(image_rgb)

# Multiple contour
utils_C.show_painting_summary(painting_id)
contour_ids=[1,2,3,4]
contours, smooth_vals, N_vals = utils_C.get_contour_info_by_id(data, painting_id=painting_id, contour_ids=contour_ids)
# contours = contours + [contour0]

##############################################################################
# EXTRACT NORMALS
##############################################################################
smooth_vals = [5]*len(contours)  # smoothness for each contour
N_vals = [5]*len(contours)

all_spline_x: List[Any] = []
all_spline_y: List[Any] = []
all_normals: List[Any] = []
# Loop over each (contour, smoothness) pair and call get_normals_C:
for contour, smoothness in zip(contours, smooth_vals):
    spline_x, spline_y, normals = utils_C.get_normals_C(contour, smoothness)
    all_spline_x.append(spline_x)
    all_spline_y.append(spline_y)
    all_normals.append(normals)

utils_C.plot_contour_and_normals(img_path, all_spline_x,all_spline_y,all_normals,arrow_scale=50, point_size=2)


##############################################################################
# EXTRACT TRUE CONTOUR AT SUB-PIXEL LEVEL
##############################################################################
all_new_points_x: List[Any] = []
all_new_points_y: List[Any] = []

for spline_x, spline_y, normals in zip(all_spline_x, all_spline_y, all_normals):
    new_x, new_y = utils_C.get_subpixel_contour(
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

##############################################################################
# EXTRACT TRUE LUMINANCE AT SUB-PIXEL LEVEL
##############################################################################
all_I_true: List[Any] = []
for new_x, new_y, normals in zip(all_new_points_x, all_new_points_y, all_normals):
    I_true = utils_C.get_subpixel_luminance(
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


##############################################################################
# EXTRACT LIGT DIRECTION ESTIMATION AND EMPIRICAL STANDARD DEVIATION
##############################################################################
light_direction_estimation, angle_std_degs = utils_C.compute_light_direction_C(filtered_normals_list, filtered_I_true_list)

empirical_stds: List[float] = []
for normals, I_true in zip(
    filtered_normals_list,
    filtered_I_true_list
):
    # Compute empirical std for this contour
    std_deg = utils_C.compute_empirical_std(
        normals,
        I_true,
        max_iter=10,
        N_max=3
    )
    empirical_stds.append(std_deg)

empirical_stds_arr = np.array(empirical_stds, dtype=np.float32)
average_std_deg = np.nanmean(empirical_stds_arr)

##############################################################################
# VISUALIZE SAMPLED VS THEORETICAL LUMINANCE DISTRIBUTION
##############################################################################
stacked_normals = np.vstack(filtered_normals_list)
stacked_I_true = np.concatenate(filtered_I_true_list)
# -----------------------------------------------------------------------------#
####
# gt_light_direction_estimation = np.array([-0.89, -0.45])
# ground_truth_light = gt_light_direction_estimation @ normals.T
theoretical_light = np.clip(stacked_normals @ light_direction_estimation, 0, 1)

I_true_norm = utils_C.normalize_to_01(stacked_I_true)
I_theo_norm = utils_C.normalize_to_01(theoretical_light)
# I_gt_norm   = utils.normalize_to_01(ground_truth_light)

# plot luminance distributions against angle
kx, ky = stacked_normals[:, 0], stacked_normals[:, 1]
angles_rad = np.mod(np.arctan2(-ky, kx), 2*np.pi)          
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

##############################################################################
# EXTRACT REST OF THE METRICS
##############################################################################
# Compute shading delta metric
shading_metrics: List[Tuple[float, float]] = []  # each entry: (mse, delta)
for new_x, new_y, normals, I_true in zip(
    filtered_new_points_x_list,
    filtered_new_points_y_list,
    filtered_normals_list,
    filtered_I_true_list
):
    result = utils_C.compute_shading_delta_metric(
        I_true,
        light_direction_estimation,
        normals,
        sigma=5,
        max_sigma=100,
        sigma_step=5)
    if result is not None:
        mse, delta, *_ = result
        shading_metrics.append((mse, delta))

shading_metrics_arr = np.array(shading_metrics, dtype=np.float32)  # shape (num_contours, 2)
avg_mse = float(np.nanmean(shading_metrics_arr[:, 0]))
deltas = shading_metrics_arr[:, 1]
abs_mean = float(np.nanmean(np.abs(deltas)))
sum_delta = np.nansum(deltas)
overall_sign = np.sign(sum_delta) if sum_delta != 0 else 0
final_delta = abs_mean * overall_sign

# Compute shading noise metric
noise_list: List[int] = []
for I_true_i in filtered_I_true_list:
    n_i = utils_C.compute_noise_metric(I_true_i)
    noise_list.append(n_i)

noise_arr = np.array(noise_list, dtype=np.float32)
lengths = np.array([len(I) for I in filtered_I_true_list], dtype=np.float32)
weighted_avg_noise = float(np.dot(noise_arr, lengths) / np.sum(lengths))

# compute ranges and SH
norm_smoothed_luminance_distribution = stacked_I_true / stacked_I_true.max()
brightness_range = norm_smoothed_luminance_distribution.max() - norm_smoothed_luminance_distribution.min()
normal_vectors_span = utils_C.compute_angle_range(stacked_normals)
coefficients_reg = utils_C.compute_sh_coefficients(stacked_normals, norm_smoothed_luminance_distribution)

##############################################################################
# SAVE THE CONTOUR DATA TO JSON
##############################################################################
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
utils_C.add_contour(painting_id, contour_data)

utils_C.show_painting_summary(painting_id)

# UPDATE CONTOUR DATA BY ID
utils_C.replace_contour(painting_id=painting_id, contour_id=6, new_contour_data = contour_data)

# # DELETE CONTOUR DATA BY ID
utils_C.delete_contour(painting_id=painting_id, contour_id=3)

##############################################################################
# EXTRACT LIGHT DIRECTION DEPICTED EVIDENCE: CAST SHADOWS OR SPECULAR HIGHLIGHTS
##############################################################################
# (OPTIONAL) 'depicted_light_direction_evidence' Cast shadows or specular highlights
cast_shadow = utils_C.extract_light_direction_ui(img_path)
depicted_light_direction_evidence = []
# For specular highlights use the python script in specular_highlights.py
depicted_light_direction_evidence = [
    {
        "type": "cast shadow",
        "light_direction": cast_shadow,
        "notes": "cast shadow on the rear wall"
    }
]

##############################################################################
# EXTRACT GLOBAL LIGHT DIRECTION ESTIMATION AND STANDARD DEVIATION
##############################################################################
# GLOBAL CONSISTENCY METRIC (after all contours have been extracted)

global_light_direction_estimation, global_estimation_std_degrees = utils_C.compute_global_light_from_estimates(painting_id, json_filepath)
# Plotting the result. Arrow in the middle of the image shows the estimated global direction and the wedge the std of all estimations. 
height, width = gray_image.shape
x_coordinates = np.array([width / 2], dtype=np.float32)
y_coordinates = np.array([height / 2], dtype=np.float32)
utils_C.vis_global_light_estimation_C(img_path, global_light_direction_estimation, x_coordinates, y_coordinates, global_estimation_std_degrees, scale=1000, visual_factor = 1.0)

##############################################################################
# SAVE THE REST OF THE DATA IN THE JSON
##############################################################################
# number_of_people = 4
# utils_C.add_remaining_info(painting_id, depicted_light_direction_evidence, number_of_people, global_light_direction_estimation, global_estimation_std_degrees)
utils_C.add_remaining_global_info(painting_id, global_light_direction_estimation, global_estimation_std_degrees)

##############################################################################
# VISUALIZE THE CONTOUR LIGHT ESTIMATIONS
##############################################################################
utils_C.vis_contour_light_estimations(data, painting_id, img_path, scale=400, save_fig=False)
