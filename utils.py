'''
utils.py to store functions
'''

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator
from scipy.optimize import minimize
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
import json
from matplotlib.patches import Wedge
import tempfile


def load_image_and_masks(parent_dir, img_filename, mask_filenames):
    img_path = os.path.join(parent_dir, img_filename)
    org_img = cv2.imread(img_path)
    gray_image = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    mask_paths = [os.path.join(parent_dir, mask_filename) for mask_filename in mask_filenames]
    return img_path, gray_image, mask_paths

def detect_occluding_contours(image):
    '''
    Simple function to detect occluding contours from an image using Canny edge detection and contour finding in cv2
    '''
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian blur to smoothen the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Canny edge detection: A high upper threshold may lead to missed edges, while a low threshold may result in a lot of noise.
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)
    # Find contours from the detected edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, edges

def get_subpixel_luminance(gray_image, x, y, normals, N):
    '''
    Given the image, the contour points, and the normals, this function calculates the ground truth luminance at subpixel level.
    N indicates the number of points to sample on the inwards direction of the normal vector.
    For each contour point, the function samples intensities along the normal direction and fits a continuous function to estimate the luminance break as the maximum gradient.
    The function uses a buffer zone to avoid edge effects in the gradient calculation.
    Finally, the function calculates the new subpixel coordinates by applying the displacement along the normal direction. This displacement is the mean of luminance break distances along the contour.
    Returns the new contour points and the sampled ground truth luminance.
    '''
    interp_function = RegularGridInterpolator((np.arange(gray_image.shape[0]), np.arange(gray_image.shape[1])), gray_image) # interpolator to sample intensities in sub-pixel level
    n_points = len(x)
    luminance_break_distances = np.zeros(n_points)
    new_points_x = np.zeros(n_points)
    new_points_y = np.zeros(n_points)
    new_ground_truth_luminance = np.zeros(n_points)
    
    for i in range(n_points):
        start_x, start_y = x[i], y[i]
        normal = normals[i]
        sampled_intensities = []
        sampled_positions = []
        for step in range(N+1):  # Only look in the opposite direction of the normal to sample points
            sample_x = start_x - normal[0] * step
            sample_y = start_y - normal[1] * step
            # Use interpolation to sample at subpixel locations
            sampled_intensities.append(interp_function([sample_y, sample_x])[0])  # get the interpolated value, following image (y,x) convention
            sampled_positions.append(step)
        
        sampled_positions = np.array(sampled_positions, dtype=float)
        sampled_intensities = np.array(sampled_intensities)
        
        # Use linear interpolation if insufficient points for cubic spline
        if len(sampled_positions) < 4:
            # Fit a simple linear polynomial
            poly_coeffs = np.polyfit(sampled_positions, sampled_intensities, 1)
            fit_positions = np.linspace(sampled_positions.min(), sampled_positions.max(), 1000)
            gradients = np.polyval([poly_coeffs[0]], fit_positions)  # derivative of linear fit is constant
        else:
            # Fit a cubic spline
            spline_fit = UnivariateSpline(sampled_positions, sampled_intensities, s=0, k=3)
            # Analyze the derivative
            first_derivative = spline_fit.derivative(n=1)
            fit_positions = np.linspace(sampled_positions.min(), sampled_positions.max(), 1000)
            gradients = first_derivative(fit_positions)
        
        # Find luminance break (maximum gradient) as a simple proxy for the "break"
        buffer_fraction = 0.1
        buffer_size = int(buffer_fraction * len(fit_positions))
        valid_gradients = gradients[buffer_size:-buffer_size]
        valid_positions = fit_positions[buffer_size:-buffer_size]
        max_grad_index = np.argmax(np.abs(valid_gradients))
        luminance_break_position = valid_positions[max_grad_index]
        
        new_ground_truth_luminance[i] = np.interp(luminance_break_position, sampled_positions, sampled_intensities)
        luminance_break_distances[i] = luminance_break_position
    
    # Calculate mean luminance break distance
    mean_luminance_break_distance = np.mean(luminance_break_distances)
    new_points_x = x - normals[:, 0] * mean_luminance_break_distance
    new_points_y = y - normals[:, 1] * mean_luminance_break_distance
    new_ground_truth_luminance = interp_function(np.vstack((new_points_y, new_points_x)).T)  # sample sub-pixel luminance values at the adjusted points
    return new_points_x, new_points_y, new_ground_truth_luminance

def plot_luminance_break_vis(gray_image, x, y, normals, N, point_idx):
    # Create the interpolator
    interp_function = RegularGridInterpolator(
        (np.arange(gray_image.shape[0]), np.arange(gray_image.shape[1])), gray_image
    )
    # Get the selected contour point and its normal
    start_x, start_y = x[point_idx], y[point_idx]
    normal = normals[point_idx]
    # Define steps: two pixels outside (negative) and N+1 pixels inside (0 to N)
    steps = list(range(-2, 0)) + list(range(0, N+1))
    sampled_intensities = []
    sampled_positions = []
    sampled_x = []
    sampled_y = []
    for step in steps:
        sample_x = start_x - normal[0]*step
        sample_y = start_y - normal[1]*step
        sampled_x.append(sample_x)
        sampled_y.append(sample_y)
        intensity = interp_function([sample_y, sample_x])[0]
        sampled_intensities.append(intensity)
        sampled_positions.append(step)
    sampled_positions = np.array(sampled_positions, dtype=float)
    sampled_intensities = np.array(sampled_intensities)
    # Fit a continuous function to the intensity profile
    if len(sampled_positions) < 4:
        poly_coeffs = np.polyfit(sampled_positions, sampled_intensities, 1)
        fit_positions = np.linspace(sampled_positions.min(), sampled_positions.max(), 1000)
        gradients = np.full_like(fit_positions, poly_coeffs[0])
    else:
        spline_fit = UnivariateSpline(sampled_positions, sampled_intensities, s=0, k=3)
        first_derivative = spline_fit.derivative(n=1)
        fit_positions = np.linspace(sampled_positions.min(), sampled_positions.max(), 1000)
        gradients = first_derivative(fit_positions)
    # Apply buffer to avoid edge effects
    buffer_fraction = 0.01
    buffer_size = int(buffer_fraction * len(fit_positions))
    valid_gradients = gradients[buffer_size:-buffer_size]
    valid_positions = fit_positions[buffer_size:-buffer_size]
    max_grad_index = np.argmax(np.abs(valid_gradients))
    luminance_break_position = valid_positions[max_grad_index]
    # Determine discrete step (next sampled point) as per the new strategy
    new_step = int(np.ceil(luminance_break_position))
    new_step = np.clip(new_step, 1, N)
    # Compute the new contour point based on the discrete step
    new_point_x = start_x - normal[0]*new_step
    new_point_y = start_y - normal[1]*new_step
    new_intensity = interp_function([new_point_y, new_point_x])[0]
    # Plotting
    plt.figure(figsize=(12,5))
    # Left: Spatial view on image
    plt.subplot(1,2,1)
    plt.imshow(gray_image, cmap='gray')
    # Plot all sampled points
    plt.scatter(sampled_x, sampled_y, c='b', s=15, label='Sampled Points')
    # Highlight the original contour point
    plt.scatter(start_x, start_y, c='orange', s=30, label='Original Contour Point')
    # Highlight the new contour point (discrete chosen one)
    plt.scatter(new_point_x, new_point_y, c='g', s=30, label='New Contour Point')
    plt.legend()
    plt.title('Spatial: Sampled Points, Computed Break, and New Contour')
    # Right: Intensity profile along normal
    plt.subplot(1,2,2)
    plt.plot(sampled_positions, sampled_intensities, 'bo-', label='Sampled Intensities')
    plt.axvline(luminance_break_position, color='r', linestyle='--', label='Computed Luminance Break (subpixel)')
    # Mark the discrete step chosen (new contour point)
    plt.axvline(new_step, color='g', linestyle='--', label='Discrete New Contour Step')
    # Mark original contour intensity at step=0
    plt.scatter(0, np.interp(0, sampled_positions, sampled_intensities), c='orange', s=40, label='Original Contour Intensity')
    plt.xlabel('Position along normal (step)')
    plt.ylabel('Intensity')
    plt.title('Intensity Profile with Luminance Break and Discrete Selection')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def select_occluding_contour(img, mask, contour):
    '''
    Visualize the occluding contours on the image where contour is a single contour from cv2.findContours.
    Allows the user to select 2 points on the contour and returns the segment of the contour between these points.
    '''
    # Helper function to find the closest point on the contour to a given point
    def closest_contour_point(contour, point):
        min_dist = float('inf')
        closest_point = None
        closest_idx = -1
        for idx, c_point in enumerate(contour):
            dist = np.linalg.norm(c_point[0] - point)
            if dist < min_dist:
                min_dist = dist
                closest_point = c_point
                closest_idx = idx
        return closest_point, closest_idx
    # Display the image with contour
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    drawing = cv2.drawContours(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), [contour], -1, (255, 255, 0), 2)
    ax.imshow(drawing)
    selected_points = []
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            point = np.array([event.xdata, event.ydata], dtype=np.float32)
            closest_point, closest_idx = closest_contour_point(contour, point)
            selected_points.append((closest_point, closest_idx))
            ax.plot(closest_point[0][0], closest_point[0][1], 'ro')
            fig.canvas.draw()
            if len(selected_points) == 2:
                fig.canvas.mpl_disconnect(cid)
                contour_segment = extract_contour_segment()
                # Display the segment
                fig_segment, ax_segment = plt.subplots()
                drawing_segment = cv2.drawContours(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), [contour_segment], -1, (255, 0, 0), 2)
                ax_segment.imshow(drawing_segment)
                plt.show()
                return contour_segment
    def extract_contour_segment():
        # Ensure points are in order
        selected_points.sort(key=lambda x: x[0][0][0])
        idx1 = selected_points[0][1]
        idx2 = selected_points[1][1]
        if idx1 < idx2:
            segment1 = contour[idx1:idx2+1]
            segment2 = np.vstack((contour[idx2:], contour[:idx1+1]))
        else:
            segment1 = contour[idx2:idx1+1]
            segment2 = np.vstack((contour[idx1:], contour[:idx2+1]))
        if len(segment1) <= len(segment2):
            return segment1
        else:
            return segment2
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    # Ensure to return the contour segment after the plot closes
    if len(selected_points) == 2:
        return extract_contour_segment()
    else:
        return None
    
def compute_light_directions(img_path, new_points_x, new_points_y, normals, ground_truth_subpixel, n_patches, lambda_reg):
    org_img = cv2.imread(img_path)
    # normalize input variables
    height, width = org_img.shape[0], org_img.shape[1]
    img_norm = max(height, width)
    # normalize the contour points but maintain the aspect ratio!
    spline_point_x_norm = new_points_x / img_norm
    spline_point_y_norm = new_points_y / img_norm
    n_total_points = len(spline_point_x_norm)
    points_per_patch = n_total_points // n_patches
    # Get the centers for each patch
    xy = np.stack((spline_point_x_norm, spline_point_y_norm), axis=-1)
    xy_patches = [xy[i * points_per_patch:(i + 1) * points_per_patch] for i in range(n_patches)]
    centers = np.array([xy_patches[i][points_per_patch // 2] for i in range(n_patches)])
    # Construct matrix M and vector b
    M = np.zeros((n_patches * points_per_patch, n_patches * 2 + 1))
    b = np.zeros(n_patches * points_per_patch)
    for i, patch in enumerate(xy_patches):
        # Start and end index for the current patch
        start_idx = i * points_per_patch
        end_idx = (i + 1) * points_per_patch
        # Assign normals to M
        M[start_idx:end_idx, i * 2:(i + 1) * 2] = normals[start_idx:end_idx]
        # Assign luminance values to b
        b[start_idx:end_idx] = ground_truth_subpixel[start_idx:end_idx]
    M[:, -1] = 1
    # Get initial estimation
    v0, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
    def cost_function(v):
        L = v[:-1].reshape((n_patches, 2))
        E1 = np.linalg.norm(M @ v - b) ** 2
        # E2 term
        E2 = 0
        for i in range(n_patches):
            delta_i = (L[i] - centers[i]) / np.linalg.norm(L[i] - centers[i])
            C_i = np.eye(2) - np.outer(delta_i, delta_i)
            E2 += np.linalg.norm(C_i @ L[i]) ** 2
        return E1 + lambda_reg * E2
    # The light direction estimation needs to be in the range of the normals
    def constraint(v):
        L = v[:-1].reshape((n_patches, 2))
        constraints = []
        for i in range(n_patches):
            dot_product = np.dot(normals[i * points_per_patch:(i + 1) * points_per_patch], L[i])
            constraints.extend(dot_product)
        return np.array(constraints)
    cons = {'type': 'ineq', 'fun': constraint}
    result = minimize(cost_function, v0, method='SLSQP', constraints=cons)
    # Optimized light directions and ambient term
    v_optimized = result.x
    light_directions = v_optimized[:-1].reshape(-1, 2)
    norms = np.linalg.norm(light_directions, axis=1, keepdims=True)
    normalized_directions = light_directions / norms
    overall_direction = np.mean(light_directions, axis=0)
    overall_direction /= np.linalg.norm(overall_direction)
    mid_contour = np.mean(xy, axis=0)
    return normalized_directions, overall_direction, mid_contour, centers

def process_image(img_path, mask_path, contour_index = 0):
    # Read the original image and convert to grayscale
    org_img = cv2.imread(img_path)
    gray_image = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    # Read and process the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred_image, threshold1=20, threshold2=200)
    # Find contours and select the largest one
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # occluding_contour = max(contours, key=cv2.contourArea)
    contour_segment = select_occluding_contour(org_img, mask, contours[contour_index])
    return contour_segment

def process_masks(img_path, mask_paths):
    contours = []
    for mask_path in mask_paths:
        contour = process_image(img_path, mask_path, contour_index=0)
        contours.append(contour)
    return contours

def calculate_intersection_vectors(directions, centers):
    '''
    This is to calculate the light source location as the intersectino of light direction estimations from contours. 
    Minimize the sum of squared deviations from each light direction's ideal intersection.
    The least squares solution finds the point closest to the intersection of these lines.
    This point might not lie exactly on any of the lines, but it minimizes the overall distance to all of them.
    '''
    if len(directions) < 2:
        raise ValueError("We need at least 2 directions to calculate the intersection")
    # Construct matrix A and vector b
    A = np.array([[direction[1], -direction[0]] for direction in directions])
    b = np.array([
        center[0] * direction[1] - center[1] * direction[0]
        for center, direction in zip(centers, directions)
    ])
    # Solve for the intersection point using least-squares
    light_source_position = np.linalg.lstsq(A, b, rcond=None)
    return light_source_position[0]

def plot_contour_and_normals(img_path, new_points_x, new_points_y, normals):
    org_img = cv2.imread(img_path)
    gray_image = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))
    ax.scatter(new_points_x, new_points_y, color='red', label='Subpixel Groundtruth', s=2)
    num_normals_to_plot = int(0.1 * len(normals))
    for i in np.linspace(0, len(normals) - 1, num=num_normals_to_plot).astype(int):
        ax.arrow(new_points_x[i], new_points_y[i], normals[i, 0] * 50, normals[i, 1] * 50, color='skyblue', head_width=1)
    ax.axis('off')
    ax.legend()
    plt.show()

def get_Lightdirection_pdf(light_direction_estimations, ground_truths, normals_list):
    '''
    This is MSE-based variance estimation. 
    The variance of the gaussian is the variance of the MSE between the measured luminance and the calculated luminance for a 180 range of light directions
    This is NOT measuring the uncertainty of the estimated light direction.
    Rather, this is a weighted representation of the angles that can explain the measured luminance at the contour. This measures how unique is a light direction for a given contour. 
    '''
    probabilities_list = []
    angles_list = []
    variances_list = [] # for later use in Priors
    for i in range(len(light_direction_estimations)):
        direction = light_direction_estimations[i]
        ground_truth = ground_truths[i]
        normals = normals_list[i]
        # we want +- 90 degrees from the estimated direction
        direction_angle = np.degrees(np.arctan2(direction[1], direction[0]))
        angles = np.linspace(direction_angle - 90, direction_angle + 90, 180)
        radians = np.radians(angles)
        # Calculate luminance for each angle and compute MSE
        expected_luminance = np.zeros((len(radians), len(normals)))
        mse_values = np.zeros(len(radians))
        for j, theta in enumerate(radians):
            light_dir = np.array([np.cos(theta), np.sin(theta)])
            light_dir = light_dir / np.linalg.norm(light_dir)  # Ensure the light direction vector is normalized
            # Calculate luminance at this light direction
            expected_luminance[j] = [max(0, np.dot(normals[k], light_dir)) for k in range(len(normals))]
            # Calculate MSE between measured luminance and calculated
            mse_values[j] = np.mean((ground_truth - expected_luminance[j])**2)
        # Gaussian distribution
        mean_mse = np.mean(mse_values)
        variance_mse = np.var(mse_values + 1e-6)
        variances_list.append(variance_mse)
        sigma = np.sqrt(variance_mse) # this is the standard deviation across angles in pixel intensity error space, MSE space. 
        likelihoods = np.exp(-mse_values / (2 * sigma ** 2))
        total_likelihood = np.sum(likelihoods)
        probabilities = likelihoods / total_likelihood
        # Compute shift
        max_prob_index = np.argmax(probabilities)
        angle_max_prob = angles[max_prob_index]
        shift = direction_angle - angle_max_prob
        shifted_angles = angles + shift
        shifted_angles = (shifted_angles + 180) % 360 - 180 # avoid unexpected angle range
        probabilities_list.append(probabilities)
        angles_list.append(shifted_angles)
        # Calcutate circular std in degrees
        max_prob_index = np.argmax(probabilities)
        direction_angle = angles[max_prob_index] 
        angles_rad = np.radians(angles)
        R_x = np.sum(probabilities * np.cos(angles_rad))
        R_y = np.sum(probabilities * np.sin(angles_rad))
        R = np.sqrt(R_x**2 + R_y**2)
        eps = 1e-6
        if R < eps:
            # nearly uniform distribution, huge spread
            circular_std_deg = 180.0
        elif R > (1 - eps):
            # extremely peaked distribution
            circular_std_deg = 0.0
        else:
            circular_std_rad = np.sqrt(-2.0 * np.log(R + eps))
            circular_std_deg = np.degrees(circular_std_rad) # this measures the error in angle space
    return probabilities_list, angles_list, variances_list, round(sigma, 4), round(circular_std_deg, 4)

def MLE_map(gray_image, contours, probabilities_list, angles_list, x_coordinates, y_coordinates, resize_factor = 0.1, use_priors=False, combined_weights=None):
    '''
    The resize_factor is used to reduce computation. We process the MLE map per pixel in a lower resolution and then resize to the original image size.
    We just use the gray_image for dimensions.
    '''
    new_height = int(gray_image.shape[0] * resize_factor)
    new_width = int(gray_image.shape[1] * resize_factor)
    resized_gray_image = cv2.resize(gray_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_likelihood_map = np.ones((new_height, new_width))
    # Scaling factors to map back to the original image
    scale_x = gray_image.shape[1] / new_width
    scale_y = gray_image.shape[0] / new_height
    for i, contour in enumerate(contours):
        probabilities = probabilities_list[i]
        direction_angles = angles_list[i]
        # Midpoint of the contour
        x_mid = np.mean(x_coordinates[i])
        y_mid = np.mean(y_coordinates[i])
        # Convert midpoint to resized coordinated system
        resized_x_mid = x_mid / scale_x
        resized_y_mid = y_mid / scale_y
        if use_priors:
            weight = combined_weights[i]
        for j in range(new_height):
            for k in range(new_width):
                # Calculate the direction from the pixel to the midpoint of the contour
                pixel_direction = np.array([k - resized_x_mid, j - resized_y_mid])
                if np.linalg.norm(pixel_direction) == 0:
                    continue
                pixel_direction = pixel_direction / np.linalg.norm(pixel_direction)  
                pixel_angle = np.degrees(np.arctan2(pixel_direction[1], pixel_direction[0]))
                # Adjusted angle difference to handle circular nature
                angle_diff = np.abs(direction_angles - pixel_angle)
                angle_diff = np.minimum(angle_diff, 360 - angle_diff)  # Ensure smallest angle difference is considered
                # Extract probability from the contour probability distribution
                min_diff_index = np.argmin(angle_diff)
                extracted_prob = probabilities[min_diff_index]
                if use_priors:
                    weighted_prob = extracted_prob * weight
                else:
                    weighted_prob = extracted_prob
                # Update the likelihood map with the weighted probability
                resized_likelihood_map[j, k] *= weighted_prob
    # Normalize probability map for visuallization
    resized_probability_map  = resized_likelihood_map / np.max(resized_likelihood_map)
    # resize the probability map back to original resolution
    probability_map = cv2.resize(resized_probability_map, (gray_image.shape[1], gray_image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return probability_map

def vis_MLE_map(img_path, probability_map, directions, x_coordinates, y_coordinates, probabilities_list, angles_list, L_x, L_y, scale = 250, plot_MLE = True, plot_contours_flag = False, plot_only_contours = False):
    fig, ax = plt.subplots()
    # Set the background color to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    org_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Adjust the axes to display the entire plot with a black background and set image opacity
    ax.imshow(org_img, cmap='gray', alpha=0.6, extent=[0, org_img.shape[1], org_img.shape[0], 0])
    if plot_only_contours:
        # Plot only the contours if plot_only_contours is True
        for i in range(len(x_coordinates)):
            ax.plot(x_coordinates[i], y_coordinates[i], '.', markersize=4, color='blue')
    else:
        if plot_MLE:
            heatmap = ax.imshow(probability_map, cmap='inferno', alpha=0.7, extent=[0, org_img.shape[1], org_img.shape[0], 0])
            plt.scatter(L_x, L_y, color='red', s=100, marker='x')
        if plot_contours_flag:
            for i in range(len(x_coordinates)):
                ax.plot(x_coordinates[i], y_coordinates[i], '.', markersize=2, color='red')
        for i in range(len(directions)):
            center_x = np.mean(x_coordinates[i])
            center_y = np.mean(y_coordinates[i])
            scaling_factor = scale / max(probabilities_list[i])
            # Collect boundary points for creating the filled area
            boundary_points = []
            for j, theta in enumerate(angles_list[i]):
                light_dir = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
                length = probabilities_list[i][j] * scaling_factor  # Scaling factor for visualization
                boundary_points.append([center_x + length * light_dir[0], center_y + length * light_dir[1]])
            # Create the filled area with a gradient in opacity from 0.1 to 0.5
            if len(boundary_points) > 1:
                boundary_points = np.array(boundary_points)
                for k in range(len(boundary_points) - 1):
                    x = [center_x, boundary_points[k][0], boundary_points[k + 1][0]]
                    y = [center_y, boundary_points[k][1], boundary_points[k + 1][1]]
                    alpha = 0.1 + (0.4 * (k / (len(boundary_points) - 1)))  # Gradient from 0.1 to 0.5
                    ax.fill(x, y, color='yellow', alpha=alpha)
            # Plot main light direction with increased arrow width
            ax.arrow(center_x, center_y, directions[i][0] * scale, directions[i][1] * scale, head_width=10, head_length=10, width=4, fc='green', ec='green', label='Max prob direction')
    # Set limits to ensure filled area and arrows are visible with black background and expand limits
    ax.set_xlim([-0.1 * org_img.shape[1], 1.1 * org_img.shape[1]])
    ax.set_ylim([1.1 * org_img.shape[0], -0.1 * org_img.shape[0]])  # Expand y-axis to accommodate arrows and reverse y-axis
    # Remove axis for a cleaner look with black background
    ax.axis('off')
    plt.show()

def get_lightsource_location(probability_map):
    '''
    We calculate the light source location as the maximum probability pixel in the MLE map.
    '''
    max_prob_idx = np.unravel_index(np.argmax(probability_map), probability_map.shape)
    L_y, L_x = max_prob_idx 
    return np.array([L_x, L_y])

def get_distance_priors(Ls_location, contours, x_coordinates, y_coordinates, low_boundary):
    '''
    Function to get the Priors (confidences) for each contour based on distance.
    A closer contour is considered to have greater lighting information, hence a larger prior.
    Since closer distances have lower values, we invert the calculated distances to get the final normalized priors.
    As we do not want to discard any contour, the normalization is between a low boundary and 1, instead of [0,1]
    '''
    L_x, L_y = Ls_location
    distances = []
    for i, contour in enumerate(contours):
        # Midpoint of the contour
        x_mid = np.mean(x_coordinates[i])
        y_mid = np.mean(y_coordinates[i])
        distance = np.array([L_x - x_mid, L_y - y_mid])
        distance = np.linalg.norm(distance)
        distances.append(distance)
    distances = np.array(distances)
    inverted_distances = 1.0 / (distances + 1e-16)**2  # we invert the distances to get the closest distance to 1, add inverse squared relationship between light and distance
    min_distance = np.min(inverted_distances)
    max_distance = np.max(inverted_distances)
    normalized_distances = ((1-low_boundary)*(inverted_distances - min_distance) / (max_distance - min_distance)) + low_boundary
    return normalized_distances

def get_variance_priors(variances_list, low_boundary):
    '''
    A contour with lower variance is considered to be more certain and hence a greater prior is assigned to it. 
    As lower variances imply greater priors we invert the values. 
    As we do not want to discard any contour, the normalization is between a low boundary and 1, instead of [0,1]
    '''
    variances = np.array(variances_list)
    inverted_variances = 1.0 / (variances + 1e-16) 
    min_var = np.min(inverted_variances)
    max_var = np.max(inverted_variances)
    normalized_variances = ((1-low_boundary)*(inverted_variances - min_var) / (max_var - min_var)) + low_boundary
    return normalized_variances


def get_priors(contours, Ls_location, x_coordinates, y_coordinates, variances_list, weight_distance = 0.2, weight_variance = 0.8, low_boundary = 0.15):
    '''
    The priors are a combination of varance and distance priors. 
    We use weights to represent the importance of each type of prior.
    As we do not want to discard any contour, the normalization of the priors is between a low boundary and 1, instead of [0,1]
    '''    
    normalized_distances = get_distance_priors(Ls_location, contours, x_coordinates, y_coordinates, low_boundary)
    normalized_variances = get_variance_priors(variances_list, low_boundary)
    priors = weight_distance * normalized_distances + weight_variance * normalized_variances
    return priors


def get_relative_area(gray_image, probability_map, lower_bound = 0.2, upper_bound = 1.0, plot = True):
    '''
    This function computes the relative area of the MLE map as the number of pixels between an specified probability range and the total number of pixels.
    relative area is given in percentage.
    '''
    # Create a binary mask for the pixels within the specified probability range
    mask = (probability_map >= lower_bound) & (probability_map <= upper_bound)
    # Calculate the area within the range
    area_in_range = np.sum(mask)
    # Calculate the total number of pixels
    total_pixels = probability_map.size
    # Calculate the relative area in %
    relative_area = (area_in_range / total_pixels) * 100
    if plot:
        colored_image = np.dstack([gray_image] * 3)  # Stack the grayscale image into 3 channels (R, G, B)
        yellow = np.array([255, 255, 0])  # RGB values for yellow
        alpha = 0.75  # Opacity level (0 = fully transparent, 1 = fully opaque)
        colored_image[mask] = (1 - alpha) * colored_image[mask] + alpha * yellow
        plt.figure(figsize=(10, 10))
        plt.imshow(colored_image, cmap='gray')
        plt.axis('off')  # Turn off axis labels for a cleaner look
        # Show the plot
        plt.show()
    
    return relative_area

def calculate_angular_coverage(x_coords_list, y_coords_list, L_x, L_y, gap_threshold_degrees=5.0):
    """
    Calculates the angular coverage given the contour points and light source position.

    Parameters:
    - x_coords_list: List of numpy arrays of x coordinates for each contour.
    - y_coords_list: List of numpy arrays of y coordinates for each contour.
    - L_x: x coordinate of the light source.
    - L_y: y coordinate of the light source.
    - gap_threshold_degrees: Threshold in degrees to consider gaps as significant.

    Returns:
    - angular_coverage_degrees: Total angular coverage in degrees.
    """
    # Collect all angles
    theta_list = []
    # For each contour
    for x_coords, y_coords in zip(x_coords_list, y_coords_list):
        # For each point in the contour
        for x_i, y_i in zip(x_coords, y_coords):
            dx = L_x - x_i
            dy = L_y - y_i
            theta = np.arctan2(dy, dx)
            # Normalize angle to [0, 2π)
            theta = theta % (2 * np.pi)
            theta_list.append(theta)
    # Convert list to numpy array
    theta_array = np.array(theta_list)
    # Sort the angles
    sorted_theta = np.sort(theta_array)
    # Compute the gaps between consecutive angles
    gaps = np.diff(sorted_theta)
    # Include the wrap-around gap
    wrap_gap = (sorted_theta[0] + 2 * np.pi) - sorted_theta[-1]
    gaps = np.append(gaps, wrap_gap)
    # Convert gap threshold from degrees to radians
    gap_threshold = np.radians(gap_threshold_degrees)
    # Identify significant gaps (gaps larger than the threshold)
    significant_gaps = gaps[gaps > gap_threshold]
    # Sum up significant gaps to get total uncovered angle
    total_uncovered_angle = np.sum(significant_gaps)
    # Compute total angular coverage
    angular_coverage = (2 * np.pi) - total_uncovered_angle
    # Convert angular coverage to degrees
    angular_coverage_degrees = np.degrees(angular_coverage)
    return angular_coverage_degrees

def plot_angular_coverage(x_coords_list, y_coords_list, L_x, L_y, gap_threshold_degrees=5.0):
    theta_list = []
    # Collect all angles
    for x_coords, y_coords in zip(x_coords_list, y_coords_list):
        for x_i, y_i in zip(x_coords, y_coords):
            dx = L_x - x_i
            dy = L_y - y_i
            theta = np.arctan2(dy, dx)
            theta = theta % (2 * np.pi)
            theta_list.append(theta)
    theta_array = np.array(theta_list)
    sorted_theta = np.sort(theta_array)
    # Compute gaps
    gaps = np.diff(sorted_theta)
    wrap_gap = (sorted_theta[0] + 2 * np.pi) - sorted_theta[-1]
    gaps = np.append(gaps, wrap_gap)
    # Convert gap threshold to radians
    gap_threshold = np.radians(gap_threshold_degrees)
    # Identify significant gaps
    significant_gaps = gaps[gaps > gap_threshold]
    significant_gap_indices = np.where(gaps > gap_threshold)[0]
    # Plot the unit circle
    plt.figure(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, color='lightgray', fill=False)
    plt.gca().add_patch(circle)
    # Plot angles on the circle
    x_points = np.cos(theta_array)
    y_points = np.sin(theta_array)
    plt.plot(x_points, y_points, 'b.', label='Angles')
    # Plot significant gaps
    for idx in significant_gap_indices:
        theta_start = sorted_theta[idx]
        theta_end = sorted_theta[(idx + 1) % len(sorted_theta)]
        # Handle wrap-around
        if theta_end < theta_start:
            theta_end += 2 * np.pi
        theta_gap = np.linspace(theta_start, theta_end, 100)
        x_gap = np.cos(theta_gap)
        y_gap = np.sin(theta_gap)
        plt.plot(x_gap, y_gap, 'r-', linewidth=2)
    plt.title('Angles on Unit Circle with Significant Gaps')
    plt.xlabel('Cos(Theta)')
    plt.ylabel('Sin(Theta)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

def scale_coordinates(gray_image, x_coordinates, y_coordinates, L_x, L_y):
    '''
    this function scales the coordinates to match the synthetic image space of (640x360).
    With this, we ensure that regardless of the input image size and aspet ratio, the coordinates are mapped to the training image space. 
    '''
    scale = min(640 / gray_image.shape[1] , 360 / gray_image.shape[0])
    L_x_scaled = L_x * scale
    L_y_scaled = L_y * scale
    x_coords_scaled = [np.array(x) * scale for x in x_coordinates]
    y_coords_scaled = [np.array(y) * scale for y in y_coordinates]
    # calculate required padding to mimic the image training dimensions
    pad_x = (640 - (gray_image.shape[1] * scale)) / 2
    pad_y = (360 - (gray_image.shape[0] * scale)) / 2
    x_coords_final = [x + pad_x for x in x_coords_scaled]
    y_coords_final = [y + pad_y for y in y_coords_scaled]
    L_x_final = L_x_scaled + pad_x
    L_y_final = L_y_scaled + pad_y
    return x_coords_final, y_coords_final, L_x_final, L_y_final

def calculate_angular_coverage_vector(gray_image, x_coords_list, y_coords_list, L_x, L_y, resolution_degrees=1.0):
    '''
    Function to calculate the feature vector for angular coverage.
    '''
    # first scale coordinates to match the synthetic image space
    x_coords_list, y_coords_list, L_x, L_y = scale_coordinates(gray_image, x_coords_list, y_coords_list, L_x, L_y)
    num_bins = int(360 / resolution_degrees)
    angular_coverage_vector = np.zeros(num_bins)
    for x_coords, y_coords in zip(x_coords_list, y_coords_list):
        x_coords = np.asarray(x_coords)
        y_coords = np.asarray(y_coords)
        # Compute differences and angles for all points
        dx_points = L_x - x_coords
        dy_points = L_y - y_coords
        distances_points = np.hypot(dx_points, dy_points)
        valid_indices = distances_points > 0  # Avoid division by zero
        dx_points = dx_points[valid_indices]
        dy_points = dy_points[valid_indices]
        distances_points = distances_points[valid_indices]
        x_points = x_coords[valid_indices]
        y_points = y_coords[valid_indices]
        thetas = np.arctan2(dy_points, dx_points) % (2 * np.pi)
        thetas_deg = np.degrees(thetas)
        bin_indices = (thetas_deg / resolution_degrees).astype(int) % num_bins
        # Group points by angle bin
        unique_bins, inverse_indices = np.unique(bin_indices, return_inverse=True)
        # Initialize arrays to store sums and counts
        sum_x = np.zeros(len(unique_bins))
        sum_y = np.zeros(len(unique_bins))
        count = np.zeros(len(unique_bins))
        # Accumulate sums and counts
        np.add.at(sum_x, inverse_indices, x_points)
        np.add.at(sum_y, inverse_indices, y_points)
        np.add.at(count, inverse_indices, 1)
        # Compute centroids per angle bin
        centroid_x = sum_x / count
        centroid_y = sum_y / count
        # Compute distances from centroids to light source
        dx_centroids = L_x - centroid_x
        dy_centroids = L_y - centroid_y
        distances_centroids = np.hypot(dx_centroids, dy_centroids)
        # Compute weights (you can choose between 1/distance or 1/distance^2)
        weights = 1 / distances_centroids
        # Accumulate weights in the feature vector
        angular_coverage_vector[unique_bins] += weights
    return angular_coverage_vector

def plot_feature_vector(angular_coverage_vector, resolution_degrees=1.0):
    # Generate angle bins for plotting
    num_bins = len(angular_coverage_vector)
    angle_bins = np.linspace(0, 360, num_bins, endpoint=False)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    angles_rad = np.radians(angle_bins)
    ax.bar(angles_rad, angular_coverage_vector, width=np.radians(resolution_degrees), bottom=0.0, edgecolor='k')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    plt.title('Angular Coverage Feature Vector (Polar Plot)')
    plt.show()

def calculate_light_position_deviation(Ls_location, img_path):
    # Load the image
    img = mpimg.imread(img_path)
    img_height, img_width = img.shape[:2]
    # Initialize the user-selected coordinates
    user_selected_point = []
    # Define a click event handler
    def onclick(event):
        # Check if the mouse button is pressed within the image boundaries
        if event.xdata is not None and event.ydata is not None:
            # Check if toolbar is active (zoom or pan)
            if fig.canvas.toolbar.mode == '':
                # Store the x and y coordinates
                user_selected_point.append((event.xdata, event.ydata))
                print(f"User clicked on: ({event.xdata:.2f}, {event.ydata:.2f})")
                # Disconnect the event and close the plot
                fig.canvas.mpl_disconnect(cid)
                plt.close()
            else:
                # Toolbar is active (zoom or pan), ignore the click
                pass
    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title("Zoom and pan to locate the light source.\nThen deactivate zoom/pan and click on the light source location.")
    plt.axis('off')  # Hide axis ticks and labels for better visibility
    # Connect the click event handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # Show the plot and wait for the user to click
    plt.show()
    # After the plot is closed, check if the user has selected a point
    if len(user_selected_point) == 0:
        print("No point was selected.")
        return None
    # Get the user-selected coordinates
    x_user, y_user = user_selected_point[0]
    # Get the predicted light source location
    L_x, L_y = Ls_location
    # Normalize the coordinates
    x_user_norm = x_user / img_width
    y_user_norm = y_user / img_height
    L_x_norm = L_x / img_width
    L_y_norm = L_y / img_height
    # Compute the deviation metric (Euclidean distance)
    normalized_deviation_metric = np.sqrt((x_user_norm - L_x_norm)**2 + (y_user_norm - L_y_norm)**2)
    print(f"Predicted light source location: ({L_x:.2f}, {L_y:.2f})")
    print(f"Deviation metric (pixels): {normalized_deviation_metric :.2f}")
    return normalized_deviation_metric, np.array([x_user, y_user])

# Spherical Harmonics for distant light sources
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
        # basis[i,2] = 0.488603 * z
        basis[i,2] = 0.488603 * x
        basis[i,3] = 1.092548 * x * y
        # basis[i,5] = 1.092548 * y * z
        # basis[i,6] = 0.315392 * (3 * z ** 2 - 1)
        # basis[i,7] = 1.092548 * x * z
        basis[i,4] = 0.546274 * (x ** 2 - y ** 2)
    return basis

def compute_sh_coefficients(normals, luminance, lambda_reg = 0.1):
    basis = sh_basis_cartesian(normals)
    C = np.diag([1, 2, 2, 3, 3])
    regularized_basis = basis.T @ basis + lambda_reg * C
    regularized_inverse = np.linalg.inv(regularized_basis)
    coefficients_reg = regularized_inverse @ basis.T @ luminance
    return coefficients_reg

# CALCULATE D METRIC
def corr(w1, w2):
    """
    Calculates the normalized correlation.
    Q (np.array): Matrix used for normalization.
    """
    diagonal_values = [0, np.pi/6, np.pi/6, 5*np.pi/4, 5*np.pi/4]
    Q = np.diag(diagonal_values)
    # Calculate the terms for the correlation
    term1 = np.dot(w1.T, np.dot(Q, w2))
    term2 = np.sqrt(np.dot(w1.T, np.dot(Q, w1)))
    term3 = np.sqrt(np.dot(w2.T, np.dot(Q, w2)))
    # Calculate the normalized correlation
    return term1 / (term2 * term3)

def D(w1, w2):
    """
    Calculates the distance between two spherical harmonic vectors
    """
    return 0.5 * (1 - corr(w1, w2))

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

def compute_light_direction_C(img_path, new_points_x, new_points_y, normals, ground_truth_subpixel, max_iter):
    # Read the original image for plotting later.
    org_img = cv2.imread(img_path)
    height, width = org_img.shape[:2]
    img_norm = max(height, width)
    # (Optional) Normalize contour coordinates if needed.
    spline_point_x_norm = new_points_x / img_norm
    spline_point_y_norm = new_points_y / img_norm
    n_total_points = len(new_points_x)
    # Initialize mask: start with all points.
    mask = np.ones(n_total_points, dtype=bool)
    prev_mask = None
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        current_idx = np.where(mask)[0]
        # Solve LS using only the currently filtered points.
        M = normals[current_idx]  # shape (n_filtered, 2)
        b = ground_truth_subpixel[current_idx]
        v0, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
        Lx, Ly = v0
        L_norm = np.sqrt(Lx**2 + Ly**2)
        if L_norm > 1e-12:
            L = np.array([Lx, Ly]) / L_norm
            # Ensure physically plausible: if average normal of filtered points is inward, flip L.
            n_avg = np.mean(M, axis=0)
            n_avg_norm = n_avg / np.linalg.norm(n_avg)
            if np.dot(n_avg_norm, L) < 0:
                L = -L
        else:
            L = np.array([0, 0])
        # Compute the dot product for the filtered points only.
        dots_filtered = np.sum(M * L, axis=1)
        new_mask_filtered = dots_filtered > 0  # Boolean array for filtered points
        # If very few points remain, fallback to keeping all points.
        if np.sum(new_mask_filtered) < 3:
            new_mask = np.ones_like(new_mask_filtered, dtype=bool)
        # Create a new full-length mask.
        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[current_idx] = new_mask_filtered
        # Check if the mask has converged.
        if prev_mask is not None and np.array_equal(new_mask, mask):
            break
        prev_mask = mask.copy()
        mask = new_mask.copy()
    # Final LS estimation using the converged mask.
    final_idx = np.where(mask)[0]
    M_final = normals[final_idx]
    b_final = ground_truth_subpixel[final_idx]
    v0, residuals, rank, s = np.linalg.lstsq(M_final, b_final, rcond=None)
    Lx, Ly = v0
    L_norm = np.sqrt(Lx**2 + Ly**2)
    if L_norm > 1e-12:
        L_final = np.array([Lx, Ly]) / L_norm
        n_avg = np.mean(M_final, axis=0)
        n_avg_norm = n_avg / np.linalg.norm(n_avg)
        if np.dot(n_avg_norm, L_final) < 0:
            L_final = -L_final
    else:
        L_final = L
    # Compute LS uncertainty and additional uncertainty via cost function curvature.
    normal_angles = np.arctan2(M_final[:, 1], M_final[:, 0])
    sort_idx = np.argsort(normal_angles)
    angles_sorted = normal_angles[sort_idx]
    intensity_sorted = b_final[sort_idx]
    # Uncertainty based on cost function curvature:
    def cost_function(theta):
        theoretical = np.maximum(0, np.cos(angles_sorted - theta))
        mse = np.mean((intensity_sorted - theoretical)**2)
        return mse
    theta_opt = np.arctan2(Ly, Lx)
    C0 = cost_function(theta_opt)
    delta = 1e-4  # small angular increment in radians
    C_plus = cost_function(theta_opt + delta)
    C_minus = cost_function(theta_opt - delta)
    second_deriv = (C_plus + C_minus - 2 * C0) / (delta**2)
    if second_deriv <= 0:
        curvature_uncertainty_rads = float('inf')
    else:
        curvature_uncertainty_rads = np.sqrt(2.0 / second_deriv)
    curvature_uncertainty_degs = np.degrees(curvature_uncertainty_rads)
    return L_final, curvature_uncertainty_degs

def compute_empirical_std(img_path, new_points_x, new_points_y, normals, ground_truth_subpixel, max_iter, N_max):
    # List to store the estimated light direction for each partition
    light_directions = []
    # Total number of contour points; assumed they are ordered.
    total_points = len(new_points_x)
    for N in range(1, N_max + 1):
        # Compute partition boundaries for the current N.
        indices = np.arange(total_points)
        partitions = np.array_split(indices, N)
        # For each partition, slice the arrays and compute the estimate.
        for part in partitions:
            # Ensure at least a minimal number of points
            if len(part) < 3:
                continue
            part_points_x = new_points_x[part]
            part_points_y = new_points_y[part]
            part_normals = normals[part]
            part_ground_truth = ground_truth_subpixel[part]
            # Compute light direction for this partition.
            L_est, _ = compute_light_direction_C(
                img_path, 
                part_points_x, 
                part_points_y, 
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

# Extract the normals from contours.
def get_normals_C(contour, smoothness = 10):
    '''
    This function takes as an input a contour from cv2.findContours and returns the normals of the contour by fitting a smooth spline along the contour points.
    First we fit a line to the x,y points of the contour. We find the nearest point in the spline to each contour point. The normal vector for each point at the spline is calculated as the perpendicular vector to the tangent of the spline at that point.
    The function returns the spline points, the indices of the spline points that are closest to the contour points, and the normals at the contour points, where normals is shape (n_points, 2).
    left = True indicates that the light is coming rom the left. Then we take into account only the left side of the boundary.
    '''
    contour = np.array(contour)
    x, y = contour[:, 0], contour[:, 1]
    tck, u = splprep([x, y], s=smoothness)  # s is the smoothness parameter
    dense_u = np.linspace(0, 1, 1000) # think about how to set this parameter fixed, for all resolutions, as a funciton of the number if points in the contour
    spline_point_x, spline_point_y = splev(dense_u, tck)
    der1 = splev(dense_u, tck, der=1)
    tangents = np.array(der1).T
    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    # Normalize the normals
    norms = np.linalg.norm(normals, axis=1)
    normals /= norms[:, np.newaxis]
    normals = correct_normal_orientations(x, y, spline_point_x, spline_point_y, normals)
    return spline_point_x, spline_point_y, normals

def process_contour_C(img_path, gray_image, contour, smoothness, N, sigma = 10):
    spline_point_x, spline_point_y, normals = get_normals_C(contour, smoothness)
    new_points_x, new_points_y, ground_truth_subpixel = get_subpixel_luminance(gray_image, spline_point_x, spline_point_y, normals, N)
    ground_truth_subpixel_smooth = gaussian_filter1d(ground_truth_subpixel, sigma=sigma)
    direction, angle_std_degs = compute_light_direction_C(img_path, new_points_x, new_points_y, normals, ground_truth_subpixel_smooth, max_iter=10)
    empirical_std_deg = compute_empirical_std(img_path, new_points_x, new_points_y, normals, ground_truth_subpixel_smooth, max_iter=10, N_max=3)
    return ground_truth_subpixel, direction, angle_std_degs, empirical_std_deg, normals, new_points_x, new_points_y

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
            # Assign a new contour_id based on the current length of the list
            new_id = len(contours_list) + 1
            new_contour = {
                "contour_id": new_id,
                "contour_coordinates": new_contour_data.get("contour_coordinates", [None, None]),
                "smoothness": new_contour_data.get("smoothness", None),
                "Parameter_N": new_contour_data.get("Parameter_N", None),
                "brightness_range": new_contour_data.get("brightness_range", [None]),
                "normal_vectors_span": new_contour_data.get("normal_vectors_span", [None]),
                "contour_type": new_contour_data.get("contour_type", None),
                "belongs_to_person": new_contour_data.get("belongs_to_person", None),
                "light_direction_estimation": new_contour_data.get("light_direction_estimation", [None]),
                "estimation_std_degrees": new_contour_data.get("estimation_std_degrees", [None]),
                "mean_squared_error": new_contour_data.get("mean_squared_error", [None]),
                "std_mse": new_contour_data.get("std_mse", [None]),
                "std_mse_deg": new_contour_data.get("std_mse_deg", [None]),
                "shading_rate": new_contour_data.get("shading_rate", [None]),
                "level_of_noise": new_contour_data.get("level_of_noise", [None]),
                "spherical_harmonics_coeffs": new_contour_data.get("spherical_harmonics_coeffs", [None])
            }
            contours_list.append(new_contour)
            painting["analysis"]["contours"] = contours_list
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

def vis_light_estimation_C(
    img_path,
    direction,
    x_coordinates,
    y_coordinates,
    angle_std_degs=None,
    scale=250,
    visual_factor=2.0,
    plot_Ld=True,
    plot_contours_flag=False,
    plot_only_contours=False,
    create_new_figure=True,
    show_background=True
):
    if create_new_figure:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
    else:
        ax = plt.gca()
    # 2) Optionally show the background image
    if show_background:
        org_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if org_img is not None:
            ax.imshow(
                org_img,
                cmap='gray',
                alpha=0.7,
                extent=[0, org_img.shape[1], org_img.shape[0], 0]
            )
        else:
            print(f"Could not read image from {img_path}")
    # 3) Plot either only the contour or contour + arrow
    if plot_only_contours:
        ax.plot(x_coordinates, y_coordinates, '.', markersize=4, color='green')
    else:
        # Possibly plot the contour points
        if plot_contours_flag:
            ax.plot(x_coordinates, y_coordinates, '.', markersize=2, color='green')
        # Draw the light direction arrow if requested
        if plot_Ld and direction is not None:
            center_x = np.mean(x_coordinates)
            center_y = np.mean(y_coordinates)
            dx = direction[0] * scale
            dy = direction[1] * scale
            # Arrow in yellow
            ax.arrow(
                center_x, center_y,
                dx, dy,
                head_width=30, head_length=30, width=6,
                fc='yellow', ec='yellow', zorder=5
            )
            # If we have angle uncertainty, draw wedge
            if angle_std_degs is not None:
                angle_degs = np.degrees(np.arctan2(direction[1], direction[0]))
                enlarged_std = angle_std_degs * visual_factor
                angle_min = angle_degs - enlarged_std
                angle_max = angle_degs + enlarged_std
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
    # 4) Adjust axes if we created them
    if create_new_figure and show_background:
        if org_img is not None:
            h, w = org_img.shape
            ax.set_xlim([-0.1 * w, 1.1 * w])
            ax.set_ylim([1.1 * h, -0.1 * h])
    ax.axis('off')
    # 5) If we created a figure, show it
    if create_new_figure:
        plt.show()

def plot_painting_contours_C(
    painting_id,
    json_filepath,
    img_path,
    scale=250,
    plot_contours_flag=False,
    plot_only_contours=False
):
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    target_painting = None
    for painting in data.get("paintings", []):
        if painting.get("painting_id") == painting_id:
            target_painting = painting
            break
    if target_painting is None:
        print(f"No painting found with painting_id={painting_id}")
        return
    contours_list = target_painting.get("analysis", {}).get("contours", [])
    if not contours_list:
        print(f"No contours found for painting_id={painting_id}")
        return
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    org_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if org_img is None:
        print(f"Could not load image at path: {img_path}")
        return
    ax.imshow(
        org_img,
        cmap='gray',
        alpha=0.7,
        extent=[0, org_img.shape[1], org_img.shape[0], 0]
    )
    for contour in contours_list:
        smoothness = contour.get("smoothness", None)
        N = contour.get("Parameter_N", None)
        c_coords = contour.get("contour_coordinates", None)
        if not c_coords or len(c_coords) < 2:
            print(f"Skipping contour_id={contour.get('contour_id')}, invalid coordinates.")
            continue
        (ground_truth_subpixel,
         light_direction_est,
         angle_std_degs,
         empirical_std_deg,
         normals,
         x_coords,
         y_coords) = process_contour_C(img_path, org_img, c_coords, smoothness, N, sigma = 10)
        vis_light_estimation_C(
            img_path=img_path,
            direction=light_direction_est,
            x_coordinates=x_coords,
            y_coordinates=y_coords,
            angle_std_degs=empirical_std_deg,
            scale=scale,
            visual_factor=1.0,
            plot_Ld=not plot_only_contours,
            plot_contours_flag=plot_contours_flag,
            plot_only_contours=plot_only_contours,
            create_new_figure=False,
            show_background=False
        )
    # 5) Adjust axes, no repeated background
    h, w = org_img.shape
    ax.set_xlim([-0.1 * w, 1.1 * w])
    ax.set_ylim([1.1 * h, -0.1 * h])
    ax.axis('off')
    plt.show()


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

def visualize_delta_metric(I_meas_norm, I_theo_norm,meas_block, theo_block,title='Shading distributions (index domain)',lo_frac=0.1, hi_frac=0.9):
    # convert blocks to sets for quick membership checking
    set_meas = set(meas_block)
    set_theo = set(theo_block)
    x_indices = np.arange(len(I_meas_norm))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)
    # --- Plot 1: Measured distribution
    ax1.plot(x_indices, I_meas_norm, '-o', ms=3, label='Measured')
    # highlight the sub-block
    block_meas_vals = []
    block_meas_idx = []
    for i in meas_block:
        block_meas_idx.append(i)
        block_meas_vals.append(I_meas_norm[i])
    ax1.plot(block_meas_idx, block_meas_vals, 'ro', ms=4, label='Meas sub-block')
    ax1.axhline(lo_frac, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(hi_frac, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Contour index')
    ax1.set_ylabel('Normalized brightness')
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True)
    ax1.legend()
    ax1.set_title("Measured Distribution")
    # --- Plot 2: Theoretical distribution
    ax2.plot(x_indices, I_theo_norm, '-o', ms=3, label='Theoretical')
    # highlight the sub-block
    block_theo_vals = []
    block_theo_idx = []
    for i in theo_block:
        block_theo_idx.append(i)
        block_theo_vals.append(I_theo_norm[i])
    ax2.plot(block_theo_idx, block_theo_vals, 'go', ms=4, label='Theo sub-block')
    ax2.axhline(lo_frac, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(hi_frac, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Contour index')
    ax2.set_ylabel('Normalized brightness')
    ax2.set_ylim([-0.05, 1.05])
    ax2.grid(True)
    ax2.legend()
    ax2.set_title("Theoretical Distribution")
    plt.show()

def compute_global_estimation(painting_id, img_path, json_filepath, process_contour_C, gray_image):
    # Load the JSON file
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    # Locate the target painting
    for painting in data["paintings"]:
        if painting["painting_id"] == painting_id:
            contours_list = painting["analysis"].get("contours", [])
            break
    else:
        raise ValueError(f"Painting ID {painting_id} not found in JSON file.")
    # Initialize M and b stacking lists
    all_normals = []
    all_luminance = []
    light_directions = []
    for contour in contours_list:
        contour_coords = np.array(contour.get("contour_coordinates", []))
        smoothness = contour.get("smoothness", None)
        N = contour.get("Parameter_N", None)
        if contour_coords is None or smoothness is None or N is None:
            continue  # Skip if required data is missing
        # Extract necessary variables
        ground_truth_subpixel, light_direction_estimation, _, _, normals, _, _ = process_contour_C(
            img_path, gray_image, contour_coords, smoothness, N
        )
        # Smooth the ground truth subpixel luminance
        ground_truth_subpixel_smooth = gaussian_filter1d(ground_truth_subpixel, sigma=30)
        # Stack matrices
        all_normals.append(normals)
        all_luminance.append(ground_truth_subpixel_smooth)
        light_directions.append(light_direction_estimation)
    if not all_normals:
        raise ValueError("No valid contour data found for light direction estimation.")
    # Stack all data into final matrices
    M = np.vstack([normals[:, :2] for normals in all_normals])
    b = np.concatenate(all_luminance)
    # Solve least squares problem
    v0, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
    Lx, Ly = v0
    # Normalize light direction
    L_norm = np.sqrt(Lx**2 + Ly**2)
    L = np.array([Lx, Ly]) / L_norm if L_norm > 1e-12 else None
    # compute empirical uncertainty from the light direction estimations of individual contours.
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
    return L, circ_std_deg
