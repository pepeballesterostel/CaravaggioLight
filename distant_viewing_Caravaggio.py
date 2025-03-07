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

#############################################################################################################
#############################################################################################################
# Load the master JSON
json_filepath = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json'
with open(json_filepath, 'r') as f:
    data = json.load(f)

##############################################################################
# Variables to change
painting_id = 47
utils.show_painting_summary(painting_id)
url = data['paintings'][painting_id-1]['url']           

parent_dir = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/images'
img_filename = str(painting_id) + '.jpg'
# img_filename = 'sun_sphere_lambertian.png'
img_path = os.path.join(parent_dir, img_filename)
image = cv2.imread(img_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

##############################################################################
# Extract lighting information from paintings
##############################################################################

contour = utils.extract_contour_ui(image_rgb)
# or, get contour coordinates from the JSON file by ID
# contour = data['paintings'][painting_id-1]['analysis']['contours'][0]['contour_coordinates']

smoothness = 15
N = 10
sigma = 30

ground_truth_subpixel, light_direction_estimation, angle_std_degs, empirical_std_deg, normals, x_coordinates, y_coordinates = utils.process_contour_C(img_path, gray_image, contour, smoothness, N, sigma)
utils.plot_contour_and_normals(img_path, x_coordinates, y_coordinates, normals)

# get MSE metrics
meas_smooth = gaussian_filter1d(ground_truth_subpixel, sigma=sigma)
theoretical_light = np.clip(light_direction_estimation @ normals.T, 0, 1)
I_meas_norm = utils.normalize_to_01(meas_smooth)
I_theo_norm = utils.normalize_to_01(theoretical_light)
plt.plot(I_meas_norm)
plt.plot(I_theo_norm)
plt.show()

mse, delta, meas_block, theo_block, I_meas_norm, I_theo_norm, sigma = utils.compute_shading_delta_metric(ground_truth_subpixel, light_direction_estimation, normals, sigma=sigma, max_sigma=100, sigma_step=5)
level_of_noise = utils.compute_noise_metric(ground_truth_subpixel)
_, _, _, _, std_mse_deg = utils.get_Lightdirection_pdf([light_direction_estimation], [I_meas_norm], [normals])
# compute ranges
norm_smoothed_luminance_distribution = meas_smooth / meas_smooth.max()
brightness_range = norm_smoothed_luminance_distribution.max() - norm_smoothed_luminance_distribution.min()
normal_vectors_span = utils.compute_angle_range(normals)
utils.vis_light_estimation_C(img_path, light_direction_estimation, x_coordinates, y_coordinates, empirical_std_deg, scale = 600, visual_factor = 1.0, plot_Ld=True, plot_contours_flag=True, plot_only_contours=False)


utils.visualize_delta_metric(I_meas_norm, I_theo_norm, meas_block, theo_block, title='Shading distributions (index domain)', lo_frac=0.1, hi_frac=0.9)

# SH
coefficients_reg = utils.compute_sh_coefficients(normals, norm_smoothed_luminance_distribution)
contour_type = "drapery"
belongs_to_person = "person 1"

contour_data = {    
    "contour_coordinates": contour,
    "smoothness": smoothness,
    "Parameter_N": N,
    "brightness_range": [round(brightness_range, 4)],
    "normal_vectors_span": [normal_vectors_span],
    "contour_type": contour_type,
    "belongs_to_person": belongs_to_person,
    "light_direction_estimation": [light_direction_estimation],
    "empirical_std_deg": [empirical_std_deg],
    "estimation_std_degrees": [angle_std_degs],
    "mean_squared_error": [mse],
    "std_mse_deg": [std_mse_deg],
    "shading_rate": [delta],
    "level_of_noise": [level_of_noise],
    "spherical_harmonics_coeffs": [coefficients_reg]
}

# ADD THE CONTOUR DATA TO THE JSON FILE
utils.add_contour(painting_id, contour_data)

    
# UPDATE CONTOUR DATA BY ID
utils.replace_contour(painting_id=painting_id, contour_id=4, new_contour_data = contour_data)

# # DELETE CONTOUR DATA BY ID
utils.delete_contour(painting_id=painting_id, contour_id=1)

# (OPTIONAL) 'depicted_light_direction_evidence' Cast shadows or specular highlights: light direction estiamtions with no variance
cast_shadow = utils.extract_light_direction_ui(img_path)

# depicted_light_direction_evidence = cast_shadow
depicted_light_direction_evidence = []
# For specular highlights use the python script in C:/Users/pepel/PROJECTS/FaceArt/castShadow/specular_highlights.py


depicted_light_direction_evidence = [
    {
        "type": "cast shadow",
        "light_direction": cast_shadow,
        "notes": "cast shadow of the music sheet"
    }
]

# GLOBAL CONSISTENCY METRIC (after all contours have been extracted)
global_light_direction_estimation, global_estimation_std_degrees = utils.compute_global_estimation(painting_id, img_path, json_filepath, utils.process_contour_C, gray_image)
utils.vis_global_light_estimation_C(img_path, global_light_direction_estimation, x_coordinates, y_coordinates, global_estimation_std_degrees, scale=1000, visual_factor = 1.0)

number_of_people = 1
utils.add_remaining_info(painting_id, depicted_light_direction_evidence, number_of_people, global_light_direction_estimation, global_estimation_std_degrees)
utils.plot_painting_contours_C(painting_id, json_filepath, img_path, scale=300, plot_contours_flag=True, plot_only_contours=False)





##############################################################################
# UPDATE EXISTING CONTOURS WITH LOOPING
##############################################################################


json_filepath = "C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"
with open(json_filepath, 'r') as f:
    data = json.load(f)
failed_contours = []
for painting in data.get("paintings", []):
    painting_id = painting.get("painting_id")
    contours_list = painting.get("analysis", {}).get("contours", [])
    parent_dir = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/images'
    img_filename = str(painting_id) + '.jpg'
    img_path = os.path.join(parent_dir, img_filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not load image at path: {img_path} for painting_id={painting_id}")
        continue
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not contours_list:
        print(f"No contours found for painting_id={painting_id}.")
        continue
    for contour in contours_list:
        contour_id = contour.get("contour_id")
        try:
            # Extract parameters from the contour
            smoothness = contour.get("smoothness", None)
            N = contour.get("Parameter_N", None)
            contour_coordinates = contour.get("contour_coordinates", None)
            contour_type = contour.get("contour_type", None)
            belongs_to_person = contour.get("belongs_to_person", None)
            if not contour_coordinates or len(contour_coordinates) < 2:
                raise ValueError("Invalid contour coordinates.")
            ground_truth_subpixel, light_direction_estimation, angle_std_degs, normals, x_coordinates, y_coordinates = utils.process_contour_C(img_path, gray_image, contour_coordinates, smoothness, N, sigma=30)
            # Compute shading delta metric and MSE
            result = utils.compute_shading_delta_metric(ground_truth_subpixel, light_direction_estimation, normals, sigma=10, max_sigma=100, sigma_step=5)
            if result is None:
                utils.delete_contour(painting_id=painting_id, contour_id=contour_id)
                print(f"Contour_id={contour_id} in painting_id={painting_id} removed (delta metric=NONE).")
                continue
            else:
                mse, delta, meas_block, theo_block, I_meas_norm, I_theo_norm, sigma = result
            # Compute noise metric on the raw measured brightness
            smoothed_luminance_distribution = gaussian_filter1d(ground_truth_subpixel, sigma=10)
            norm_smoothed_luminance_distribution = smoothed_luminance_distribution / smoothed_luminance_distribution.max()
            level_of_noise = utils.compute_noise_metric(ground_truth_subpixel)
            # Get light direction uncertainty metrics
            # Assuming utils.get_Lightdirection_pdf returns (probabilities, angles, variances, std_mse, std_mse_deg)
            _, _, _, std_mse, std_mse_deg = utils.get_Lightdirection_pdf(
                [light_direction_estimation],
                [norm_smoothed_luminance_distribution],
                [normals]
            )
            brightness_range = float(np.round(norm_smoothed_luminance_distribution.max() - norm_smoothed_luminance_distribution.min(), 4))
            normal_vectors_span = utils.compute_angle_range(normals)
            # Compute spherical harmonics coefficients (SH)
            coefficients_reg = utils.compute_sh_coefficients(normals, norm_smoothed_luminance_distribution)
            # Prepare new contour data (convert arrays to lists where necessary)
            new_contour_data = {
                "contour_coordinates": contour_coordinates,
                "smoothness": smoothness,
                "Parameter_N": N,
                "brightness_range": [brightness_range],
                "normal_vectors_span": [normal_vectors_span],
                "contour_type": contour_type,
                "belongs_to_person": belongs_to_person,
                "light_direction_estimation": [light_direction_estimation.tolist()],
                "estimation_std_degrees": [angle_std_degs],
                "mean_squared_error": [mse],
                "std_mse": [std_mse],
                "std_mse_deg": [std_mse_deg],
                "shading_rate": [delta],
                "level_of_noise": [level_of_noise],
                "spherical_harmonics_coeffs": [coefficients_reg.tolist()]
            }
            utils.replace_contour(painting_id=painting_id, contour_id=contour_id, new_contour_data=new_contour_data)
        except Exception as e:
            failed_contours.append((painting_id, contour.get("contour_id"), str(e)))
            print(f"Error processing painting_id={painting_id}, contour_id={contour.get('contour_id')}: {e}")
            continue
    global_light_direction_estimation, global_estimation_std_degrees = utils.compute_global_estimation(painting_id, img_path, json_filepath, utils.process_contour_C, gray_image)
    # Update the painting's analysis information with the global estimation.
    painting.setdefault("analysis", {})["global_light_direction_estimation"] = global_light_direction_estimation.tolist() if global_light_direction_estimation is not None else None
    painting["analysis"]["global_estimation_std_degrees"] = global_estimation_std_degrees
    # Optionally, plot the painting contours and global estimation to check results.
    utils.plot_painting_contours_C(painting_id, json_filepath, img_path, scale=800, plot_contours_flag=True, plot_only_contours=False)
    utils.vis_global_light_estimation_C(img_path, global_light_direction_estimation, x_coordinates, y_coordinates, global_estimation_std_degrees, scale=1000, visual_factor=2.0)



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
# BACKGROUD ANALYSIS
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