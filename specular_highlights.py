'''
Run all the code. 
first, select the circle and when finished press c. The function will return the center and the radii in pxz. 
Next, select the point on the specular highlight 
The fuction will return the Light direction from the center of the sphere to the specular highlight. z
'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# Function to fit a circle given points
def fit_circle(points):
    def residuals(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r
    if len(points) < 3:
        return None  # Not enough points to fit a circle
    x, y = points[:, 0], points[:, 1]
    x_m, y_m = np.mean(x), np.mean(y)
    initial_guess = [x_m, y_m, np.std(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))]
    result = least_squares(residuals, initial_guess, args=(x, y))
    if result.success:
        return result.x  # xc, yc, r
    else:
        return None

# Global variables
selected_points = []
mode = 'zoom'
point_size = 1
circle_center = None
zoom_extent = None

# Paths
painting_id = 69
parent_dir = 'C:/Users/pepel/PROJECTS/DATA/Caravaggio/images'
img_filename = str(painting_id) + '.jpg'
img_path = os.path.join(parent_dir, img_filename)
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mouse click event handler
def onclick(event):
    global selected_points, mode
    if mode == 'annotation' and event.xdata is not None and event.ydata is not None:
        selected_points.append((int(event.xdata), int(event.ydata)))
        draw_current_selections()

# Draw current selections
def draw_current_selections():
    img_with_points = image.copy()
    for pt in selected_points:
        cv2.circle(img_with_points, pt, point_size, (255, 0, 0), -1)
    plt.imshow(img_with_points)
    if zoom_extent:
        ax.set_xlim(zoom_extent[0], zoom_extent[1])
        ax.set_ylim(zoom_extent[3], zoom_extent[2])
    plt.axis('off')
    plt.draw()

# Draw the circle and center
def draw_circle_and_center():
    global circle_center
    if len(selected_points) >= 3:
        points = np.array(selected_points)
        result = fit_circle(points)
        if result is None:
            print("Failed to fit a circle. Please ensure the points are well-distributed.")
            return
        xc, yc, r = result
        circle_center = (xc, yc)
        theta = np.linspace(0, 2 * np.pi, 500)
        x_circle = (xc + r * np.cos(theta)).astype(int)
        y_circle = (yc + r * np.sin(theta)).astype(int)
        img_with_circle = image.copy()
        for pt in selected_points:
            cv2.circle(img_with_circle, pt, point_size, (255, 0, 0), -1)
        for x, y in zip(x_circle, y_circle):
            if 0 <= x < img_with_circle.shape[1] and 0 <= y < img_with_circle.shape[0]:
                cv2.circle(img_with_circle, (x, y), 1, (0, 255, 0), -1)
        cv2.circle(img_with_circle, (int(xc), int(yc)), 1, (0, 0, 255), -1)
        plt.imshow(img_with_circle)
        if zoom_extent:
            ax.set_xlim(zoom_extent[0], zoom_extent[1])
            ax.set_ylim(zoom_extent[3], zoom_extent[2])
        plt.axis('off')
        plt.draw()
        print(f"Circle center: ({xc:.2f}, {yc:.2f}), Radius: {r:.2f}")
    else:
        print("At least 3 points are required to fit a circle.")

# Key press event handler
def on_key(event):
    global mode, selected_points
    if event.key == 'p':
        mode = 'annotation'
        print("Mode changed to: Annotation")
    elif event.key == 'z':
        mode = 'zoom'
        print("Mode changed to: Zoom")
    elif event.key == 'c':
        print("Calculating and drawing circle...")
        draw_circle_and_center()
    elif event.key == 'r':
        selected_points.clear()
        print("Selected points cleared.")
        plt.imshow(image)
        if zoom_extent:
            ax.set_xlim(zoom_extent[0], zoom_extent[1])
            ax.set_ylim(zoom_extent[3], zoom_extent[2])
        plt.axis('off')
        plt.draw()
    elif event.key == 'q':
        print("Exiting...")
        plt.close()

# Function to calculate and plot light direction
def calculate_light_direction(image, circle_center):
    mode = 'zoom'
    highlight_point = None
    normalized_light_direction = None
    def onclick_light(event):
        nonlocal highlight_point, normalized_light_direction
        if mode == 'annotation' and event.xdata is not None and event.ydata is not None:
            xh, yh = int(event.xdata), int(event.ydata)
            highlight_point = (xh, yh)
            xc, yc = circle_center
            light_vector = np.array([xh - xc, yh - yc])
            normalized_light_direction = light_vector / np.linalg.norm(light_vector)
            img_with_line = image.copy()
            height, width = img_with_line.shape[:2]
            # Extend line to the image boundary
            if light_vector[0] != 0:  # Avoid division by zero
                slope = light_vector[1] / light_vector[0]
                if light_vector[0] > 0:
                    x_end = width
                else:
                    x_end = 0
                y_end = yc + slope * (x_end - xc)
                # Clamp y_end to image boundaries if out of bounds
                if y_end < 0:
                    y_end = 0
                    x_end = xc - (yc / slope) if slope != 0 else xc
                elif y_end > height:
                    y_end = height
                    x_end = xc + ((height - yc) / slope) if slope != 0 else xc
            else:  # Vertical line
                x_end = xc
                y_end = 0 if light_vector[1] < 0 else height
            cv2.line(img_with_line, (int(xc), int(yc)), (int(x_end), int(y_end)), (255, 255, 0), 2)
            cv2.circle(img_with_line, (xh, yh), 1, (0, 255, 255), -1)
            plt.imshow(img_with_line)
            plt.axis('off')
            plt.draw()
            print(f"Normalized light direction: {normalized_light_direction}")
    def on_key(event):
        nonlocal mode
        if event.key == 'z':
            mode = 'zoom'
            print("Mode changed to: Zoom")
        elif event.key == 'p':
            mode = 'annotation'
            print("Mode changed to: Annotation")
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.axis('off')
    def onselect(eclick, erelease):
        if mode == 'zoom':
            ax.set_xlim(eclick.xdata, erelease.xdata)
            ax.set_ylim(erelease.ydata, eclick.ydata)
            plt.draw()
    from matplotlib.widgets import RectangleSelector
    rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    fig.canvas.mpl_connect('button_press_event', onclick_light)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return normalized_light_direction

# Create the figure and axes
fig, ax = plt.subplots()
ax.imshow(image)
if zoom_extent:
    ax.set_xlim(zoom_extent[0], zoom_extent[1])
    ax.set_ylim(zoom_extent[3], zoom_extent[2])
ax.axis('off')
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

normalized_light_direction = calculate_light_direction(image, circle_center)