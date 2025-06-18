'''
Analysis of distant viewing Caravggio lighting results from caravaggio.json
Here we want to implement the code to analyse the results and generate visuals that respond to distant viewing questions. 
'''

import json
import os
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt


##############################################################################
# DISTANT VIEWING ANALYSIS
##############################################################################


json_filepath = "C:/Users/pepel/PROJECTS/DATA/Caravaggio/caravaggio.json"
with open(json_filepath, 'r') as f:
    data = json.load(f)

# 
##############################################################################
# 1. LIGHT COLLAGE PER PAINTING. Clustering of light directions for each painting
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def extract_painting_angles(data, painting_id):
    """
    For a given painting ID, extract the light direction estimation angles (in radians)
    from all contours. Also returns the associated contour types.
    """
    for painting in data["paintings"]:
        if painting["painting_id"] == painting_id:
            contours = painting["analysis"]["contours"]
            angles = []
            contour_types = []
            for contour in contours:
                # Assume each contour has one light_direction_estimation vector.
                ld = contour["light_direction_estimation"][0]
                # Compute the angle using arctan2 (result in radians).
                angle = np.arctan2(-ld[1], ld[0])
                # Normalize angle to the range [0, 2π)
                if angle < 0:
                    angle += 2 * np.pi
                angles.append(angle)
                contour_types.append(contour["contour_type"])
            return np.array(angles), contour_types
    return None, None


def optimal_kmeans(angles, max_k=10):
    X = np.column_stack((np.cos(angles), np.sin(angles)))
    # Limit the maximum number of clusters to the number of points.
    max_k = min(max_k, len(angles))
    best_k = 2
    best_score = -1
    best_model = None
    best_labels = None
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"Number of clusters: {k}, silhouette score: {score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
            best_labels = labels
    return best_model, best_k, best_score, best_labels


def plot_clustered_light_directions(angles, labels, cluster_centers):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    cmap = plt.get_cmap("winter")
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    for angle, label in zip(angles, labels):
        color = cmap(label / (k - 1)) if k > 1 else "blue"
        ax.annotate("", xy=(angle, 1), xytext=(angle, 0),
                    arrowprops=dict(arrowstyle="->", color=color, linewidth=2))
    ref_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    threshold = np.deg2rad(5)
    cluster_center_angles = []
    for center in cluster_centers:
        center_angle = np.arctan2(center[1], center[0])
        if center_angle < 0:
            center_angle += 2 * np.pi
        cluster_center_angles.append(center_angle)
    cluster_center_angles = np.array(cluster_center_angles)
    for ra in ref_angles:
        if not np.any(np.abs(ra - cluster_center_angles) < threshold):
            if ra == 0:
                ha = "left"
                va = "center"
                r = 1.05
            elif ra == np.pi/2:
                ha = "center"
                va = "bottom"
                r = 1.05
            elif ra == np.pi:
                ha = "right"
                va = "center"
                r = 1.05
            elif ra == 3*np.pi/2:
                ha = "center"
                va = "top"
                r = 1.05
            ax.text(ra, r, f"{np.rad2deg(ra):.0f}°", ha=ha, va=va, fontsize=14, color="black")
    for center in cluster_centers:
        center_angle = np.arctan2(center[1], center[0])
        if center_angle < 0:
            center_angle += 2 * np.pi
        ax.text(center_angle, 1.15, f"{np.rad2deg(center_angle):.1f}°", ha="center", va="center", fontsize=14, color="red")
    pos = ax.get_position()
    ax_cart = fig.add_axes(pos, frameon=False)
    ax_cart.set_xlim(-1.2, 1.2)
    ax_cart.set_ylim(-1.2, 1.2)
    ax_cart.set_xticks([])
    ax_cart.set_yticks([])
    ax_cart.patch.set_alpha(0)
    plt.tight_layout()
    plt.show()

angles, contour_types = extract_painting_angles(data, painting_id)
utils.plot_painting_contours_C(painting_id, json_filepath, img_path, scale=700, plot_contours_flag=True, plot_only_contours=False, plot_all_contours=True, contour_ids=None)
model, k, score, labels = optimal_kmeans(angles, max_k=5)
plot_clustered_light_directions(angles, labels, model.cluster_centers_)


#################################################################################
# 2. Clustering of light directions for all paintings using the global light direction estimation

def extract_global_angles(data):
    global_angles = []
    for painting in data["paintings"]:
        analysis = painting.get("analysis", {})
        if "global_light_direction_estimation" in analysis:
            gl = analysis["global_light_direction_estimation"]
            angle = np.arctan2(-gl[1], gl[0])
            if angle < 0:
                angle += 2 * np.pi
            global_angles.append(angle)
    return np.array(global_angles)


global_angles = extract_global_angles(data)
model, best_k, best_score, labels = optimal_kmeans(global_angles, max_k=5)
cluster_centers = model.cluster_centers_
plot_clustered_light_directions(global_angles, labels, cluster_centers)

# extract only paintings from the right
def extract_paintings_from_right(data):
    right_ids = []
    for painting in data["paintings"]:
        analysis = painting.get("analysis", {})
        gl = analysis.get("global_light_direction_estimation")
        if gl is None:
            continue
        # compute angle in radians [0,2π)
        angle = np.arctan2(-gl[1], gl[0])
        if angle < 0:
            angle += 2 * np.pi
        # convert to degrees
        angle_deg = np.degrees(angle)
        # check if between 0° and 90°
        if 0 <= angle_deg <= 90:
            right_ids.append(painting["painting_id"])
    return right_ids

right_paintings = extract_paintings_from_right(data)
print(f"Found {len(right_paintings)} paintings with light from the right:")
print(right_paintings)

# Light direction in time: using the global light direction estimation
import re
from collections import Counter

def clean_date(date_str):
    if not date_str:
        return None
    date_str = date_str.lower()
    date_str = re.sub(r'c[.]?\s?|ca[.]?\s?', '', date_str)
    years = re.findall(r'\d{4}', date_str)
    if years:
        return int(years[0])
    return None

# MEAN PEOPLE DEPICTED BY TIME
early_counts = []
late_counts = []
for p in data["paintings"]:
    y = clean_date(p.get("date", ""))
    if y is None:
        continue
    num_pers = p.get("analysis", {}).get("num_people")
    if num_pers is None:
        continue
    if 1592 <= y <= 1599:
        early_counts.append(num_pers)
    elif 1600 <= y <= 1610:
        late_counts.append(num_pers)

early = np.array(early_counts)
late  = np.array(late_counts)
mean_early = early.mean()
mean_late  = late.mean()
pct_change = 100 * (mean_late - mean_early) / mean_early

# Distribution of years
years = []
for p in data.get("paintings", []):
    contours = p.get("analysis", {}).get("contours", [])
    if not contours:
        continue                       # skip un‐analysed paintings
    y = clean_date(p.get("date", ""))
    if y is not None:
        years.append(y)

year_counts = Counter(years)
# sort by year
sorted_years = sorted(year_counts.items())
# 6) (Optional) Plot as a bar chart
plt.figure(figsize=(8,4))
yrs, cnts = zip(*sorted_years)
plt.bar(yrs, cnts, width=0.8, edgecolor='black')
plt.xlabel("Year")
plt.ylabel("Number of analysed paintings")
plt.title("Distribution of Analysis‐Done Paintings by Year")
plt.xticks(yrs, rotation=45)
plt.tight_layout()
plt.show()

def extract_global_light_and_dates(data):
    results = []
    for painting in data["paintings"]:
        analysis = painting.get("analysis", {})
        if "global_light_direction_estimation" in analysis:
            gl = analysis["global_light_direction_estimation"]
            angle = np.arctan2(-gl[1], gl[0])
            if angle < 0:
                angle += 2 * np.pi
            s_value = abs(np.sin(angle))
            year = clean_date(painting.get("date", ""))
            if year is not None:
                results.append((year, s_value))
    results.sort(key=lambda x: x[0])
    return results

points = extract_global_light_and_dates(data)

years, s_values = zip(*points)
plt.figure(figsize=(10,6))
plt.scatter(years, s_values, color="blue", s=50, label="Global Light Direction (|sin(θ)|)")
coeffs = np.polyfit(years, s_values, 1)
trend_line = np.poly1d(coeffs)
years_fit = np.linspace(min(years), max(years), 100)
plt.plot(years_fit, trend_line(years_fit), color="red", linewidth=2, label="Trend")
plt.xlabel("Year")
plt.ylabel("|sin(θ)|")
plt.title("Global Light Direction Trend Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# Light direction in time: lowest std contour in light direction estimation
def extract_depicted_light_direction_and_date(data):
    results = []
    for painting in data["paintings"]:
        analysis = painting.get("analysis", {})
        evidence = analysis.get("depicted_light_direction_evidence", [])
        year = clean_date(painting.get("date", ""))
        if year is None:
            continue
        for item in evidence:
            if isinstance(item, dict):
                ld = item.get("light_direction")
                if ld is None:
                    continue
                angle = np.arctan2(-ld[1], ld[0])
                if angle < 0:
                    angle += 2 * np.pi
                s_value = abs(np.sin(angle))
                results.append((year, s_value))
    results.sort(key=lambda x: x[0])
    return results

points = extract_depicted_light_direction_and_date(data)

years, s_values = zip(*points)
plt.figure(figsize=(10,6))
plt.scatter(years, s_values, color="blue", s=50, label="Meaningful Light Direction (|sin(θ)|)")
coeffs = np.polyfit(years, s_values, 1)
trend_line = np.poly1d(coeffs)
years_fit = np.linspace(min(years), max(years), 100)
plt.plot(years_fit, trend_line(years_fit), color="red", linewidth=2, label="Trend")
plt.xlabel("Year")
plt.ylabel("|sin(θ)|")
plt.title("Meaningful Light Direction Trend Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# basic stats about # of contours and contour segments
def compute_contour_stats(data):
    """
    Returns per‐painting stats:
      - contour_counts: number of contours per painting
      - point_counts: total number of points (contour_coordinates) per painting
      - mean_contours: average #contours across paintings
      - mean_points: average #points across paintings
    """
    contour_counts = []
    point_counts   = []
    for p in data.get("paintings", []):
        contours = p.get("analysis", {}).get("contours", [])
        if not contours:
            continue
        contour_counts.append(len(contours))
        # sum the lengths of each contour's coordinate list
        total_pts = sum(len(c.get("contour_coordinates", []))
                        for c in contours)
        point_counts.append(total_pts)
    contour_counts = np.array(contour_counts)
    point_counts   = np.array(point_counts)
    return {
        "contour_counts": contour_counts,
        "point_counts": point_counts,
        "mean_contours": contour_counts.mean() if contour_counts.size else 0,
        "mean_points":   point_counts.mean()   if point_counts.size   else 0
    }

# Usage
stats = compute_contour_stats(data)
print("Mean #contours per painting:     ", stats["mean_contours"])
print("Mean #points per painting:       ", stats["mean_points"])

# Paintings with congruent/incongruent Ld evidence from "depicted_light_direction_evidence"
def count_paintings_with_depicted_evidence(data):
    count = 0
    for painting in data["paintings"]:
        analysis = painting.get("analysis", {})
        evidence = analysis.get("depicted_light_direction_evidence", [])
        if evidence and len(evidence) > 0:
            count += 1
    return count

def count_paintings_with_multiple_evidence(data):
    count = 0
    for painting in data["paintings"]:
        analysis = painting.get("analysis", {})
        evidence = analysis.get("depicted_light_direction_evidence", [])
        if len(evidence) > 1:
            count += 1
    return count

def compute_mean_resultant_length(angles):
    n = len(angles)
    sum_cos = np.sum(np.cos(angles))
    sum_sin = np.sum(np.sin(angles))
    R = np.sqrt(sum_cos**2 + sum_sin**2) / n
    return R

def extract_R_values(data):
    results = []
    for painting in data["paintings"]:
        analysis = painting.get("analysis", {})
        evidence = analysis.get("depicted_light_direction_evidence", [])
        if len(evidence) > 1:
            angles = []
            for item in evidence:
                if isinstance(item, dict) and "light_direction" in item:
                    ld = item["light_direction"]
                    angle = np.arctan2(-ld[1], ld[0])
                    if angle < 0:
                        angle += 2 * np.pi
                    angles.append(angle)
            if len(angles) > 1:
                R = compute_mean_resultant_length(angles)
                results.append(R)
    return results

evidence_count = count_paintings_with_depicted_evidence(data)
total_multiple = count_paintings_with_multiple_evidence(data)

R_values = extract_R_values(data)
plt.figure(figsize=(10,6))
min_R = min(R_values)
max_R = max(R_values)
bins = np.linspace(min_R - 0.001, max_R + 0.001, 51)
plt.hist(R_values, bins=bins, color="blue", edgecolor="black", alpha=0.7)
plt.xlabel("Mean Resultant Length (R)")
plt.ylabel("Number of Paintings")
plt.title("Histogram of Light Direction Congruence")
plt.tight_layout()
plt.show()

def circular_diff_degrees(angle1, angle2):
    diff_deg = abs(np.rad2deg(angle1 - angle2))
    return min(diff_deg, 360 - diff_deg)


def compute_angle_differences(data):
    results = []
    for painting in data["paintings"]:
        evidence = painting.get("analysis", {}).get("depicted_light_direction_evidence", [])
        if len(evidence) < 2:
            continue
        angles = []
        for item in evidence:
            if isinstance(item, dict) and "light_direction" in item:
                ld = item["light_direction"]
                angle = np.arctan2(-ld[1], ld[0])
                if angle < 0:
                    angle += 2 * np.pi
                angles.append(angle)
        if len(angles) < 2:
            continue
        differences = []
        n = len(angles)
        for i in range(n):
            for j in range(i + 1, n):
                differences.append(circular_diff_degrees(angles[i], angles[j]))
        if differences:
            max_diff = max(differences)
            results.append({
                "painting_id": painting.get("painting_id"),
                "title": painting.get("title"),
                "year": clean_date(painting.get("date", "")),
                "differences": differences,
                "max_difference": max_diff
            })
    return results

results = compute_angle_differences(data)
results_sorted = sorted(results, key=lambda r: r["max_difference"], reverse=True)

labels = []
max_diffs = []
for r in results_sorted:
    title = r["title"]
    year = r["year"]
    label = f"{title} ({year})" if year is not None else title
    labels.append(label)
    max_diffs.append(r["max_difference"])

fig, ax = plt.subplots(figsize=(10, 0.5 * len(labels) + 2))
y_pos = np.arange(len(labels))
colors = ["red" if diff > 40 else "green" for diff in max_diffs]
ax.barh(y_pos, max_diffs, align="center", color=colors, edgecolor="black")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Maximum Angle Difference (°)")
ax.set_title("Paintings with Incongruent Light Depiction Evidence")
for i, diff in enumerate(max_diffs):
    ax.text(diff + 1, i, f"{diff:.1f}°", va="center", fontsize=10)

ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.axvline(90, color="black", linestyle="--", linewidth=1)
ax.axvline(180, color="black", linestyle="--", linewidth=1)
ax.axvline(270, color="black", linestyle="--", linewidth=1)
plt.tight_layout()
plt.show()

def find_painting_with_min_R(data):
    min_R = float('inf')
    min_painting = None
    for painting in data["paintings"]:
        analysis = painting.get("analysis", {})
        evidence = analysis.get("depicted_light_direction_evidence", []) 
        # collect valid angles
        angles = []
        for item in evidence:
            if isinstance(item, dict) and "light_direction" in item:
                dx, dy = item["light_direction"]
                angle = np.arctan2(-dy, dx)         # screen‐coord to math‐coord
                if angle < 0:                      
                    angle += 2 * np.pi
                angles.append(angle)
        # only compute R if there are at least two samples
        if len(angles) > 1:
            R = compute_mean_resultant_length(angles)
            if R < min_R:
                min_R = R
                min_painting = painting
    return min_painting, min_R

min_painting, min_R = find_painting_with_min_R(data)


# Light direction by contour type: see if some contours are more tied to certain directions (e.g., faces, hands.)
from collections import defaultdict, Counter
import math

def get_contour_types(data):
    contour_types = []
    for painting in data["paintings"]:
        contours = painting.get("analysis", {}).get("contours", [])
        for contour in contours:
            ct = contour.get("contour_type", "").strip().lower()
            if ct:
                contour_types.append(ct)
    return contour_types

contour_types = get_contour_types(data)
type_counts = Counter(contour_types)
# This is a plot of contour types as recorded in the JSON datafile.
plt.figure(figsize=(10,6))
types = list(type_counts.keys())
counts = list(type_counts.values())
y_pos = np.arange(len(types))
plt.barh(y_pos, counts, align="center", color="blue", edgecolor="black")
plt.yticks(y_pos, types)
plt.xlabel("Number of Occurrences")
plt.ylabel("Contour Type")
plt.title("Contour Type Distribution")
plt.tight_layout()
plt.show()

def normalize_contour_type(ct):
    ct = ct.strip().lower()
    if "drapery" in ct or "drapey" in ct:
        return "drapery"
    for word in ["fruit", "basket", "lemon", "bowl", "skull", "instrument", "rock", "sword", "halo", "horse", "bread", "stone"]:
        if word in ct:
            return "still-life"
    for word in ["finguer", "leg", "hand", "baby", "madonna", "angel", "knee", "ear", "hands"]:
        if word in ct:
            return "person"
    return ct

def get_all_normalized_contour_types(data):
    types_list = []
    for painting in data["paintings"]:
        contours = painting.get("analysis", {}).get("contours", [])
        for contour in contours:
            ct = contour.get("contour_type", "")
            if ct:
                normalized = normalize_contour_type(ct)
                types_list.append(normalized)
    return types_list

# This is to plot the homoenized contour types as a count plot. 
contour_types = get_all_normalized_contour_types(data)
type_counts = Counter(contour_types)
plt.figure(figsize=(10,6))
types = list(type_counts.keys())
counts = list(type_counts.values())
y_pos = np.arange(len(types))
plt.barh(y_pos, counts, align="center", color="blue", edgecolor="black")
plt.yticks(y_pos, types)
plt.xlabel("Number of Occurrences")
plt.ylabel("Contour Type")
plt.title("Contour Type Distribution")
plt.tight_layout()
plt.show()


def get_contour_angles_by_type(data):
    type_angles    = defaultdict(list)
    total_contours = 0
    skipped        = 0
    for painting in data["paintings"]:
        contours = painting.get("analysis", {}).get("contours", [])
        for contour in contours:
            ct = contour.get("contour_type", "")
            if not ct:
                continue
            norm_ct = normalize_contour_type(ct)
            ld_list = contour.get("light_direction_estimation", [])
            # Case A: flat [dx, dy]
            if (isinstance(ld_list, list)
                and len(ld_list) == 2
                and all(isinstance(v, (int, float)) for v in ld_list)):
                dx, dy = ld_list
            # Case B: nested [[dx, dy]]
            elif (isinstance(ld_list, list)
                  and len(ld_list) == 1
                  and isinstance(ld_list[0], (list, tuple))
                  and len(ld_list[0]) == 2
                  and all(isinstance(v, (int, float)) for v in ld_list[0])):
                dx, dy = ld_list[0]
            # Case C: list of dicts [{"light_direction": [dx, dy]}, …]
            elif (isinstance(ld_list, list)
                  and len(ld_list) > 0
                  and isinstance(ld_list[0], dict)
                  and "light_direction" in ld_list[0]):
                dx, dy = ld_list[0]["light_direction"]
            else:
                # truly malformed entry
                skipped += 1
                continue
            # convert to angle in [0, 2π)
            angle = np.arctan2(-dy, dx)
            if angle < 0:
                angle += 2 * np.pi
            type_angles[norm_ct].append(angle)
            total_contours += 1
    print(f"Skipped {skipped} invalid entries.")
    return type_angles, total_contours


type_angles, total_contours = get_contour_angles_by_type(data)
print(f"Extracted {total_contours} contours across {len(type_angles)} types.")

# This is to plot the rose diagrams of every homogenized contour type. 
type_angles, total_contours = get_contour_angles_by_type(data)
print("Total number of contours:", total_contours)
counts = Counter({k: len(v) for k, v in type_angles.items()})
print("Normalized Contour Type Counts:")
for t, cnt in counts.items():
    print(f"{t}: {cnt}")

categories = list(type_angles.keys())
num_categories = len(categories)
cols = 4
rows = math.ceil(num_categories / cols)
fig, axes = plt.subplots(rows, cols, subplot_kw={"projection": "polar"}, figsize=(4 * cols, 4 * rows))
axes = axes.flatten()
for i, cat in enumerate(categories):
    ax = axes[i]
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    angles = type_angles[cat]
    bins = np.linspace(0, 2 * np.pi, 25)
    ax.hist(angles, bins=bins, color="skyblue", edgecolor="black", alpha=0.8)
    # Remove default title and add external text for category label
    ax.text(0.5, 1.0, f"{cat} (n={len(angles)})", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    

for j in range(num_categories, len(axes)):
    fig.delaxes(axes[j])


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# CLEAN THE LIGHT DIRECTION ESTIMATION VECTORS FOR STANDARD FORMATTING
def clean_light_directions(data):
    import copy
    cleaned_data = copy.deepcopy(data)
    cleaned = 0
    skipped = []
    for painting in cleaned_data.get("paintings", []):
        pid = painting.get("painting_id")
        for contour in painting.get("analysis", {}).get("contours", []):
            raw = contour.pop("light_direction_estimation", None)
            dx_dy = None
            # Case A: flat [dx, dy]
            if isinstance(raw, list) and len(raw) == 2 \
               and all(isinstance(v, (int, float)) for v in raw):
                dx_dy = raw
            # Case B: nested [[dx, dy]]
            elif isinstance(raw, list) and len(raw) == 1 \
                 and isinstance(raw[0], (list, tuple)) and len(raw[0]) == 2 \
                 and all(isinstance(v, (int, float)) for v in raw[0]):
                dx_dy = list(raw[0])
            # Case C: list of dicts [{"light_direction": [dx,dy]}, …]
            elif isinstance(raw, list) and len(raw) > 0 \
                 and isinstance(raw[0], dict) and "light_direction" in raw[0]:
                ld = raw[0]["light_direction"]
                if isinstance(ld, (list, tuple)) and len(ld) == 2 \
                   and all(isinstance(v, (int, float)) for v in ld):
                    dx_dy = list(ld)
            if dx_dy is not None:
                contour["light_direction"] = dx_dy
                cleaned += 1
            else:
                skipped.append({
                    "painting_id": pid,
                    "contour_id": contour.get("contour_id"),
                    "raw_value": raw
                })
    return cleaned_data, cleaned, skipped

cleaned_data_ld, num, skips = clean_light_directions(data)

def clean_global_light_directions(data):
    import copy
    cleaned_data = copy.deepcopy(data)
    cleaned = 0
    skipped = []
    for painting in cleaned_data.get("paintings", []):
        pid = painting.get("painting_id")
        analysis = painting.setdefault("analysis", {})
        raw = analysis.pop("global_light_direction_estimation", None)
        dx_dy = None
        # Case A: flat [dx, dy]
        if isinstance(raw, list) and len(raw) == 2 \
           and all(isinstance(v, (int, float)) for v in raw):
            dx_dy = raw
        # Case B: nested [[dx, dy]]
        elif isinstance(raw, list) and len(raw) == 1 \
             and isinstance(raw[0], (list, tuple)) and len(raw[0]) == 2 \
             and all(isinstance(v, (int, float)) for v in raw[0]):
            dx_dy = list(raw[0])
        # Case C: dict with "light_direction"
        elif isinstance(raw, dict) and "light_direction" in raw:
            ld = raw["light_direction"]
            if isinstance(ld, (list, tuple)) and len(ld) == 2 \
               and all(isinstance(v, (int, float)) for v in ld):
                dx_dy = list(ld)
        # Case D: list of dicts [{"light_direction": [dx,dy]}, …]
        elif isinstance(raw, list) and len(raw) > 0 \
             and isinstance(raw[0], dict) and "light_direction" in raw[0]:
            ld = raw[0]["light_direction"]
            if isinstance(ld, (list, tuple)) and len(ld) == 2 \
               and all(isinstance(v, (int, float)) for v in ld):
                dx_dy = list(ld)
        if dx_dy is not None:
            analysis["global_light_direction"] = dx_dy
            cleaned += 1
        else:
            skipped.append({
                "painting_id": pid,
                "raw_value": raw
            })
    return cleaned_data, cleaned, skipped

cleaned_data, num_cleaned, skips = clean_global_light_directions(cleaned_data_ld)
print(f"Cleaned global light direction for {num_cleaned} paintings.")
print(f"Skipped {len(skips)} malformed global entries:")
for entry in skips[:5]:
    print(" ", entry)


# analyse empirical_std_deg values by contour type
def get_contour_std_by_type(data):
    type_stds = defaultdict(list)
    total_contours = 0
    for painting in data["paintings"]:
        for contour in painting.get("analysis", {}).get("contours", []):
            ct = contour.get("contour_type", "")
            std_deg = contour.get("empirical_std_deg", None)
            if ct and std_deg is not None:
                norm_ct = normalize_contour_type(ct)
                type_stds[norm_ct].append(std_deg)
                total_contours += 1
    return type_stds, total_contours

type_stds, total_contours = get_contour_std_by_type(cleaned_data)

# Counts per type
counts = {t: len(vals) for t, vals in type_stds.items()}
print("Contour counts per type:")
for t, cnt in counts.items():
    print(f"  {t}: {cnt}")

# Descriptive stats per type
print("\nStd‐value summary per type (deg):")
for t, vals in type_stds.items():
    arr = np.array(vals)
    print(f"  {t:15s} n={len(arr):3d}  μ={arr.mean():5.2f}  σ={arr.std():5.2f}  min={arr.min():5.2f}  max={arr.max():5.2f}")

# --- Plotting


# Light direction by genre type (e.g., see if portrait is more consistent than other genres.)
def extract_global_angles_by_type(data):
    # Group global light direction angles by painting type.
    groups = defaultdict(list)
    for painting in data["paintings"]:
        analysis = painting.get("analysis", {})
        gl = analysis.get("global_light_direction")
        if gl is None:
            continue
        # Compute the angle in image coordinates.
        angle = np.arctan2(-gl[1], gl[0])
        if angle < 0:
            angle += 2 * np.pi
        # Normalize painting type.
        ptype = painting.get("type", "").strip().lower()
        if ptype:
            groups[ptype].append(angle)
    return groups

def mean_resultant_length(angles):
    n = len(angles)
    sum_cos = np.sum(np.cos(angles))
    sum_sin = np.sum(np.sin(angles))
    R = np.sqrt(sum_cos**2 + sum_sin**2) / n
    return R

def circular_std(R):
    # Avoid log(0) issues.
    if R <= 0:
        return float('inf')
    return math.sqrt(-2 * np.log(R))

def normalize_painting_type(ptype):
    ptype = ptype.strip().lower()
    if ptype == "still-life":
        return "genre"
    return ptype

#Count plot for painting types
type_counts = Counter()
paintings_by_type = defaultdict(list)
for painting in cleaned_data["paintings"]:
    ptype = painting.get("type", "").strip().lower()
    ptype = normalize_painting_type(ptype)
    if ptype:
        type_counts[ptype] += 1
        paintings_by_type[ptype].append(painting)

types = list(type_counts.keys())
counts = [type_counts[t] for t in types]
plt.figure(figsize=(10,6))
plt.bar(types, counts, color="skyblue", edgecolor="black")
plt.xlabel("Painting Type")
plt.ylabel("Number of Paintings")
plt.title("Number of Paintings per Type")
plt.tight_layout()
plt.show()

# Analysis: for each painting, extract global light direction estimation and its std (in degrees)
# Group these by painting type.
global_angles_by_type = defaultdict(list)
global_std_by_type = defaultdict(list)
for ptype, paintings in paintings_by_type.items():
    for painting in paintings:
        analysis = painting.get("analysis", {})
        gl = analysis.get("global_light_direction")
        std_deg = analysis.get("global_estimation_std_degrees")
        if gl is not None and std_deg is not None:
            angle = np.arctan2(-gl[1], gl[0])
            if angle < 0:
                angle += 2 * np.pi
            global_angles_by_type[ptype].append(angle)
            global_std_by_type[ptype].append(std_deg)

print("Global Light Direction Consistency by Painting Type:")
for t in types:
    if global_angles_by_type[t]:
        R = mean_resultant_length(global_angles_by_type[t])
        avg_std = np.mean(global_std_by_type[t])
        mean_angle = np.arctan2(np.sum(np.sin(global_angles_by_type[t])), np.sum(np.cos(global_angles_by_type[t])))
        if mean_angle < 0:
            mean_angle += 2 * np.pi
        print(f"Type: {t}, Count: {len(global_angles_by_type[t])}, Mean R: {R:.3f}, Avg Global Std: {avg_std:.1f}°, Mean Angle: {np.rad2deg(mean_angle):.1f}°")

# Plot a boxplot for global_estimation_std_degrees per TYPE
types_with_data = [t for t in types if global_std_by_type[t]]
data_to_plot = [global_std_by_type[t] for t in types_with_data]
fig, ax = plt.subplots(figsize=(10,6))
ax.boxplot(data_to_plot, labels=types_with_data)
ax.set_xlabel("Painting Type")
ax.set_ylabel("Global Estimation Std (°)")
ax.set_title("Global Light Direction Dispersion by Painting Type")
plt.tight_layout()
plt.show()


# INSTEAD OF USING ARBITRARY PAINTING TYPES, WE WILL COMPARE PAINTINGS BY NUMBER OF PEOPLE. 

# STD analysis of global std: 1 person vs all others
def split_global_std_by_num_people(cleaned_data):
    model_ids, other_ids = [], []
    model_stds, other_stds = [], []
    total = 0
    for p in cleaned_data.get("paintings", []):
        analysis = p.get("analysis", {})
        if not analysis.get("contours"):
            continue
        # count this painting
        total += 1
        pid        = p["painting_id"]
        num_people = analysis.get("num_people", 0)
        std_deg    = analysis.get("global_estimation_std_degrees", None)
        # skip if no global std
        if std_deg is None:
            continue
        if num_people == 1:
            model_ids.append(pid)
            model_stds.append(std_deg)
        else:
            other_ids.append(pid)
            other_stds.append(std_deg)
    return (model_ids, model_stds), (other_ids, other_stds), total

(model_ids, model_stds), (other_ids, other_stds), total = split_global_std_by_num_people(cleaned_data)

print(f"Total analysed paintings: {total}")         # should be 68
print(f"Model paintings (n={len(model_ids)}):", model_ids)
print(f"Other paintings (n={len(other_ids)}):", other_ids)
model_arr = np.array(model_stds)
other_arr = np.array(other_stds)
# 1) Quantitative analysis
print("Descriptive Statistics")
print("----------------------")
print(f"Model group (n={model_arr.size}): μ={model_arr.mean():.2f}°, σ={model_arr.std(ddof=1):.2f}°")
print(f"Other group (n={other_arr.size}): μ={other_arr.mean():.2f}°, σ={other_arr.std(ddof=1):.2f}°\n")

fig, ax = plt.subplots(figsize=(6, 5))
# Boxplot with means
bp = ax.boxplot([model_arr, other_arr],
                labels=['Model (1 person)', 'Other (>1 person)'],
                showmeans=True,
                meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'})

# Overlay individual points with jitter
for i, arr in enumerate([model_arr, other_arr], start=1):
    x = np.random.normal(i, 0.05, size=arr.size)
    ax.scatter(x, arr, alpha=0.4, s=20)

ax.set_ylabel("Global Estimation Std (degrees)")
ax.set_title("Comparison of Global Std by NumPeople Category")
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# STD analysis of emprical_std_deg, for only person contour type: 1 person vs all others

def get_person_contour_std_by_group(cleaned_data):
    model_stds = []
    other_stds = []
    for painting in cleaned_data.get("paintings", []):
        analysis = painting.get("analysis", {})
        if not analysis.get("contours"):
            continue
        num_people = analysis.get("num_people", 0)
        target_list = model_stds if num_people == 1 else other_stds
        for contour in analysis.get("contours", []):
            ct = contour.get("contour_type", "")
            # if normalize_contour_type(ct) != "person":
            if ct != "person":
                continue
            std_deg = contour.get("empirical_std_deg", None)
            if std_deg is not None:
                # flatten if nested
                if isinstance(std_deg, (list, tuple)) and len(std_deg) == 1:
                    std_deg = std_deg[0]
                target_list.append(std_deg)
    return model_stds, other_stds

model_stds, other_stds = get_person_contour_std_by_group(cleaned_data)
model_arr = np.array(model_stds)
other_arr = np.array(other_stds)
print("Person-contour std (1-person paintings):", f"n={model_arr.size}, μ={model_arr.mean():.2f}°, σ={model_arr.std(ddof=1):.2f}°")
print("Person-contour std (multi-person paintings):", f"n={other_arr.size}, μ={other_arr.mean():.2f}°, σ={other_arr.std(ddof=1):.2f}°")

fig, ax = plt.subplots(figsize=(6,4))
parts = ax.violinplot([model_arr, other_arr],
                      positions=[1,2],
                      showmeans=True, showextrema=True)
for pos, arr in zip([1,2], [model_arr, other_arr]):
    x = np.random.normal(pos, 0.05, size=arr.size)
    ax.scatter(x, arr, alpha=0.5, s=10)

ax.set_xticks([1,2])
ax.set_xticklabels(['Person contours\n(1-person)', 'Person contours\n(multi-person)'])
ax.set_ylabel("Empirical Std (deg)")
ax.set_title("Std of Person-Contours by Painting Group")
plt.tight_layout()
plt.show()


# 3. Painting type in TIME - YEAR
years = []
types = []
for painting in cleaned_data["paintings"]:
    year = clean_date(painting.get("date", ""))
    ptype = painting.get("type", "")
    ptype = normalize_painting_type(ptype) # to move still life to genre
    if year is not None and ptype:
        years.append(year)
        types.append(ptype)


unique_types = sorted(list(set(types)))
type_to_y = {t: i for i, t in enumerate(unique_types)}
y_values = [type_to_y[t] for t in types]
plt.figure(figsize=(10,6))
plt.scatter(years, y_values, color="blue", alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Painting Type")
plt.yticks(list(type_to_y.values()), list(type_to_y.keys()))
plt.title("Painting Types Over Time")
plt.tight_layout()
plt.show()

# count plot of "num_people" variable
num_people_list = []
for painting in data["paintings"]:
    analysis = painting.get("analysis", {})
    num_people = analysis.get("num_people")
    if num_people is not None:
        num_people_list.append(num_people)

counts = Counter(num_people_list)
x_vals = list(counts.keys())
y_vals = list(counts.values())
plt.figure(figsize=(8,6))
plt.bar(x_vals, y_vals, color="skyblue", edgecolor="black")
plt.xlabel("Number of People")
plt.ylabel("Count of Paintings")
plt.title("Count Plot of 'num_people' in Paintings")
plt.xticks(x_vals)
plt.tight_layout()
plt.show()

# Number of people per painting in time
years = []
num_people_list = []
for painting in data["paintings"]:
    year = clean_date(painting.get("date", ""))
    analysis = painting.get("analysis", {})
    num_people = analysis.get("num_people")
    if year is not None and num_people is not None:
        years.append(year)
        num_people_list.append(num_people)

years = np.array(years)
num_people_list = np.array(num_people_list)
plt.figure(figsize=(10,6))
plt.scatter(years, num_people_list, color="blue", s=50, alpha=0.7, label="Number of People")
if len(years) > 1:
    coeffs = np.polyfit(years, num_people_list, 1)
    trend_line = np.poly1d(coeffs)
    years_fit = np.linspace(years.min(), years.max(), 100)
    plt.plot(years_fit, trend_line(years_fit), color="red", linewidth=2, label="Trend")

plt.xlabel("Year")
plt.ylabel("Number of People")
plt.title("Number of People in Paintings Over Time")
plt.legend()
plt.tight_layout()
plt.show()


# Canvas size (in m2) in time. 
def extract_canvas_size(technique):
    pattern = r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*cm'
    match = re.search(pattern, technique.lower())
    if match:
        height = float(match.group(1))
        width = float(match.group(2))
        return height, width
    return None, None

canvas_data = []
for painting in data["paintings"]:
    technique = painting.get("technique", "")
    height, width = extract_canvas_size(technique)
    year = clean_date(painting.get("date", ""))
    if width is not None and year is not None:
        canvas_data.append((year, height, width))

print("Total paintings with canvas size info:", len(canvas_data))
unique_sizes = set((h, w) for _, h, w in canvas_data)
print("Unique canvas sizes (height x width in cm):")
for s in sorted(unique_sizes):
    print(f"{s[0]} x {s[1]}")

canvas_data = np.array(canvas_data)
years = canvas_data[:, 0]
heights = canvas_data[:, 1]
widths = canvas_data[:, 2]
areas = heights * widths 
plt.figure(figsize=(10,6))
plt.scatter(years, areas, color="blue", alpha=0.7)
coeffs = np.polyfit(years, areas, 1)
poly_fit = np.poly1d(coeffs)
year_fit = np.linspace(years.min(), years.max(), 100)
plt.plot(year_fit, poly_fit(year_fit), color="red", linewidth=2, label="Polynomial Fit")
plt.xlabel("Year")
plt.ylabel("Canvas Area (cm²)")
plt.title("Canvas Area Over Time")
plt.legend()
plt.tight_layout()
plt.show()


def extract_canvas_data(data):
    canvas_data = []
    for painting in data["paintings"]:
        technique = painting.get("technique", "")
        width, height = extract_canvas_size(technique)
        year = clean_date(painting.get("date", ""))
        if width is not None and year is not None:
            canvas_data.append((year, width, height))
    return canvas_data

def categorize_area(area, q1, q2, q3):
    if area < q1:
        return "small"
    elif area < q2:
        return "medium"
    elif area < q3:
        return "big"
    else:
        return "large"

# same but with categories and colors by quantiles. 
from matplotlib.lines import Line2D
canvas_data = extract_canvas_data(data)
canvas_data = np.array(canvas_data)
years = canvas_data[:,0]
widths = canvas_data[:,1]
heights = canvas_data[:,2]
areas = widths * heights
q1, q2, q3 = np.percentile(areas, [15,60,85])
category_mapping = {"small": "blue", "medium": "green", "big": "orange", "large": "red"}
categories = [categorize_area(a, q1, q2, q3) for a in areas]
point_colors = [category_mapping[cat] for cat in categories]
plt.figure(figsize=(10,6))
plt.scatter(years, areas, c=point_colors, alpha=0.7, s=50)
coeffs = np.polyfit(years, areas, 1)
linear_fit = np.poly1d(coeffs)
year_fit = np.linspace(years.min(), years.max(), 100)
plt.plot(year_fit, linear_fit(year_fit), color="black", linewidth=2, label="Linear Regression")
plt.xlabel("Year")
plt.ylabel("Canvas Area (cm²)")
plt.title("Canvas Area Over Time (Colored by Size Category)")
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Small (<{:.1f} cm²)'.format(q1), markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Medium ({:.1f}-{:.1f} cm²)'.format(q1, q2), markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Big ({:.1f}-{:.1f} cm²)'.format(q2, q3), markerfacecolor='orange', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Large (>{:.1f} cm²)'.format(q3), markerfacecolor='red', markersize=10)
]
plt.legend(handles=legend_elements, title="Canvas Size Categories", loc="best")
plt.tight_layout()
plt.show()



##############################################################################
# Locate Caravaggio's oeuvre in a MAP.
import time
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

def get_painting_locations(data):
    locations = []
    geolocator = Nominatim(user_agent="caravaggio_map")
    for painting in data["paintings"]:
        loc_str = painting.get("location", "").strip()
        if loc_str:
            try:
                location = geolocator.geocode(loc_str)
                if location:
                    locations.append({
                        "title": painting.get("title", ""),
                        "location_str": loc_str,
                        "latitude": location.latitude,
                        "longitude": location.longitude
                    })
                else:
                    print(f"Geocoding failed for: {loc_str}")
            except Exception as e:
                print(f"Error geocoding {loc_str}: {e}")
            time.sleep(1)  # pause to respect Nominatim usage policy
    return locations

painting_locations = get_painting_locations(data)
if painting_locations:
    avg_lat = sum(loc["latitude"] for loc in painting_locations) / len(painting_locations)
    avg_lon = sum(loc["longitude"] for loc in painting_locations) / len(painting_locations)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6)
    for loc in painting_locations:
        folium.Marker(
            location=[loc["latitude"], loc["longitude"]],
            popup=f"{loc['title']} ({loc['location_str']})",
            tooltip=loc["location_str"]
        ).add_to(m)
    m.save("caravaggio_map.html")
    print("Map saved to caravaggio_map.html")
else:
    print("No valid painting locations found.")


def get_all_location_strings(data):
    # Returns a list of tuples (painting_id, location_str)
    all_locations = []
    for painting in data["paintings"]:
        loc_str = painting.get("location", "").strip()
        if loc_str:
            all_locations.append((painting.get("painting_id"), loc_str))
    return all_locations

def geocode_location(loc_str, geolocator, timeout=2):
    try:
        location = geolocator.geocode(loc_str, timeout=timeout)
        return location
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        return None
    
def simplify_location(loc_str):
    parts = [p.strip() for p in loc_str.split(",")]
    if len(parts) >= 2:
        return ", ".join(parts[-2:])
    elif parts:
        return parts[0]
    else:
        return loc_str

def get_successful_locations(data):
    geolocator = Nominatim(user_agent="caravaggio_map", timeout=2)
    painting_locations = []
    failed_locations = []
    for painting in data["paintings"]:
        loc_str = painting.get("location", "").strip()
        if not loc_str:
            continue
        location = geocode_location(loc_str, geolocator, timeout=2)
        if location:
            painting_locations.append({
                "painting_id": painting.get("painting_id"),
                "title": painting.get("title", ""),
                "location_str": loc_str,
                "latitude": location.latitude,
                "longitude": location.longitude
            })
        else:
            print(f"Geocoding failed for: {loc_str}")
            failed_locations.append((painting.get("painting_id"), loc_str, painting.get("title", "")))
        time.sleep(2)
    return painting_locations, failed_locations

def retry_failed_locations(failed_locations):
    geolocator = Nominatim(user_agent="caravaggio_map_retry", timeout=5)
    new_successes = []
    still_failed = []
    for pid, loc_str, title in failed_locations:
        simplified = simplify_location(loc_str)
        location = geocode_location(simplified, geolocator, timeout=5)
        if location:
            print(f"Retry succeeded for: {loc_str} -> {simplified}")
            new_successes.append({
                "painting_id": pid,
                "title": title,
                "location_str": simplified,
                "latitude": location.latitude,
                "longitude": location.longitude
            })
        else:
            print(f"Retry failed for: {loc_str} -> {simplified}")
            still_failed.append((pid, loc_str, title))
        time.sleep(1)
    return new_successes, still_failed


painting_locations, failed_locations = get_successful_locations(data)
print(f"\nInitial successful geocoding: {len(painting_locations)}")
print(f"Initial failed geocoding: {len(failed_locations)}")
new_successes, still_failed = retry_failed_locations(failed_locations)
print(f"\nNew successes on retry: {len(new_successes)}")
print(f"Remaining failed locations: {len(still_failed)}")
all_locations = painting_locations + new_successes

if all_locations:
    avg_lat = sum(loc["latitude"] for loc in all_locations) / len(all_locations)
    avg_lon = sum(loc["longitude"] for loc in all_locations) / len(all_locations)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6)
    for loc in all_locations:
        folium.Marker(
            location=[loc["latitude"], loc["longitude"]],
            popup=f"{loc['title']} ({loc['location_str']})",
            tooltip=loc["location_str"]
        ).add_to(m)
    m.save("caravaggio_map_updated.html")
    print("Map saved to caravaggio_map_updated.html")
else:
    print("No valid painting locations found.")

##############################################################################



# 3. Global STD in time, 
# distribution of global STD
std_values = []
for painting in cleaned_data["paintings"]:
    analysis = painting.get("analysis", {})
    std_deg = analysis.get("global_estimation_std_degrees")
    if std_deg is not None:
        std_values.append(std_deg)
  
std_values = np.array(std_values)
plt.figure(figsize=(10,6))
plt.hist(std_values, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
plt.xlabel("Global Estimation Std (°)")
plt.ylabel("Frequency")
plt.title("Distribution of Global Light Direction Standard Deviations")
plt.tight_layout()
plt.show()

# Global STD in TIME 
years = []
std_values = []
for painting in cleaned_data["paintings"]:
    analysis = painting.get("analysis", {})
    std_deg = analysis.get("global_estimation_std_degrees")
    year = clean_date(painting.get("date", ""))
    if std_deg is not None and year is not None:
        years.append(year)
        std_values.append(std_deg)


plt.rc("text", usetex=True)
plt.rc("font", family="serif")

years = np.array(years)
std_values = np.array(std_values)
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(years, std_values,
           color="blue", s=50, alpha=0.7,
           label=r"\textbf{Std per painitng}")
if len(years) > 1:
    coeffs = np.polyfit(years, std_values, 1)
    linear_fit = np.poly1d(coeffs)
    xs = np.linspace(years.min(), years.max(), 200)
    ax.plot(xs, linear_fit(xs),
            color="red", linewidth=2,
            label=r"\textbf{Linear Trend}")

years_int = np.unique(years.astype(int))
ax.set_xticks(years_int)
ax.set_xticklabels([str(y) for y in years_int], fontsize=12)
# === labels, title, legend with custom font sizes ===
ax.set_xlabel(r"\textbf{Year}", fontsize=14)
ax.set_ylabel(r"\textbf{Std of Light Direction (°)}", fontsize=14)
ax.set_title(r"\textbf{Global Light Direction Std Over Time}", fontsize=16, pad=12)
ax.legend(fontsize=12, frameon=False)
# tight layout
plt.tight_layout()
plt.show()

# SAVE THE FIG!
fig.savefig("global_std_over_time.pdf",format="pdf",dpi=300,bbox_inches="tight")


# IS THIS TREND STATISTICALLY SIGNIFICANT?
from scipy.stats import linregress, pearsonr, spearmanr

pearson_r, pearson_p = pearsonr(years, std_values)
print("Pearson correlation:")
print(f"  r       = {pearson_r:.3f}")
print(f"  p‐value = {pearson_p:.3f}\n")

spearman_r, spearman_p = spearmanr(years, std_values)
print("Spearman rank correlation:")
print(f"  ρ       = {spearman_r:.3f}")
print(f"  p‐value = {spearman_p:.3f}")


# # of extracted contours in TIME

years = []
contour_counts = []
for p in cleaned_data.get("paintings", []):
    analysis = p.get("analysis", {})
    contours = analysis.get("contours", [])
    if not contours:
        continue  # skip un-analyzed
    year = clean_date(p.get("date", ""))
    if year is None:
        continue  # skip if date unparseable
    years.append(year)
    contour_counts.append(len(contours))

# Convert to numpy arrays
years_arr = np.array(years)
counts_arr = np.array(contour_counts)

# Fit linear trend
slope, intercept = np.polyfit(years_arr, counts_arr, 1)

# Plot  contour counts with scatter plot over time
plt.figure(figsize=(8, 5))
plt.scatter(years_arr, counts_arr, alpha=0.7, label="Paintings")
plt.plot(years_arr, slope*years_arr + intercept, '--',
         label=f"Trend: slope={slope:.2f} contours/year")
plt.xlabel("Year")
plt.ylabel("Number of Contours")
plt.title("Contour Count per Analyzed Painting Over Time")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# COUNT PLOT OF # OF CONTOUR SEGMENTS PER YEAR
segments_per_year = defaultdict(int)

for p in cleaned_data.get("paintings", []):
    analysis = p.get("analysis", {})
    contours = analysis.get("contours", [])
    if not contours:
        continue
    year = clean_date(p.get("date", ""))
    if year is None:
        continue
    # Sum segments in this painting
    total_segments = sum(len(c.get("contour_coordinates", [])) for c in contours)
    segments_per_year[year] += total_segments

# 2) Prepare data for a count plot
years = sorted(segments_per_year.keys())
totals = [segments_per_year[y] for y in years]
# 3) Plot bar chart
plt.figure(figsize=(8,5))
plt.bar(years, totals, width=0.8, edgecolor='black')
plt.xlabel("Year")
plt.ylabel("Total Contour Segments")
plt.title("Total Contour Segments Extracted per Year")
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()

# COMPARE BETWEEN EARLY AND LATE STAGES IN TERMS OF CONTOUR SEGMENT COUNT
early_counts = []
late_counts = []

for p in cleaned_data.get("paintings", []):
    analysis = p.get("analysis", {})
    contours = analysis.get("contours", [])
    if not contours:
        continue  # skip un‐analysed
    year = clean_date(p.get("date", ""))
    if year is None:
        continue
    # count segments in this painting
    segs = sum(len(c.get("contour_coordinates", [])) for c in contours)
    if 1592 <= year < 1600:
        early_counts.append(segs)
    elif 1600 <= year <= 1610:
        late_counts.append(segs)

# 3) Compute statistics
n_early = len(early_counts)
n_late  = len(late_counts)
mean_early = np.mean(early_counts) if n_early else float('nan')
mean_late  = np.mean(late_counts)  if n_late  else float('nan')
pct_increase = 100 * (mean_late - mean_early) / mean_early

# 4) Print results
print(f"Early period (1592–1599): n={n_early} paintings, "
      f"mean={mean_early:.1f} contour segments")
print(f"Late period (1600–1610):  n={n_late} paintings, "
      f"mean={mean_late:.1f} contour segments")
print(f"→ {pct_increase:+.0f}% change from early to late")

# 4. # of people vs global STD. 
num_people = []
global_std = []
for painting in cleaned_data["paintings"]:
    analysis = painting.get("analysis", {})
    n_people = analysis.get("num_people")
    std_deg = analysis.get("global_estimation_std_degrees")
    if n_people is not None and std_deg is not None:
        num_people.append(n_people)
        global_std.append(std_deg)

num_people = np.array(num_people)
global_std = np.array(global_std)
plt.figure(figsize=(10,6))
plt.scatter(num_people, global_std, color="blue", alpha=0.7, s=50)
plt.xlabel("Number of People")
plt.ylabel("Global Estimation Std (°)")
plt.title("Number of People vs Global Light Direction Standard Deviation")
plt.tight_layout()
plt.show()


## Find common pattern in contours with low std. What are the characteristics that define a low std contour?
# NORMAL SPAN VS EMPIRICAL STD
stds = []
spans = []
for p in cleaned_data.get("paintings", []):
    for c in p.get("analysis", {}).get("contours", []):
        sd = c.get("empirical_std_deg", None)
        span = c.get("normal_vectors_span", None)
        if sd is None or span is None:
            continue
        # unwrap single‐value lists if needed
        if isinstance(sd, (list,tuple)) and len(sd)==1:
            sd = sd[0]
        if isinstance(span, (list,tuple)) and len(span)==1:
            span = span[0]
        stds.append(sd)
        spans.append(span)


stds = np.array(stds)
spans = np.array(spans)
print(f"Collected {len(stds)} contours with both std and span.")

# 2) Overall correlation
pear_r, pear_p = pearsonr(spans, stds)
spear_r, spear_p = spearmanr(spans, stds)
print(f"Pearson r = {pear_r:.3f}, p = {pear_p:.3f}")
print(f"Spearman ρ = {spear_r:.3f}, p = {spear_p:.3f}")

plt.figure(figsize=(6,4))
plt.scatter(spans, stds, alpha=0.4)
plt.xlabel("normal_vectors_span (°)")
plt.ylabel("empirical_std_deg (°)")
plt.title("Contour‐level std vs. normal_vectors_span")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# 
angles = []
stds = []

for p in cleaned_data.get("paintings", []):
    for c in p.get("analysis", {}).get("contours", []):
        # get cleaned light direction [dx, dy]
        ld = c.get("light_direction")
        if not ld or len(ld) != 2:
            continue
        dx, dy = ld
        # convert to angle in degrees [0, 360)
        ang = np.degrees(np.arctan2(-dy, dx))
        if ang < 0:
            ang += 360
        angles.append(ang)
        # get empirical std_deg (unwrap single-element lists)
        sd = c.get("empirical_std_deg")
        if isinstance(sd, (list, tuple)) and len(sd) == 1:
            sd = sd[0]
        stds.append(sd)

angles = np.array(angles)
stds = np.array(stds)

# 2) Scatter plot: angle vs std
plt.figure(figsize=(7,5))
plt.scatter(angles, stds, alpha=0.4, s=20)
plt.xlabel("Estimated Light Direction (°)")
plt.ylabel("Empirical Std (°)")
plt.title("Contour-level Std vs. Estimated Direction")
plt.grid(True, linestyle='--', alpha=0.3)
# 3) Binned averages
bins = np.arange(0, 361, 30)   # 12 bins of 30°
bin_indices = np.digitize(angles, bins)
bin_centers = bins[:-1] + 15
mean_stds = [stds[bin_indices == i].mean() if np.any(bin_indices == i) else np.nan
             for i in range(1, len(bins))]
# overlay binned means
plt.plot(bin_centers, mean_stds, '-o', linestyle='--')
plt.tight_layout()
plt.show()

##############################################################################
# 8. Spherical harmonic analysis. 
##############################################################################

##############################################################################
# 9. Checking hypothesis: nearby areas are more likely to have similar light directions. Check lD congruence by distance of contours. 
##############################################################################
# less informative now that we use contour segments instead of contours for example for people. 

def centroid_from_coords(coords):
    pts = []
    for seg in coords:
        if isinstance(seg[0], (list, tuple)):
            pts.extend(seg)
        else:
            pts.append(seg)
    arr = np.array(pts, dtype=float)
    return arr.mean(axis=0)

total_paintings = 0
count_matches = 0

for painting in cleaned_data["paintings"]:
    contours = painting.get("analysis", {}).get("contours", [])
    if len(contours) < 2:
        continue
    total_paintings += 1
    centroids = []
    thetas = []
    for c in contours:
        centroids.append(centroid_from_coords(c["contour_coordinates"]))
        dx, dy = c["light_direction"]
        theta = np.degrees(np.arctan2(dy, dx)) % 360
        thetas.append(theta)
    centroids = np.array(centroids)
    thetas = np.array(thetas)
    # compute all pairwise distances and angle differences
    n = len(centroids)
    min_dist = np.inf
    min_dist_idx = None
    min_ang_diff = np.inf
    min_ang_idx = None
    for i in range(n):
        for j in range(i+1, n):
            # distance
            d = np.linalg.norm(centroids[i] - centroids[j])
            # angular difference
            diff = abs(thetas[i] - thetas[j])
            if diff > 180:
                diff = 360 - diff
            if d < min_dist:
                min_dist = d
                min_dist_idx = (i, j)
            if diff < min_ang_diff:
                min_ang_diff = diff
                min_ang_idx = (i, j)
    # check if the closest pair is also the most congruent
    if min_dist_idx == min_ang_idx:
        count_matches += 1

print(f"Total paintings considered: {total_paintings}")
print(f"Paintings where closest contours have lowest angle difference: {count_matches}")
print(f"Proportion: {count_matches/total_paintings:.2%}")

##############################################################################
# 10. When Ld evidence is available, compare it to the global estimation (and check std). is there any similar contour with low std? 
##############################################################################

# get rear wall light dir evidence paintings
wall_evidence = []

for p in data.get("paintings", []):
    # guard: skip anything that isn’t a dict
    if not isinstance(p, dict):
        continue
    pid   = p.get("painting_id")
    title = p.get("title", "<no title>")
    analysis = p.get("analysis")
    # guard: analysis might be missing or not a dict
    if not isinstance(analysis, dict):
        continue
    evidence = analysis.get("depicted_light_direction_evidence", [])
    # guard: evidence should be a list
    if not isinstance(evidence, list):
        continue
    for ev in evidence:
        # guard: each entry should be a dict
        if not isinstance(ev, dict):
            continue
        # match “cast shadow” in type and “wall” anywhere in notes
        if ev.get("type","").strip().lower() == "cast shadow" \
           and "wall" in ev.get("notes","").lower():
            wall_evidence.append({
                "painting_id": pid,
                "title": title,
                "light_direction": ev.get("light_direction"),
                "notes": ev.get("notes")
            })


gl_dict = {
    p["painting_id"]: p["analysis"].get("global_light_direction")
    for p in cleaned_data["paintings"]
    if p.get("analysis", {}).get("global_light_direction") is not None
}


# Compare each wall‐evidence entry with global light dir estimation. 
diffs = []
for ev in wall_evidence:
    pid = ev["painting_id"]
    title = ev["title"]
    # evidence vector
    dx_e, dy_e = ev["light_direction"]
    # evidence angle in [0,2π)
    θ_e = np.arctan2(-dy_e, dx_e)
    if θ_e < 0: θ_e += 2*np.pi
    # global vector
    gl = gl_dict.get(pid)
    if gl is None:
        continue
    dx_g, dy_g = gl
    θ_g = np.arctan2(-dy_g, dx_g)
    if θ_g < 0: θ_g += 2*np.pi
    # minimal angular difference
    δ = abs(θ_e - θ_g)
    if δ > np.pi:
        δ = 2*np.pi - δ
    δ_deg = np.degrees(δ)
    diffs.append({"painting_id": pid, "title": title, "Δ_deg": δ_deg})

for d in diffs:
    print(f"{d['painting_id']:>3d} “{d['title']}”  {d['Δ_deg']:6.1f}°")

delta_vals = np.array([d["Δ_deg"] for d in diffs])
print(f"Contours within ±10°: {np.sum(delta_vals <= 10)} / {len(delta_vals)}")


# try to do the same experiment but only taking into account person contours.
# 1) Compute each painting’s mean “person” direction
person_mean_angles = {}
for p in cleaned_data["paintings"]:
    pid = p["painting_id"]
    angles = []
    for c in p.get("analysis", {}).get("contours", []):
        if normalize_contour_type(c.get("contour_type", "")) == "person":
            dx, dy = c.get("light_direction", (None, None))
            if dx is None or dy is None:
                continue
            θ = np.arctan2(-dy, dx)
            if θ < 0: θ += 2*np.pi
            angles.append(θ)
    if angles:
        mean_sin = np.mean(np.sin(angles))
        mean_cos = np.mean(np.cos(angles))
        mean_angle = np.arctan2(mean_sin, mean_cos)
        if mean_angle < 0: mean_angle += 2*np.pi
        person_mean_angles[pid] = mean_angle
        
# 2) Compare “wall” cast‐shadow evidence to person‐based mean
results = []
for ev in wall_evidence:
    pid = ev["painting_id"]
    mean_angle = person_mean_angles.get(pid)
    if mean_angle is None:
        continue  # no person contours
    dx_e, dy_e = ev["light_direction"]
    θ_e = np.arctan2(-dy_e, dx_e)
    if θ_e < 0: θ_e += 2*np.pi
    δ = abs(θ_e - mean_angle)
    if δ > np.pi:
        δ = 2*np.pi - δ
    δ_deg = np.degrees(δ)
    results.append({"painting_id": pid, "title": ev["title"], "Δ_deg": δ_deg})

1
deltas = np.array([r["Δ_deg"] for r in results])
print(f"{np.sum(deltas <= 10)} / {len(deltas)} within ±10°")

# putting this into context. 
from itertools import combinations
baseline_diffs = []
for p in cleaned_data["paintings"]:
    dirs = []
    for c in p["analysis"]["contours"]:
        ld = c.get("light_direction")
        if not ld: 
            continue
        dx, dy = ld
        θ = np.degrees(np.arctan2(-dy, dx)) % 360
        dirs.append(θ)
    # all pairs within this painting
    for θ1, θ2 in combinations(dirs, 2):
        diff = abs(θ1 - θ2)
        baseline_diffs.append(min(diff, 360 - diff))

baseline_diffs = np.array(baseline_diffs)

p10 = np.percentile(baseline_diffs, 10)
p25 = np.percentile(baseline_diffs, 25)
p50 = np.percentile(baseline_diffs, 50)

print(f"Baseline contour‐pair Δ’s: 10th %ile={p10:.1f}°, 25th %ile={p25:.1f}°, median={p50:.1f}°")

# 3) See where your wall‐evidence Δ_deg lie relative to that
wall_deltas = np.array([ r["Δ_deg"] for r in results ])  # from your last step
for thr, label in [(p10,"10th %ile"), (p25,"25th %ile"), (p50,"median")]:
    count = np.sum(wall_deltas <= thr)
    print(f"{count}/10 wall‐evidence Δ’s ≤ {label} of baseline")

# 4) If you want a quick table of percentiles of your wall‐Δ’s themselves:
print("Your wall‐evidence Δ’s percentiles:",
      np.percentile(wall_deltas, [10,25,50,75,90]).round(1))

##############################################################################
# 11. When is the light congruent? close to theoretical lighting? aka, when is the mse low? when is the shading rate low?
##############################################################################



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


