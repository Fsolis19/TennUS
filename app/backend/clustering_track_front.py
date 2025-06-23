import os
import json
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from sklearn.cluster import DBSCAN

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATS_FILES_DIR = os.path.join(BASE_DIR, 'statistics')

def plot_projected_court_heatmap(point_index):
    detections_log_path = os.path.join(STATS_FILES_DIR, f"detections_log_point_{point_index}.json")
    track_mapping_path = os.path.join(STATS_FILES_DIR, f"track_mapping_point_{point_index}.json")
    merged_stats_path = os.path.join(STATS_FILES_DIR, f"movement_point_{point_index}_merged.json")

    detections_log = json.load(open(detections_log_path))
    track_id_to_player_mapping = json.load(open(track_mapping_path))
    merged_stats = json.load(open(merged_stats_path))

    src_points = np.array(merged_stats["src_pts"], dtype=np.float32)
    dst_points = np.array(merged_stats["dst_pts"], dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)

    projected_points_per_player = defaultdict(list)
    for detection in detections_log:
        player_id = track_id_to_player_mapping.get(str(detection["track_id"]))
        if player_id is None:
            continue

        image_point = np.array([[[detection["cx"], detection["cy"]]]], dtype=np.float32)
        projected_point = cv2.perspectiveTransform(image_point, homography_matrix)[0][0]

        projected_point[1] = 23.77 - projected_point[1]

        projected_points_per_player[player_id].append(projected_point)

    filtered_points_per_player = {}
    for player_id, points in projected_points_per_player.items():
        points_array = np.array(points)
        num_points = len(points_array)
        min_samples = max(5, num_points // 10)

        if num_points < min_samples:
            filtered_points_per_player[player_id] = points_array
            continue

        clustering = DBSCAN(eps=10.0, min_samples=min_samples).fit(points_array)
        labels = clustering.labels_

        valid_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(valid_labels) == 0:
            filtered_points_per_player[player_id] = points_array
        else:
            main_cluster_label = valid_labels[np.argmax(counts)]
            filtered_points_per_player[player_id] = points_array[labels == main_cluster_label]

    non_empty_x_coords = [filtered_points_per_player[pid][:, 0] for pid in filtered_points_per_player if filtered_points_per_player[pid].size > 0]
    if not non_empty_x_coords:
        print(f"No hay puntos válidos tras DBSCAN para point {point_index}")
        return

    court_width_m = 10.97
    court_length_m = 23.77
    margin_x = 4.0
    margin_y = 4.0

    fig, ax = plt.subplots(figsize=(6, 6 * court_length_m / court_width_m))
    ax.add_patch(Rectangle((0, 0), court_width_m, court_length_m, linewidth=2, fill=False, color="black"))

    singles_width_m = 8.23
    margin_inner = (court_width_m - singles_width_m) / 2
    ax.add_patch(Rectangle((margin_inner, 0), singles_width_m, court_length_m, linewidth=1, fill=False, color="black"))

    center_y = court_length_m / 2
    service_line_distance = 6.40
    ax.plot([0, court_width_m], [center_y, center_y], color="black", linewidth=1)
    ax.plot([margin_inner, court_width_m - margin_inner], [center_y + service_line_distance] * 2, color="black", linewidth=1)
    ax.plot([margin_inner, court_width_m - margin_inner], [center_y - service_line_distance] * 2, color="black", linewidth=1)

    center_x = court_width_m / 2
    ax.plot([center_x, center_x], [center_y - service_line_distance, center_y + service_line_distance], color="black", linewidth=1)

    player_colors = ["#1f77b4", "#ff7f0e"]
    for player_id, color in zip(sorted(filtered_points_per_player), player_colors):
        points_array = filtered_points_per_player[player_id]
        if points_array.size > 0:
            ax.scatter(points_array[:, 0], points_array[:, 1], s=5, alpha=0.5, c=color, label=f"Jugador {player_id}")

    ax.set_title(f"Heatmap proyectado de los jugadores")
    ax.set_xlabel("Ancho pista (m)")
    ax.set_ylabel("Largo pista (m)")
    ax.set_xlim(-margin_x, court_width_m + margin_x)
    ax.set_ylim(-margin_y, court_length_m + margin_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f'./app/backend/images/heatmap.png')
    plt.close()

if __name__ == '__main__':
    detections_log_files = sorted(glob.glob(os.path.join(STATS_FILES_DIR, "detections_log_point_*.json")))
    for file_path in detections_log_files:
        point_index = int(os.path.basename(file_path).split("_")[-1].split(".")[0])
        plot_projected_court_heatmap(point_index)
