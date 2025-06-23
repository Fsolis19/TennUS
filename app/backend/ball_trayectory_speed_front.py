import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.spatial.distance import euclidean

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECTED_BALL_JSON_PATH = os.path.join(BASE_DIR, 'statistics', 'ball_track_projected_with_speed.json')
FRAMES_PER_SECOND = 30.0

def load_projected_ball_data(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def draw_tennis_court(ax):
    court_width_m = 10.97
    court_length_m = 23.77
    singles_width_m = 8.23
    service_box_length_m = 6.40

    ax.set_facecolor('white')
    ax.add_patch(Rectangle((0, 0), court_width_m, court_length_m, linewidth=2, fill=False, color='black'))

    margin = (court_width_m - singles_width_m) / 2
    ax.add_patch(Rectangle((margin, 0), singles_width_m, court_length_m, linewidth=1, fill=False, color='black'))

    center_y = court_length_m / 2
    ax.plot([0, court_width_m], [center_y, center_y], color='black', linewidth=1)
    ax.plot([margin, court_width_m - margin], [center_y + service_box_length_m] * 2, color='black', linewidth=1)
    ax.plot([margin, court_width_m - margin], [center_y - service_box_length_m] * 2, color='black', linewidth=1)

    service_center_x = margin + singles_width_m / 2
    ax.plot([service_center_x] * 2, [center_y - service_box_length_m, center_y + service_box_length_m],
            color='black', linewidth=1)

    ax.set_xlim(0, court_width_m)
    ax.set_ylim(0, court_length_m)
    ax.set_aspect('equal')
    ax.set_xlabel('Ancho pista (m)')
    ax.set_ylabel('Largo pista (m)')
    ax.grid(False)

def plot_ball_trajectory_on_court(ball_data):
    coords = [(p['x_m'], p['y_m']) for p in ball_data if p.get('x_m') is not None and p.get('y_m') is not None]

    if not coords:
        print('No hay coordenadas válidas para generar el gráfico.')
        return

    x_vals, y_vals = zip(*coords)
    court_width_m = 10.97
    court_length_m = 23.77

    fig, ax = plt.subplots(figsize=(6, 6 * court_length_m / court_width_m))
    ax.set_facecolor('white')
    draw_tennis_court(ax)
    ax.scatter(x_vals, y_vals, s=10, c='red', alpha=0.2)

    ax.set_title('Heatmap trayectoria de la bola')
    plt.tight_layout()
    plt.savefig('./app/backend/images/ball_trajectory.png')
    plt.close()

def plot_ball_speed_graph(ball_data):
    coordinates = [
        (p['x_m'], p['y_m']) for p in ball_data
        if p.get('x_m') is not None and p.get('y_m') is not None
    ]

    distances_m = [
        euclidean(coordinates[i], coordinates[i - 1])
        for i in range(1, len(coordinates))
    ]

    speeds_m_per_s = np.array(distances_m) * FRAMES_PER_SECOND
    speeds_kmh = speeds_m_per_s * 3.6
    speeds_kmh = np.where(speeds_kmh > 220, np.nan, speeds_kmh)
    speeds_kmh_smooth = pd.Series(speeds_kmh).rolling(window=5, min_periods=1, center=True).mean()
    time_seconds = np.arange(1, len(coordinates)) / FRAMES_PER_SECOND
    average_speed_kmh = np.nanmean(speeds_kmh)

    model_speeds_kmh = []
    model_time_seconds = []
    for i, point in enumerate(ball_data):
        speed_mps = point.get('speed_mps', -1.0)
        if speed_mps is not None and speed_mps >= 0:
            model_speeds_kmh.append(speed_mps * 3.6)
            model_time_seconds.append(i / FRAMES_PER_SECOND)

    model_average_kmh = np.nanmean(model_speeds_kmh) if model_speeds_kmh else np.nan

    plt.figure(figsize=(10, 5))
    if model_speeds_kmh:
        plt.plot(model_time_seconds, model_speeds_kmh, linewidth=1.5, label='Evolución velocidad de la bola', color='green')

    if not np.isnan(model_average_kmh):
        plt.hlines(model_average_kmh, 0, max(model_time_seconds), colors='orange',
                   linestyles='--', label=f'Velocidad media: {model_average_kmh:.1f} km/h')

    plt.title('Velocidad captada de la bola (km/h)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad (km/h)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./app/backend/images/ball.png')
    plt.close()

if __name__ == '__main__':
    projected_ball_data = load_projected_ball_data(PROJECTED_BALL_JSON_PATH)
    plot_ball_trajectory_on_court(projected_ball_data)
    plot_ball_speed_graph(projected_ball_data)


