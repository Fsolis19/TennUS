import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATS_DIR = os.path.join(BASE_DIR, 'statistics')

merged_candidates = sorted(
    glob.glob(os.path.join(STATS_DIR, 'movement_point_*_merged.json')),
    key=os.path.getmtime,
    reverse=True
)

if not merged_candidates:
    raise FileNotFoundError("No se encontró ningún archivo movement_point_*_merged.json")

MERGED_STATS_FILE = merged_candidates[0]

with open(MERGED_STATS_FILE, 'r') as file:
    merged_stats = json.load(file)

metrics_labels = [
    "Distancia (m)",
    "Vel. media (m/s)",
    "Vel. máxima (m/s)",
    "Tiempo en cuadro de individuales (s)",
    "Tiempo en cuadros de dobles (s)",
    "Tiempo en cuadros de saque (s)",
    "Tiempo fuera de zonas (s)"
]

players_data = []
for player_name, stats in merged_stats["players"].items():
    row = {
        "Jugador": player_name,
        "Distancia (m)": stats.get("total_distance_meters", 0.0),
        "Vel. media (m/s)": stats.get("average_speed_meters_per_sec", 0.0),
        "Vel. máxima (m/s)": stats.get("max_speed_meters_per_sec", 0.0),
        "Tiempo en cuadro de individuales (s)": stats.get("zones", {}).get("in-singles", {}).get("time_seconds", 0.0),
        "Tiempo en cuadros de dobles (s)": stats.get("zones", {}).get("in-doubles", {}).get("time_seconds", 0.0),
        "Tiempo en cuadros de saque (s)": stats.get("zones", {}).get("in-serve", {}).get("time_seconds", 0.0),
        "Tiempo fuera de zonas (s)": stats.get("zones", {}).get("outside_zones", {}).get("time_seconds", 0.0),
    }
    players_data.append(row)

def draw_radar(player_row, labels):
    categories = labels
    data = [player_row[label] for label in categories]

    max_values = {
        "Distancia (m)": 300,
        "Vel. media (m/s)": 7,
        "Vel. máxima (m/s)": 7,
        "Tiempo en cuadro de individuales (s)": 80,
        "Tiempo en cuadros de dobles (s)": 45,
        "Tiempo en cuadros de saque (s)": 30,
        "Tiempo fuera de zonas (s)": 120
    }

    normalized = [
        min(data[i], max_values[cat]) / max_values[cat] if max_values[cat] != 0 else 0
        for i, cat in enumerate(categories)
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    normalized += normalized[:1]
    data += data[:1]
    labels += labels[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, normalized, label=player_row["Jugador"], color="#1f77b4")
    ax.fill(angles, normalized, alpha=0.2, color="#1f77b4")

    for angle, real_val, cat in zip(angles, data, labels):
        max_val = max_values.get(cat, 1.0)
        if real_val > max_val:
            display_norm = 0.75 
        else:
            display_norm = real_val / max_val if max_val != 0 else 0

        display_val = min(real_val, max_val)
        ax.text(angle, display_norm + 0.05, f"{real_val:.2f}", ha='center', va='center', fontsize=8)


    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Estadísticas de movilidad - {player_row['Jugador']}", pad=20)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    return fig

for row in players_data:
    fig = draw_radar(row.copy(), metrics_labels.copy())
    output_path = f'./app/backend/images/stats_{row["Jugador"]}.png'
    fig.savefig(output_path)
    plt.close(fig)