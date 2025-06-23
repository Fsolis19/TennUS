
#from model import BallTrackerNet
from .model import BallTrackerNet
import torch
import cv2
#from general import postprocess
from .general import postprocess
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance
import os
import json
import glob
from scipy.signal import savgol_filter

def infer_model_streaming(video_path, model, output_path, trace=7):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if not output_path:
        raise ValueError("La ruta de salida del vídeo está vacía.")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    buffer = []
    ball_track = [(None, None)] * 2
    dists = [-1] * 2
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        buffer.append(frame)
        if len(buffer) < 3:
            out.write(frame)
            frame_idx += 1
            continue
        if len(buffer) > 3:
            buffer.pop(0)

        img = cv2.resize(buffer[2], (640, 360))
        img_prev = cv2.resize(buffer[1], (640, 360))
        img_preprev = cv2.resize(buffer[0], (640, 360))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out_model = model(torch.from_numpy(inp).float().to(device))
        output = out_model.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)

        display_frame = frame.copy()
        for i in range(trace):
            if frame_idx - i > 0 and ball_track[frame_idx - i][0]:
                x, y = int(ball_track[frame_idx - i][0]), int(ball_track[frame_idx - i][1])
                display_frame = cv2.circle(display_frame, (x, y), radius=0, color=(0, 0, 255), thickness=10 - i)
        out.write(display_frame)
        frame_idx += 1

    cap.release()
    out.release()

    return ball_track, dists, fps

def remove_outliers(ball_track, dists, max_dist=100):
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers[:]:
        if (dists[i+1] > max_dist) or (dists[i+1] == -1):
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    return ball_track

def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]
    cursor, min_value = 0, 0
    result = []
    for i, (k, length) in enumerate(groups):
        if k == 1 and 0 < i < len(groups) - 1:
            dist_seg = distance.euclidean(ball_track[cursor-1], ball_track[cursor+length])
            if (length >= max_gap) or (dist_seg/length > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + length - 1
        cursor += length
    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])
    return result

def interpolation(coords):
    def nan_helper(arr):
        return np.isnan(arr), lambda z: z.nonzero()[0]

    x = np.array([p[0] if p[0] is not None else np.nan for p in coords])
    y = np.array([p[1] if p[1] is not None else np.nan for p in coords])
    nans_x, ix = nan_helper(x)
    x[nans_x] = np.interp(ix(nans_x), ix(~nans_x), x[~nans_x])
    nans_y, iy = nan_helper(y)
    y[nans_y] = np.interp(iy(nans_y), iy(~nans_y), y[~nans_y])
    return list(zip(x, y))


def generate_ball_statistics(ball_track, dists, fps):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    stats_dir = os.path.join(base_dir, 'tennis_statistics', 'ball_stats')
    os.makedirs(stats_dir, exist_ok=True)

    stats = []
    for i, (pt, dist) in enumerate(zip(ball_track, dists)):
        x, y = pt
        stats.append({
            'frame': i,
            'x_px': x if x is not None else None,
            'y_px': y if y is not None else None,
            'dist_px': dist
        })

    with open(os.path.join(stats_dir, 'ball_track.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    stats_files_dir = os.path.join(base_dir, 'tennis_statistics', 'stats_files')
    merged_candidates = sorted(
        glob.glob(os.path.join(stats_files_dir, 'movement_point_*_merged.json')),
        key=os.path.getmtime,
        reverse=True
    )

    if not merged_candidates:
        raise FileNotFoundError("No se encontró ningún archivo movement_point_*_merged.json")

    merged_path = merged_candidates[0]
    with open(merged_path) as f:
        merged = json.load(f)

    src_pts = np.array(merged.get("src_pts", []), dtype=np.float32)
    dst_pts = np.array(merged.get("dst_pts", []), dtype=np.float32)

    if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
        raise ValueError(f"src_pts o dst_pts inválidos en {merged_path}: se requieren al menos 4 puntos")

    H, _ = cv2.findHomography(src_pts, dst_pts)

    projected_track = []
    for entry in stats:
        if entry["x_px"] is not None and entry["y_px"] is not None:
            pt_img = np.array([[[entry["x_px"], entry["y_px"]]]], dtype=np.float32)
            pt_proj = cv2.perspectiveTransform(pt_img, H)[0][0]
            projected_track.append({
                'frame': int(entry['frame']),
                'x_m': float(pt_proj[0]),
                'y_m': float(pt_proj[1]),
                'dist_px': float(entry['dist_px']) if entry['dist_px'] != -1 else -1
            })

    with open(os.path.join(stats_dir, 'ball_track_projected.json'), 'w') as f:
        json.dump(projected_track, f, indent=2)

    
    if len(projected_track) >= 7:
        xs = np.array([p['x_m'] for p in projected_track])
        ys = np.array([p['y_m'] for p in projected_track])
        xs_smooth = savgol_filter(xs, 7, 2)
        ys_smooth = savgol_filter(ys, 7, 2)
        for i in range(len(projected_track)):
            projected_track[i]['x_m'] = float(xs_smooth[i])
            projected_track[i]['y_m'] = float(ys_smooth[i])

    
    WINDOW_SIZE = 9
    MAX_SPEED_RALLY = 40     # m/s (~162 km/h)
    MAX_SPEED_SERVE = 65     # m/s (~234 km/h)
    MAX_ACCELERATION = 25    # m/s²
    MIN_REASONABLE_STEP = 0.01
    MAX_REASONABLE_STEP = 12.0
    ABSOLUTE_MAX_SPEED = 55.5
    SERVE_ZONE_Y_THRESHOLD = 3.0

    projected_with_speed = projected_track.copy()
    valid_speeds = []

    for i in range(len(projected_track) - WINDOW_SIZE + 1):
        window = projected_track[i:i + WINDOW_SIZE]
        total_dist = 0.0
        valid = True

        for j in range(1, len(window)):
            dx = window[j]['x_m'] - window[j - 1]['x_m']
            dy = window[j]['y_m'] - window[j - 1]['y_m']
            dist = np.sqrt(dx ** 2 + dy ** 2)

            if dist < MIN_REASONABLE_STEP or dist > MAX_REASONABLE_STEP:
                valid = False
                break

            total_dist += dist

        if valid:
            avg_speed = total_dist / ((WINDOW_SIZE - 1) / fps)
            is_serve = (
                i <= 2 and avg_speed > 45 and
                any(p['y_m'] > SERVE_ZONE_Y_THRESHOLD for p in window)
            )
            max_speed_allowed = MAX_SPEED_SERVE if is_serve else MAX_SPEED_RALLY

            if avg_speed <= ABSOLUTE_MAX_SPEED and avg_speed <= max_speed_allowed:
                for j in range(i, i + WINDOW_SIZE):
                    projected_with_speed[j]['speed_mps'] = round(avg_speed, 2)
                valid_speeds.append(avg_speed)
            else:
                for j in range(i, i + WINDOW_SIZE):
                    projected_with_speed[j]['speed_mps'] = -1.0
        else:
            for j in range(i, i + WINDOW_SIZE):
                projected_with_speed[j]['speed_mps'] = -1.0

    for frame in projected_with_speed:
        if 'speed_mps' not in frame:
            frame['speed_mps'] = -1.0

    accelerations = []
    for i in range(1, len(projected_with_speed)):
        prev_speed = projected_with_speed[i - 1]['speed_mps']
        curr_speed = projected_with_speed[i]['speed_mps']

        if prev_speed > 0 and curr_speed > 0:
            accel = (curr_speed - prev_speed) * fps  
            accelerations.append(accel)

            if abs(accel) > MAX_ACCELERATION:
                projected_with_speed[i]['speed_mps'] = -1.0
        else:
            accelerations.append(0)  
 
    with open(os.path.join(stats_dir, 'ball_track_projected_with_speed.json'), 'w') as f:
        json.dump(projected_with_speed, f, indent=2)
