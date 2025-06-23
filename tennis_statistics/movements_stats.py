import numpy as np
import json
from collections import defaultdict
from datetime import timedelta
import cv2
from scipy.signal import savgol_filter
import os

class MovementStatsExtractor:
    def __init__(self, fps=30, velocity_window=11): 
        self.fps = fps
        self.velocity_window = velocity_window

        self.meters_per_pixel_x = None
        self.meters_per_pixel_y = None
        self.scale_estimated = False

        self.zone_time_seconds = defaultdict(float)
        self.zone_frame_counts = defaultdict(int)
        self.heatmap_feet_positions = []
        self.total_distance_per_player = defaultdict(float)
        self.frames_per_player = defaultdict(int)
        self.total_processed_frames = 0

        self.last_speed_frame_per_player = {}
        self.current_zone_per_player = {}
        self.instantaneous_speeds_per_player = defaultdict(list)

        self.singles_court_box = None
        self.full_court_bounding_box = None
        self.homography_matrix = None

        self.feet_history_per_player = defaultdict(list)
        self.distance_history_per_player = defaultdict(list)

        self.src_pts = None
        self.dst_pts = None

    
    def get_court_homography_points(self, court_size_m=(10.97, 23.77)):
        if hasattr(self, 'src_pts') and hasattr(self, 'dst_pts'):
            return self.src_pts, self.dst_pts
        return None, None
    
    def estimate_full_court_scale(self, zone_detections, frame, frame_idx=0):
        full_court_mask = None
        for mask, name in zone_detections:
            name = name.lower().strip()
            if name in {"in-singles", "in-doubles"}:
                if full_court_mask is None:
                    full_court_mask = mask.astype(np.uint8)
                else:
                    full_court_mask = cv2.bitwise_or(full_court_mask, mask.astype(np.uint8))

        if full_court_mask is None or cv2.countNonZero(full_court_mask) < 5000:
            print("Máscara no válida para estimar homografía.")
            return

        
        coords = np.column_stack(np.where(full_court_mask > 0))
        points = np.array([[x, y] for y, x in coords], dtype=np.float32)

        x, y, w, h = cv2.boundingRect(points)
        box_pts = np.array([
            [x, y],           
            [x + w, y],       
            [x + w, y + h],   
            [x, y + h]        
        ], dtype=np.float32)

        src_pts = self._order_points_clockwise(box_pts)

        leftmost_x = min(src_pts[0][0], src_pts[3][0])
        rightmost_x = max(src_pts[1][0], src_pts[2][0])
        correction0 = 167  
        correction1 = 167

        src_pts[0][0] = leftmost_x + correction0

        src_pts[1][0] = rightmost_x - correction1


        self.debug_draw_src_points(frame.copy(), src_pts, save_path=f"./debug_outputs/src_pts_frame_{frame_idx}.jpg")
        self.debug_save_combined_mask(full_court_mask, frame, f"./debug_outputs/full_court_mask_{frame_idx}.jpg")

        dst_pts = np.array([
            [0, 0],         
            [10.97, 0],     
            [10.97, 23.77],         
            [0, 23.77]              
        ], dtype=np.float32)

        H, _ = cv2.findHomography(src_pts, dst_pts)
        if H is not None:
            self.homography_matrix = H
            self.scale_estimated = True
            self.src_pts = src_pts
            self.dst_pts = dst_pts
            self.full_court_bounding_box = [x, y, x + w, y + h]
            self.singles_court_box = self.full_court_bounding_box
        else:
            print("Falló cálculo de homografía.")
    
    def _order_points_clockwise(self, pts):
        if len(pts) != 4:
            raise ValueError("Se requieren exactamente 4 puntos para ordenar")

        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]       
        rect[2] = pts[np.argmax(s)]       
        rect[1] = pts[np.argmin(diff)]    
        rect[3] = pts[np.argmax(diff)]    

        return rect
    
    def debug_save_combined_mask(self, mask, frame, filename="./debug_outputs/mask_debug.jpg"):
        overlay = frame.copy()
        mask_rgb = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, overlay)

    def _get_feet_position(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, y2)
    
    def _is_point_inside_mask(self, point, mask):
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.pointPolygonTest(contour, point, measureDist=False) >= 0:
                return True
        return False

    def _apply_homography(self, point):
        if self.homography_matrix is None:
            return None
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.homography_matrix)[0][0]
        return tuple(projected)

    def update(self, player_detections, zone_detections, frame):
        self.total_processed_frames += 1

        if not self._is_scale_estimated() and zone_detections:
            self.estimate_full_court_scale(zone_detections, frame)

        for raw_id, bbox in player_detections:
            player_id = int(raw_id)
            self.frames_per_player[player_id] += 1

            feet_px = self._get_feet_position(bbox)

            feet_proj = self._apply_homography(feet_px)
            if feet_proj is None:
                self.heatmap_feet_positions.append(feet_proj)

            self._update_speed(player_id, feet_proj)
            self._update_distance(player_id, feet_proj)
            self._update_zone_time(player_id, feet_px, zone_detections) 

    def debug_draw_src_points(self, frame, src_pts, save_path='./debug_src_pts.jpg'):
        if src_pts is None:
            print("src_pts es None.")
            return

        for i, pt in enumerate(src_pts):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{i}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)
        #print(f"src_pts guardados en: {save_path}")

    def _is_scale_estimated(self):
        return self.scale_estimated and self.homography_matrix is not None and self.full_court_bounding_box is not None

    def _update_speed(self, player_id, feet_proj):
        VELOCITY_WINDOW = self.velocity_window
        MAX_REASONABLE_STEP_METERS = 1.0
        SCALE_CORRECTION_FACTOR = 0.85
        MAX_PLAYER_SPEED = 6.0
        MAX_PLAYER_ACCEL = 4.0
        SMOOTH_WINDOW = 9
        SMOOTH_POLYORDER = 2

        history = self.feet_history_per_player[player_id]
        history.append(feet_proj)
        if len(history) > VELOCITY_WINDOW:
            history.pop(0)

        if len(history) < VELOCITY_WINDOW:
            return

        smoothed_history = smooth_positions(history, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)
        total_dist = sum(
            np.linalg.norm(np.array(smoothed_history[i + 1]) - np.array(smoothed_history[i]))
            for i in range(len(smoothed_history) - 1)
        )
        total_time = (len(smoothed_history) - 1) / self.fps

        if total_dist >= MAX_REASONABLE_STEP_METERS:
            #print(f"Desplazamiento {total_dist:.2f}m ignorado para jugador {player_id}.")
            return

        raw_speed = total_dist / total_time
        speed = raw_speed * SCALE_CORRECTION_FACTOR

        if not (0 < speed < MAX_PLAYER_SPEED):
            return

        if is_outlier_speed(self.instantaneous_speeds_per_player[player_id], speed):
            return

        speeds = self.instantaneous_speeds_per_player[player_id]
        if speeds:
            prev_speed = speeds[-1]
            acceleration = abs(speed - prev_speed) / total_time
            if acceleration > MAX_PLAYER_ACCEL:
                #print(f"Aceleración {acceleration:.2f} m/s² sospechosa para jugador {player_id}. Ignorado.")
                return

        speeds.append(speed)

    def _update_distance(self, player_id, feet_proj):

        if feet_proj is None:
            return 
    
        MAX_REASONABLE_STEP_METERS = 1.0
        DISTANCE_WINDOW = 5

        distance_history = self.distance_history_per_player[player_id]
        distance_history.append(feet_proj)

        if len(distance_history) < DISTANCE_WINDOW:
            return

        d0 = np.array(distance_history[0])
        dn = np.array(distance_history[-1])
        dist = np.linalg.norm(dn - d0)

        if dist < MAX_REASONABLE_STEP_METERS:
            self.total_distance_per_player[player_id] += dist
        #else:
            #print(f"Desplazamiento {dist:.2f}m ignorado para jugador {player_id}.")

        self.distance_history_per_player[player_id] = []

    def _update_zone_time(self, player_id, feet_px, zone_detections):
        current_zone = None
        for zone_box, zone_name in zone_detections:
            if self._is_point_inside_mask(feet_px, zone_box):
                current_zone = zone_name
                break

        if current_zone:
            self.zone_time_seconds[(player_id, current_zone)] += 1 / self.fps
            self.zone_frame_counts[(player_id, current_zone)] += 1
            self.current_zone_per_player[player_id] = current_zone

    def export_stats(self, path="movement_stats.json"):
        output = self._build_output_structure()
        self._populate_players_stats(output["players"])

        cleaned_output = MovementStatsExtractor.convert_numpy(output)
        self._save_to_json(cleaned_output, path)

        print(f"Estadísticas exportadas")

    def _build_output_structure(self):
        court_box = (
            self.singles_court_box.tolist()
            if hasattr(self.singles_court_box, "tolist")
            else self.singles_court_box
        )
        src_pts, dst_pts = self.get_court_homography_points()
        return {
            "players": {},
            "court_box": court_box,
            "full_court_box": self.full_court_bounding_box,
            "src_pts": src_pts.tolist() if src_pts is not None else None,
            "dst_pts": dst_pts.tolist() if dst_pts is not None else None
        }

    def _populate_players_stats(self, players_output):
        MAX_VALID_PLAYER_SPEED = 7.5  # m/s

        player_ids = set(self.total_distance_per_player) | {
            pid for pid, _ in self.zone_time_seconds
        }

        for pid in player_ids:
            valid_speeds = self._get_valid_speeds(pid, MAX_VALID_PLAYER_SPEED)
            avg_speed, max_speed = self._compute_speed_stats(valid_speeds, pid)

            zones = self._get_player_zones(pid)
            players_output[str(pid)] = {
                "zones": zones,
                "total_distance_meters": round(self.total_distance_per_player[pid], 2),
                "average_speed_meters_per_sec": round(avg_speed, 2),
                "max_speed_meters_per_sec": round(max_speed, 2)
            }

    def _get_valid_speeds(self, pid, max_speed_threshold):
        speeds = np.array([
            s for s in self.instantaneous_speeds_per_player[pid]
            if 0 < s < max_speed_threshold
        ])
        if speeds.size == 0 and self.instantaneous_speeds_per_player[pid]:
            print(f"Todas las velocidades del jugador {pid} fueron descartadas por el filtro.")
        return speeds

    def _compute_speed_stats(self, speeds, pid):
        if speeds.size == 0:
            return 0.0, 0.0
        avg_speed = float(np.mean(speeds))
        max_speed = float(np.percentile(speeds, 95))
        return avg_speed, max_speed

    def _get_player_zones(self, pid):
        zones = {}
        tiempo_en_zonas = 0.0
        for (player, zone) in self.zone_time_seconds:
            if player != pid:
                continue
            time_sec = self.zone_time_seconds[(player, zone)]
            tiempo_en_zonas += time_sec
            zones[zone] = {
                "time_seconds": round(time_sec, 2),
                "time_formatted": str(timedelta(seconds=int(time_sec))),
                "frames": self.zone_frame_counts[(player, zone)]
            }

        total_time = self.frames_per_player[pid] / self.fps
        tiempo_fuera = max(0.0, total_time - tiempo_en_zonas)

        zones["outside_zones"] = {
            "time_seconds": round(tiempo_fuera, 2),
            "time_formatted": str(timedelta(seconds=int(tiempo_fuera))),
            "frames": int(tiempo_fuera * self.fps)
        }

        return zones

    def _save_to_json(self, data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def get_heatmap_points(self):
        return self.heatmap_feet_positions

    @staticmethod
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: MovementStatsExtractor.convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MovementStatsExtractor.convert_numpy(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return obj


def smooth_positions(positions, window_length=9, polyorder=2):
    if len(positions) < window_length or window_length % 2 == 0:
        return positions
    x_vals = [p[0] for p in positions]
    y_vals = [p[1] for p in positions]
    x_smooth = savgol_filter(x_vals, window_length, polyorder)
    y_smooth = savgol_filter(y_vals, window_length, polyorder)
    return list(zip(x_smooth, y_smooth))

def is_outlier_speed(speeds, current_speed, threshold=2.5):
    if len(speeds) < 5:
        return False
    mean = np.mean(speeds)
    std = np.std(speeds)
    if std == 0:
        return False
    z_score = abs(current_speed - mean) / std
    return z_score > threshold

