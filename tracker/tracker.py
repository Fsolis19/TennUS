import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import supervision as sv
from tennis_statistics.movements_stats import MovementStatsExtractor
from collections import defaultdict
from pathlib import Path
import torch

class MultiModelTracker:
    def __init__(
        self,
        seg_model_path: str,
        kp_model_path: str,
        extract_stats: bool = False,
        gap_thresh_frames: int = 20,
        scene_thresh_frames: int = 6,
        min_point_frames: int = 90,
        hist_threshold: float = 0.6,
        serve_thresh_frames: int = 2,
        min_tracklet_length: int = 3
    ):
        self.seg_model = YOLO(seg_model_path)
        self.kp_model = YOLO(kp_model_path)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.seg_model.predict(dummy_input, device=self.device, verbose=False)
        _ = self.kp_model.predict(dummy_input, device=self.device, verbose=False)

        self._log_model_device("YOLO - Segmentación", self.seg_model.model)
        self._log_model_device("YOLO - Keypoints", self.kp_model.model)

        self.mask_tracker = sv.ByteTrack()
        self.kp_tracker = sv.ByteTrack()
        self.mask_annotator = sv.MaskAnnotator()
        self.kp_box_size = 6

        self.extract_stats = extract_stats
        self.current_extr = MovementStatsExtractor(fps=30, velocity_window=11) if extract_stats else None

        project_root = Path(__file__).resolve().parents[1]
        self.stats_base_dir = project_root / 'tennis_statistics' / 'stats_files'
        self.stats_base_dir.mkdir(parents=True, exist_ok=True)

        self.gap_thresh = gap_thresh_frames
        self.scene_thresh = scene_thresh_frames
        self.min_point_frames = min_point_frames
        self.hist_threshold = hist_threshold
        self.serve_thresh = serve_thresh_frames
        self.min_tracklet_length = min_tracklet_length

        self.no_player_frames = 0
        self.scene_change_count = 0
        self.current_point_frames = 0
        self.serve_start_count = 0
        self.in_serve_flag = False
        self.prev_hist = None
        self.frame_idx = 0
        self.point_idx = 0
        self.detections_log = []

        self.palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (192, 192, 192), (128, 128, 128), (0, 0, 0), (255, 255, 255), (100, 100, 255)
        ]


    def draw_tracked_keypoints(self, frame, detections):
        if detections is None or len(detections) == 0:
            return 

        for box, track_id, class_id in zip(
            detections.xyxy, detections.tracker_id, detections.class_id
        ):
            x = int((box[0] + box[2]) / 2)
            y = int((box[1] + box[3]) / 2)
            color = self.palette[int(class_id) % len(self.palette)]
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.putText(
                frame,
                f"ID {int(track_id)}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

    def track_video(self, input_video_path, output_video_path, fps: float = 30.0):
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        cap = cv2.VideoCapture(input_video_path)
        out = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self._detect_scene_change(frame):
                self.scene_change_count += 1
            else:
                self.scene_change_count = 0

            tracked_players = self._track_players(frame)
            if self.extract_stats:
                self._update_movement_stats(tracked_players, frame)

            tracked_keypoints = self._track_keypoints(frame)
            annotated_frame = self._annotate_frame(frame, tracked_players, tracked_keypoints)

            if out is None:
                out = self._initialize_video_writer(output_video_path, frame.shape[:2], fps)

            out.write(annotated_frame)
            self.frame_idx += 1

        cap.release()
        out.release()

        if self.extract_stats:
            self._finalize_point()

        print("Vídeo y stats exportados.")

    def _detect_scene_change(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)

        scene_changed = False
        if self.prev_hist is not None:
            correlation = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
            if correlation < self.hist_threshold:
                scene_changed = True
        self.prev_hist = hist

        return scene_changed and (self.scene_change_count >= self.scene_thresh)

    def _track_players(self, frame):
        #seg_res = self.seg_model.predict(frame, verbose=False)[0]
        seg_res = self.seg_model.predict(frame, verbose=False, device=self.device)[0]
        detections = sv.Detections.from_ultralytics(seg_res)
        tracked = self.mask_tracker.update_with_detections(detections)

        for i in range(len(tracked)):
            class_id = int(tracked.class_id[i])
            class_name = self.seg_model.names[class_id]
            if class_name == "player":
                x1, y1, x2, y2 = tracked.xyxy[i]
                tid = int(tracked.tracker_id[i])
                cx, cy = (x1 + x2) / 2, y2

                self.detections_log.append({
                    "track_id": tid,
                    "frame": self.frame_idx,
                    "cx": cx,
                    "cy": cy
                })
        return tracked

    def _update_movement_stats(self, tracked_players, frame):
        player_dets = [
            (int(tracked_players.tracker_id[i]), tracked_players.xyxy[i])
            for i in range(len(tracked_players))
            if self.seg_model.names[int(tracked_players.class_id[i])] == "player"
        ]
        zone_dets = [
            (tracked_players.mask[i], self.seg_model.names[int(tracked_players.class_id[i])])
            for i in range(len(tracked_players))
            if self.seg_model.names[int(tracked_players.class_id[i])].lower().startswith("in-")
        ]
        self.current_extr.update(player_dets, zone_dets,frame)

    def _finalize_point(self):
        prefix = f"point_{self.point_idx}"
        base = self.stats_base_dir

        raw_fp    = base / f"movement_{prefix}.json"
        merged_fp = base / f"movement_{prefix}_merged.json"
        log_fp    = base / f"detections_log_{prefix}.json"
        map_fp    = base / f"track_mapping_{prefix}.json"

        self.current_extr.export_stats(str(raw_fp))

        with open(log_fp, "w") as f:
            json.dump(MovementStatsExtractor.convert_numpy(self.detections_log), f, indent=4)

        mapping = self._cluster_player_ids()
        with open(map_fp, "w") as f:
            json.dump(MovementStatsExtractor.convert_numpy(mapping), f, indent=4)

        self._merge_stats(mapping, str(raw_fp), str(merged_fp))
        print(f"Punto {self.point_idx} en crudo y mergeado listo")

        prev_x = self.current_extr.meters_per_pixel_x
        prev_y = self.current_extr.meters_per_pixel_y
        prev_ok = self.current_extr.scale_estimated

        self.point_idx += 1
        self.current_extr = MovementStatsExtractor(fps=30)
        self.current_extr.meters_per_pixel_x = prev_x
        self.current_extr.meters_per_pixel_y = prev_y
        self.current_extr.scale_estimated = prev_ok

        self.detections_log = []
        self.frame_idx = 0
        self.no_player_frames = 0
        self.scene_change_count = 0
        self.current_point_frames = 0
        self.prev_hist = None
        self.serve_start_count = 0
        self.in_serve_flag = False

    def _track_keypoints(self, frame):
        #kp_res = self.kp_model.predict(frame, verbose=False)[0]
        kp_res = self.kp_model.predict(frame, verbose=False, device=self.device)[0]
        if kp_res.keypoints is None:
            return sv.Detections.empty()

        boxes, confs, cls_ids = [], [], []
        for person in kp_res.keypoints.xy.cpu().numpy():
            for kp_id, (x, y) in enumerate(person):
                if x > 0 and y > 0:
                    h = self.kp_box_size / 2
                    boxes.append([x - h, y - h, x + h, y + h])
                    confs.append(0.9)
                    cls_ids.append(kp_id)

        if not boxes:
            return sv.Detections.empty()

        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confs),
            class_id=np.array(cls_ids)
        )
        return self.kp_tracker.update_with_detections(detections)


    def _annotate_frame(self, frame, tracked_players, tracked_keypoints):
        annotated_frame = self.mask_annotator.annotate(frame.copy(), detections=tracked_players)
        self.draw_tracked_keypoints(annotated_frame, tracked_keypoints)
        return annotated_frame

    def _initialize_video_writer(self, output_video_path, frame_size, fps):
        height, width = frame_size
        return cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (width, height)
        )

    def _cluster_player_ids(self):
        tracklets = self._build_tracklets()
        valid_tracklets = self._filter_short_tracklets(tracklets)

        if len(valid_tracklets) < 2:
            return {tid: 0 for tid in valid_tracklets}

        centroids = self._compute_centroids(valid_tracklets)
        labels = self._apply_kmeans(centroids)

        cluster_to_player = self._assign_clusters_to_players(labels['centers'])
        return self._map_tracklets_to_players(valid_tracklets, labels['labels'], cluster_to_player)

    def _build_tracklets(self):
        tracklets = defaultdict(list)
        for detection in self.detections_log:
            tracklets[detection['track_id']].append((detection['cx'], detection['cy']))
        return tracklets

    def _filter_short_tracklets(self, tracklets):
        return {tid: pts for tid, pts in tracklets.items() if len(pts) >= self.min_tracklet_length}

    def _compute_centroids(self, tracklets):
        return np.array([np.mean(pts, axis=0) for pts in tracklets.values()], dtype=np.float32)

    def _apply_kmeans(self, centroids):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            centroids, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        return {'labels': labels.flatten(), 'centers': centers}

    def _assign_clusters_to_players(self, centers):
        centroids_y = centers[:, 1]
        p0_cluster = int(np.argmax(centroids_y))  # Jugador 0 es el que más baja tenga la coordenada y
        p1_cluster = 1 - p0_cluster
        return {p0_cluster: 0, p1_cluster: 1}

    def _map_tracklets_to_players(self, tracklets, labels, cluster_to_player):
        ids = list(tracklets.keys())
        return {
            tid: cluster_to_player[int(labels[i])]
            for i, tid in enumerate(ids)
        }
    
    def _merge_stats(self, mapping: dict, src_json: str, dst_json: str):
        raw_data = self._load_json(src_json)
        metadata = self._extract_metadata(raw_data)

        merged_stats = self._initialize_merged_stats()

        players_data = raw_data.get("players", {})

        for track_id_str, player_data in players_data.items():
            track_id = self._safe_parse_track_id(track_id_str)
            if track_id is None:
                continue

            player_idx = mapping.get(track_id)
            if player_idx is None:
                print(f"Track ID {track_id} no encontrado en mapping. Se omite.")
                continue

            if not self._is_valid_player_data(player_data):
                continue

            self._merge_player_data(merged_stats[player_idx], player_data)

        final_output = self._build_final_output(metadata, merged_stats)
        self._save_json(final_output, dst_json)
        print(f"Estadísticas fusionadas guardadas")

    def _load_json(self, path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)

    def _extract_metadata(self, raw_data: dict) -> dict:
        return {
            "full_court_box": raw_data.get("full_court_box", [0, 0, 0, 0]),
            "src_pts": raw_data.get("src_pts"),
            "dst_pts": raw_data.get("dst_pts"),
            "meters_per_pixel_x": raw_data.get("meters_per_pixel_x"),
            "meters_per_pixel_y": raw_data.get("meters_per_pixel_y")
        }

    def _initialize_merged_stats(self) -> defaultdict:
        return defaultdict(lambda: {
            "zones": defaultdict(lambda: {"time_seconds": 0.0, "frames": 0}),
            "total_distance_meters": 0.0,
            "speed_list": [],
            "max_speed": 0.0
        })

    def _safe_parse_track_id(self, track_id_str: str) -> int:
        try:
            return int(track_id_str.strip())
        except ValueError:
            print(f"Track ID '{track_id_str}' no es un entero válido. Se omite.")
            return None

    def _is_valid_player_data(self, data: dict) -> bool:
        has_movement = data.get("total_distance_meters", 0) > 0
        has_zones = bool(data.get("zones"))
        has_speed = data.get("average_speed_meters_per_sec", 0.0) > 0
        return has_movement or has_zones or has_speed

    def _merge_player_data(self, merged: dict, player_data: dict):
        for zone, stats in player_data.get("zones", {}).items():
            merged_zone = merged["zones"][zone]
            merged_zone["time_seconds"] += stats.get("time_seconds", 0.0)
            merged_zone["frames"] += stats.get("frames", 0)

        merged["total_distance_meters"] += player_data.get("total_distance_meters", 0.0)
        merged["speed_list"].append(player_data.get("average_speed_meters_per_sec", 0.0))
        merged["max_speed"] = max(
            merged["max_speed"],
            player_data.get("max_speed_meters_per_sec", 0.0)
        )

    def _build_final_output(self, metadata: dict, merged_stats: dict) -> dict:
        players_output = {}
        for player_idx, stats in merged_stats.items():
            avg_speed = (
                sum(stats["speed_list"]) / len(stats["speed_list"])
                if stats["speed_list"] else 0
            )
            players_output[f" Jugador {player_idx}"] = {
                "zones": {
                    zone: {
                        "time_seconds": round(zone_stats["time_seconds"], 2),
                        "frames": zone_stats["frames"]
                    } for zone, zone_stats in stats["zones"].items()
                },
                "total_distance_meters": round(stats["total_distance_meters"], 2),
                "average_speed_meters_per_sec": round(avg_speed, 2),
                "max_speed_meters_per_sec": round(stats["max_speed"], 2)
            }

        return {
            "full_court_box": metadata["full_court_box"],
            "src_pts": metadata["src_pts"],
            "dst_pts": metadata["dst_pts"],
            "players": players_output
        }

    def _save_json(self, data: dict, path: str):
        with open(path, 'w') as f:
            json.dump(MovementStatsExtractor.convert_numpy(data), f, indent=4)

    def _log_model_device(self, name: str, model):
        try:
            device = next(model.parameters()).device
            #print(f"Modelo '{name}' está en: {device}")
        except Exception as e:
            print(f"No se pudo obtener el dispositivo de '{name}': {e}")
