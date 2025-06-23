import sys
from pathlib import Path
import torch
import re
from app.backend.utils import log_gpu_status


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent  
sys.path.append(str(PROJECT_ROOT))  

from tracknetV2.model import BallTrackerNet
from tracker.tracker import MultiModelTracker

_models = {}

def load_models():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ball_model_path = PROJECT_ROOT / 'models_weigths/object_detection/model_best.pt'
        #print(f"Buscando modelo de pelota en: {ball_model_path}")
        if ball_model_path.exists():
            model = BallTrackerNet()
            model.load_state_dict(torch.load(ball_model_path, map_location=device))
            model.to(device).eval()
            log_gpu_status("BallTrackerNet", model)
            _models['ball'] = model
            print("Modelo de pelota cargado correctamente.")
        else:
            print(f"Modelo de pelota NO encontrado en: {ball_model_path}")

        seg = PROJECT_ROOT / "models_weigths/segmentation/best.pt"
        kp  = PROJECT_ROOT / "models_weigths/keypoint_detection/best.pt"
        #print(f"Buscando modelos tracker en:\n  Segmentación: {seg}\n  Keypoints: {kp}")

        if seg.exists() and kp.exists():
            tracker = MultiModelTracker(
                seg_model_path=str(seg),
                kp_model_path=str(kp),
                extract_stats=True
            )
            _models['tracker'] = tracker
            print("Modelo de tracking cargados correctamente.")
        else:
            if not seg.exists():
                print(f"Modelo de segmentación NO encontrado en: {seg}")
            if not kp.exists():
                print(f"Modelo de keypoints NO encontrado en: {kp}")
    except Exception as e:
        print(f"Error al cargar modelos: {e}")


def get_model(name):
    return _models.get(name)

def models_ready():
    return 'ball' in _models and 'tracker' in _models
