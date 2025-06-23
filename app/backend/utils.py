import torch

def log_gpu_status(model_name: str, model):
    try:
        device = next(model.parameters()).device
        #print(f"Modelo '{model_name}' está en: {device}")
    except Exception as e:
        print(f"No se pudo obtener el dispositivo de '{model_name}': {e}")
