from fastapi import UploadFile, File, HTTPException, APIRouter, FastAPI, Request, BackgroundTasks
from pathlib import Path
import shutil
import time
import re
from fastapi.responses import StreamingResponse, JSONResponse
import subprocess
import matplotlib
import zipfile
import io
import asyncio
import multiprocessing
import json
import numpy as np
import cv2

matplotlib.use('Agg')  

from .models_loader import get_model, load_models
from tracknetV2.infer_on_video import infer_model_streaming, remove_outliers, generate_ball_statistics

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "../uploads"
BALL_DET_DIR = BASE_DIR / "../output_video/ball_detection"
TRACK_DIR = BASE_DIR / "../output_video/tracked_everything"
STATISTICS_DEST_DIR = BASE_DIR.parent / "statistics"
IMAGES_DIR = BASE_DIR / "images"

for d in (UPLOAD_DIR, BALL_DET_DIR, TRACK_DIR, STATISTICS_DEST_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)

router = APIRouter()

def slugify_filename(filename: str) -> str:
    filename = filename.replace(" ", "_")
    filename = (
        filename
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
        .replace("Á", "A").replace("É", "E").replace("Í", "I")
        .replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
    )
    filename = re.sub(r"[^\w\-.]", "", filename)
    return filename

def move_json_files(source_dir: Path, dest_dir: Path):
    for file_path in source_dir.glob("*.json"):
        dest_path = dest_dir / file_path.name  
        shutil.copy(file_path, dest_path)
        #print(f"Copiado {file_path.name} a {dest_path}")
        file_path.unlink()
        #print(f"Eliminado {file_path.name} de {file_path.parent}")

async def run_script_and_save_images(script_path: Path, output_prefix: str):
    #print(f"Ejecutando {script_path.name}")
    proc = await asyncio.create_subprocess_exec(
        "python", str(script_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    print(f"{script_path.name}:\n{stdout.decode('latin-1')}")
    if stderr:
        print(f"{script_path.name}:\n{stderr.decode('latin-1')}")


def run_tracking(upload_path, final_path):
    print("Cargando modelos en el proceso...")
    load_models() 
    tracker = get_model('tracker')
    if not tracker:
        raise Exception("Error cargando modelo de tracking")
    tracker.track_video(str(upload_path), str(final_path), fps=30)
    print("Finalizado.")

def run_detection(upload_path, detected_path, q):
    print("Cargando modelos en el proceso...")
    load_models()
    model = get_model('ball')
    if not model:
        raise Exception("Error cargando modelo de detección")
    ball_track, dists, fps = infer_model_streaming(str(upload_path), model, str(detected_path))

    safe_ball_track = [(float(x) if x is not None else None, float(y) if y is not None else None) for x, y in ball_track]
    safe_dists = [float(d) if d != -1 else -1 for d in dists]

    print(f"Finalizado. Guardando resultados para el proceso principal...")

    results_path = detected_path.parent / "detection_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "ball_track": safe_ball_track,
            "dists": safe_dists,
            "fps": fps
        }, f)

    q.put(str(results_path)) 


@router.post("/process")
async def process_video(request: Request, video: UploadFile = File(...)):
    if not video.content_type.startswith("video/"):
        raise HTTPException(400, "Tipo de archivo no soportado.")

    project_root = BASE_DIR.parent.parent  
    dirs_to_clean = [
        BASE_DIR / "../output_video/ball_detection",
        BASE_DIR / "../output_video/tracked_everything",
        BASE_DIR / "../statistics",
        BASE_DIR / "../uploads",
        BASE_DIR / "images",
        project_root / "tennis_statistics" / "stats_files",
        project_root / "tennis_statistics" / "ball_stats",
    ]
    for dir_path in dirs_to_clean:
        abs_path = dir_path.resolve()
        if abs_path.exists():
            shutil.rmtree(abs_path)
        abs_path.mkdir(parents=True, exist_ok=True)
        #print(f"Limpiado {abs_path}")

    safe_name = slugify_filename(video.filename)
    upload_path = Path(UPLOAD_DIR) / safe_name
    #print(f"Video safe name: {safe_name}")
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    detected_path = Path(BALL_DET_DIR) / f"{upload_path.stem}_detect.avi"
    final_path = Path(TRACK_DIR) / f"{upload_path.stem}_final.mp4"

    q = multiprocessing.Queue()
    tracking_process = multiprocessing.Process(target=run_tracking, args=(upload_path, final_path))
    detection_process = None

    try:
        tracking_process.start()

        while tracking_process.is_alive():
            if await request.is_disconnected():
                print("Cliente desconectado, terminando tracking...")
                tracking_process.terminate()
                tracking_process.join()
                raise HTTPException(499, "Cliente desconectado. Procesamiento cancelado.")
            await asyncio.sleep(0.5)

        tracking_process.join()

        detection_process = multiprocessing.Process(target=run_detection, args=(upload_path, detected_path, q))
        detection_process.start()

        while detection_process.is_alive():
            if await request.is_disconnected():
                print("Cliente desconectado, terminando detección...")
                detection_process.terminate()
                detection_process.join()
                raise HTTPException(499, "Cliente desconectado. Procesamiento cancelado.")
            await asyncio.sleep(0.5)

        detection_process.join()

        
        results_file = Path(q.get())
        with open(results_file, "r") as f:
            data = json.load(f)

        ball_track = data["ball_track"]
        dists = data["dists"]
        fps = data["fps"]

      
        results_file.unlink()

        generate_ball_statistics(ball_track, dists, fps)

        move_json_files(project_root / "tennis_statistics" / "stats_files", STATISTICS_DEST_DIR)
        move_json_files(project_root / "tennis_statistics" / "ball_stats", STATISTICS_DEST_DIR)

        for filename in ["ball_track.json", "ball_track_projected.json", "ball_track_projected_with_speed.json"]:
            file_path = project_root / "tennis_statistics" / "ball_stats" / filename
            if file_path.exists():
                dest_path = STATISTICS_DEST_DIR / file_path.name
                shutil.copy(file_path, dest_path)
                print(f"Copiado {file_path.name} a {dest_path}")

        scripts = [
            BASE_DIR / "clustering_track_front.py",
            BASE_DIR / "zones_distances_speed_players_front.py",
            BASE_DIR / "ball_trayectory_speed_front.py"
        ]
        for script in scripts:
            await run_script_and_save_images(script, output_prefix=safe_name)

        return {
            "message": "Procesamiento completado",
            "detect_video": detected_path.name,
            "final_video": final_path.name
        }

    finally:
        if tracking_process.is_alive():
            tracking_process.terminate()
        if detection_process and detection_process.is_alive():
            detection_process.terminate()

'''@router.get("/video/{stage}/{filename}")
def get_video(stage: str, filename: str, request: Request):
    base = BALL_DET_DIR if stage == "detect" else TRACK_DIR
    path = Path(base) / filename

    if not path.exists():
        raise HTTPException(404, "Archivo no encontrado")

    file_size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes"
    }

    range_header = request.headers.get("range")
    if range_header:
        range_value = range_header.strip().lower().replace("bytes=", "")
        range_start, range_end = range_value.split("-")
        range_start = int(range_start)
        range_end = int(range_end) if range_end else file_size - 1
        content_length = range_end - range_start + 1

        def file_iterator(start, end):
            with open(path, "rb") as f:
                f.seek(start)
                bytes_read = 0
                while bytes_read < content_length:
                    chunk_size = min(8192, content_length - bytes_read)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    bytes_read += len(data)
                    yield data

        headers.update({
            "Content-Range": f"bytes {range_start}-{range_end}/{file_size}",
            "Content-Length": str(content_length),
        })
        return StreamingResponse(file_iterator(range_start, range_end), status_code=206, media_type="video/mp4", headers=headers)

    return StreamingResponse(open(path, "rb"), media_type="video/mp4", headers={"Content-Length": str(file_size)})'''

@router.get("/images")
def list_generated_images():
    images = []
    for image_path in IMAGES_DIR.glob("*.png"):
        images.append(image_path.name)
    return JSONResponse(content={"images": images})

@router.get("/download/statistics")
def download_statistics():
    stats_dir = STATISTICS_DEST_DIR
    zip_stream = io.BytesIO()

    with zipfile.ZipFile(zip_stream, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in stats_dir.glob("*.json"):
            zipf.write(file, arcname=file.name)

    zip_stream.seek(0)  

    return StreamingResponse(
        zip_stream,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=estadisticas.zip"}
    )

@router.get("/download/images")
def download_images():
    images_dir = IMAGES_DIR
    zip_stream = io.BytesIO()

    with zipfile.ZipFile(zip_stream, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in images_dir.glob("*.png"):
            zipf.write(file, arcname=file.name)

    zip_stream.seek(0) 

    return StreamingResponse(
        zip_stream,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=imágenes.zip"}
    )

@router.get("/download/videos")
def download_videos():
    ball_detection_dir = BALL_DET_DIR
    tracked_everything_dir = TRACK_DIR
    zip_stream = io.BytesIO()

    allowed_video_extensions = [".mp4", ".webm", ".ogg", ".avi"]

    with zipfile.ZipFile(zip_stream, "w", zipfile.ZIP_DEFLATED) as zipf:
        
        for file in ball_detection_dir.glob("*"):
            if file.suffix.lower() in allowed_video_extensions:
                zipf.write(file, arcname=f"ball_detection/{file.name}")
      
        for file in tracked_everything_dir.glob("*"):
            if file.suffix.lower() in allowed_video_extensions:
                zipf.write(file, arcname=f"tracked_everything/{file.name}")

    zip_stream.seek(0)

    return StreamingResponse(
        zip_stream,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=videos.zip"}
    )

def register_routes(app: FastAPI):
    app.include_router(router)
