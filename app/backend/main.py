import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import asyncio
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.backend.routes import register_routes
from app.backend.models_loader import load_models, models_ready

from fastapi.staticfiles import StaticFiles
from pathlib import Path


if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = FastAPI(
    title="TennUS API",
    description="Procesamiento de video para análisis de tenis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  
        "http://localhost:5173",  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app)

@app.on_event("startup")
async def startup_event():
    logging.info("Iniciando servidor TennUS...")
    load_models()
    if models_ready():
        logging.info("Modelos cargados correctamente.")
    else:
        logging.error("Error al cargar modelos.")
        raise RuntimeError("Modelos no disponibles. Revisa los archivos .pt")

@app.get("/")
def root():
    return {"message": "API TennUS operativa"}

images_dir = Path(__file__).resolve().parent / "images"
app.mount("/images", StaticFiles(directory=images_dir), name="images")
