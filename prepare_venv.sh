#!/bin/bash

VENV_DIR="app/backend/venv"
REQUIREMENTS_FILE="app/backend/requirements.txt"
INSTALL_MARKER="$VENV_DIR/.installed"
PYTHON_BIN="python"  

echo "Cerrando procesos en el puerto 8000..."
for pid in $(lsof -i :8000 -t 2>/dev/null); do
  echo "Cerrando PID $pid"
  kill -9 "$pid"
done

if [ ! -d "$VENV_DIR" ]; then
  echo "Creando entorno virtual en $VENV_DIR..."
  $PYTHON_BIN -m venv "$VENV_DIR"
  echo "Entorno virtual creado."
  INSTALL_NEEDED=true
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
  source "$VENV_DIR/Scripts/activate"
else
  echo "No se encontró el script de activación del entorno virtual."
  exit 1
fi

if [ ! -f "$INSTALL_MARKER" ]; then
  echo "Instalando requisitos desde $REQUIREMENTS_FILE..."
  pip install -r "$REQUIREMENTS_FILE" && touch "$INSTALL_MARKER"
else
  echo "Requisitos ya instalados, saltando instalación."
fi

echo "Entorno activado. Ya puedes ejecutar el backend."

