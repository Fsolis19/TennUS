#!/bin/bash

echo "Cerrando procesos en el puerto 8000..."
for pid in $(lsof -i :8000 -t 2>/dev/null); do
  echo "Cerrando PID $pid"
  kill -9 "$pid"
done

echo "Activando entorno virtual..."
source app/backend/venv/Scripts/activate

echo "Entorno activado. Ya puedes ejecutar el backend"

