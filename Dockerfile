FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


COPY ./app/backend/requirements.txt .


RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 8000


CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

