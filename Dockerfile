FROM python:3.10-slim

# System libraries needed by opencv-python, mediapipe, reportlab (fonts) and
# the MySQL client. Without these, `pip install opencv-python` imports fine
# but crashes at runtime with "libGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Folders the app writes to at runtime — created here so a fresh volume
# mount doesn't start out missing them.
RUN mkdir -p uploads reports data

EXPOSE 8000

# .env is not baked into the image (see .dockerignore) — it's supplied at
# `docker run` / `docker compose` time via env_file, so secrets never end
# up inside the image layers.
CMD ["python", "app.py"]
