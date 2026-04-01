FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO weights at build time (avoids runtime delay)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
