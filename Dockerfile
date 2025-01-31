FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    libavcodec-extra \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="protocol_whitelist;file,rtp,udp"
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV GST_DEBUG=3
ENV OPENCV_LOG_LEVEL=DEBUG

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create a non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "run:app"]