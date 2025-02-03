FROM python:3.9-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev \
    pkg-config \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="video_device=/dev/video0"

# Create video group and add permissions
RUN groupadd -r video && \
    chgrp -R video /app && \
    chmod -R g+rwx /app && \
    mkdir -p /tmp/.X11-unix && \
    chmod 777 /tmp/.X11-unix

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["python3", "run.py"]