FROM python:3.11-slim

WORKDIR /app

# System libraries yang dibutuhkan OpenCV headless & PIL
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install semua package, lalu ganti opencv-python -> opencv-python-headless
# supaya tidak butuh library GUI (libxcb) yang tidak ada di server
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir opencv-python-headless

COPY . .

CMD gunicorn --chdir backend app:app --bind 0.0.0.0:$PORT
