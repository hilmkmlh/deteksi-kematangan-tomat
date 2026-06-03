FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages lalu ganti opencv-python -> headless (tidak butuh display/GUI)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir opencv-python-headless

COPY . .

EXPOSE 7860

CMD gunicorn --chdir backend app:app --bind 0.0.0.0:7860
