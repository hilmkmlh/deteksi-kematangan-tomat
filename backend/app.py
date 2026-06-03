import os
import io
import base64
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

# ============================================================
# PATH
# ============================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))          # .../backend/
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend") # .../frontend/
MODEL_PATH   = os.path.join(BASE_DIR, "best.pt")

# ============================================================
# KONFIGURASI
# ============================================================
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

CLASS_MAPPING = {
    "matang":          "matang",
    "setengah_matang": "setengah_matang",
    "mentah":          "tidak_matang",
}

COLOR_PIL = {
    "matang":          (220, 66,  66),
    "setengah_matang": (245, 165, 60),
    "tidak_matang":    (90,  180, 90),
}

LABEL_DISPLAY = {
    "matang":          "Matang",
    "setengah_matang": "Setengah Matang",
    "tidak_matang":    "Tidak Matang",
}

# ============================================================
# INIT FLASK & MODEL
# ============================================================
app = Flask(__name__)
CORS(app)

def _load_model():
    if os.path.exists(MODEL_PATH):
        m = YOLO(MODEL_PATH)
        print(f"[INFO] Model dimuat. Kelas: {m.names}")
        return m

    # Fallback: download dari URL jika env MODEL_URL diset
    url = os.environ.get("MODEL_URL")
    if url:
        import urllib.request
        print(f"[INFO] Mendownload model dari {url} ...")
        urllib.request.urlretrieve(url, MODEL_PATH)
        return YOLO(MODEL_PATH)

    print(f"[WARNING] File model tidak ditemukan: {MODEL_PATH}")
    return None

model = _load_model()


# ============================================================
# HELPERS
# ============================================================
def normalize_class_name(raw: str) -> str:
    key = raw.lower().strip().replace(" ", "_").replace("-", "_")
    return CLASS_MAPPING.get(key, key)


def draw_detections(image_pil, results):
    from PIL import ImageDraw, ImageFont

    img  = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font_size = max(14, int(min(img.size) / 40))
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        font = ImageFont.load_default()

    for box in results.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf      = float(box.conf[0])
        raw_label = model.names[int(box.cls[0])]
        label     = normalize_class_name(raw_label)
        color     = COLOR_PIL.get(label, (200, 200, 200))
        thickness = max(2, int(min(img.size) / 250))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        text = f"{LABEL_DISPLAY.get(label, raw_label)} {conf:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        ty  = max(0, y1 - th - pad * 2)
        draw.rectangle([x1, ty, x1 + tw + pad * 2, ty + th + pad * 2], fill=color)
        draw.text((x1 + pad, ty + pad), text, fill=(255, 255, 255), font=font)

    return img


def pil_to_base64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# ROUTES — FRONTEND
# ============================================================
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ============================================================
# ROUTES — API
# ============================================================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "model_loaded":  model is not None,
        "model_classes": list(model.names.values()) if model else [],
        "timestamp":     datetime.now().isoformat(),
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({"error": "Model belum dimuat. Pastikan file best.pt ada."}), 500

    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang dikirim."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    try:
        image_pil = Image.open(file.stream).convert("RGB")

        results = model.predict(
            source=image_pil,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )[0]

        counts     = {"matang": 0, "setengah_matang": 0, "tidak_matang": 0}
        detections = []

        for box in results.boxes:
            raw_label = model.names[int(box.cls[0])]
            label     = normalize_class_name(raw_label)
            conf      = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            if label in counts:
                counts[label] += 1

            detections.append({
                "label":         label,
                "label_display": LABEL_DISPLAY.get(label, raw_label),
                "confidence":    round(conf, 3),
                "bbox":          [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
            })

        annotated     = draw_detections(image_pil, results)
        annotated_b64 = pil_to_base64(annotated)

        return jsonify({
            "success":        True,
            "total":          sum(counts.values()),
            "counts":         counts,
            "detections":     detections,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "image_size":     {"width": image_pil.width, "height": image_pil.height},
        })

    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 500


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
