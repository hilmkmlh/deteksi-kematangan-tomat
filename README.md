# 🍅 Tomatech — Deteksi Kematangan Tomat dengan YOLOv8

Website untuk mendeteksi dan menghitung tingkat kematangan buah tomat menggunakan YOLOv8.

## 📁 Struktur Folder

```
tomato-detection/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt    # Dependensi Python
│   └── best.pt             # ⚠️ MODEL DARI COLAB (perlu kalian tambahkan)
└── frontend/
    └── index.html          # Website (HTML + CSS + JS)
```

---

## 🚀 PANDUAN LENGKAP: Dari Google Colab ke Website

### **LANGKAH 1: Export Model dari Google Colab**

Setelah training selesai di Colab, model YOLOv8 disimpan dalam file `best.pt`. Ini cara mengambilnya:

#### Opsi A — Download manual dari Colab
```python
# Di Google Colab, jalankan cell ini:
from google.colab import files

# Lokasi default hasil training YOLOv8:
# runs/detect/train/weights/best.pt
files.download('runs/detect/train/weights/best.pt')
```

File `best.pt` akan terdownload ke komputer kalian.

#### Opsi B — Save ke Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy('runs/detect/train/weights/best.pt',
            '/content/drive/MyDrive/best.pt')
```

#### Opsi C — Cek nama kelas model (PENTING!)
```python
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
print(model.names)
# Contoh output: {0: 'ripe', 1: 'half_ripe', 2: 'unripe'}
# atau:          {0: 'matang', 1: 'setengah_matang', 2: 'belum_matang'}
```

⚠️ **Catat nama kelas-nya!** Akan kita pakai di langkah berikutnya.

---

### **LANGKAH 2: Setup Backend (Python)**

1. **Pindah ke folder backend:**
   ```bash
   cd backend
   ```

2. **Letakkan file `best.pt`** dari Colab ke folder `backend/`

3. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
   *Catatan: Instalasi `ultralytics` & `torch` bisa makan waktu 5-10 menit.*

4. **Sesuaikan `CLASS_MAPPING` di `app.py`** (jika perlu):

   Buka `app.py` dan cek bagian ini:
   ```python
   CLASS_MAPPING = {
       "ripe": "matang",
       "half_ripe": "setengah_matang",
       "unripe": "tidak_matang",
       # ... tambahkan variasi nama kelas dari model kalian
   }
   ```

   Pastikan **key** (sebelah kiri) sesuai dengan output `model.names` di Colab.
   Contoh: kalau di Colab kelasnya `{0: 'green', 1: 'red', 2: 'yellow'}`, maka tambahkan:
   ```python
   CLASS_MAPPING = {
       "green":  "tidak_matang",
       "red":    "matang",
       "yellow": "setengah_matang",
   }
   ```

5. **Jalankan server:**
   ```bash
   python app.py
   ```

   Output yang diharapkan:
   ```
   [INFO] Memuat model dari: best.pt
   [INFO] Model berhasil dimuat. Kelas: {0: 'ripe', 1: 'half_ripe', 2: 'unripe'}
    * Running on http://127.0.0.1:5000
   ```

---

### **LANGKAH 3: Buka Website**

1. **Pindah ke folder frontend:**
   ```bash
   cd ../frontend
   ```

2. **Buka `index.html`** dengan salah satu cara:

   **Cara 1 — Double-click** file `index.html` (paling mudah).

   **Cara 2 — Gunakan Live Server** (VSCode extension), atau:

   **Cara 3 — Python server:**
   ```bash
   python -m http.server 8000
   ```
   Lalu buka `http://localhost:8000` di browser.

3. **Pastikan backend tetap jalan** di terminal yang lain!

4. Indicator di pojok kanan atas akan menampilkan:
   - 🟢 **Model Aktif** → siap digunakan
   - 🔴 **Backend Offline** → backend belum jalan
   - 🔴 **Model Belum Dimuat** → file `best.pt` tidak ditemukan

---

## 🎨 Fitur Website

- ✅ **Upload foto** (drag & drop atau pilih file)
- ✅ **Capture langsung dari kamera** (mobile + desktop)
- ✅ **Bounding box** otomatis dengan warna berbeda per kategori
- ✅ **Total hitungan** + breakdown per kategori
- ✅ **Confidence score** per deteksi
- ✅ **Status indicator** untuk koneksi backend
- ✅ **Responsive** — bisa dipakai di HP

---

## 🔧 Tips & Troubleshooting

### 🐛 "Backend Offline"
Backend belum berjalan. Cek terminal — apakah `python app.py` masih jalan?

### 🐛 "CORS error" di browser console
Sudah saya antisipasi dengan `flask-cors`. Kalau masih error, coba akses frontend lewat `http://localhost:8000` (bukan `file://...`).

### 🐛 Hasil deteksi 0 tomat padahal ada
Coba turunkan `CONF_THRESHOLD` di `app.py`:
```python
CONF_THRESHOLD = 0.15   # default 0.25
```

### 🐛 Kategori tidak sesuai
Cek output `print(model.names)` di Colab, lalu pastikan `CLASS_MAPPING` di `app.py` cocok.

### 🐛 Inferensi lambat (>5 detik)
Ini normal kalau pakai CPU. Kalau punya GPU NVIDIA, install PyTorch versi CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🌐 Deploy ke Internet (Opsional)

Kalau mau presentasi dan tidak mau install di laptop dosen/penguji:

| Komponen | Layanan Gratis | Catatan |
|---|---|---|
| **Backend** | Render, Railway, HuggingFace Spaces | HF Spaces paling cocok untuk model ML |
| **Frontend** | GitHub Pages, Netlify, Vercel | Update `API_URL` di `index.html` |

**Rekomendasi: Hugging Face Spaces** — bisa host backend Python + model YOLOv8 gratis, dan biasanya cukup cepat untuk demo.

---

## 👥 Kredit

Projek mata kuliah **Visual Komputer Cerdas**
Tema: Deteksi dan Penghitungan Tingkat Kematangan Buah Tomat Berbasis YOLOv8
