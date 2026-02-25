# 🎨 StudioAI: Creating Amazing Paint with Stable Diffusion

Proyek submission **BFGAI (Belajar Fundamental Generative AI)** — Aplikasi generasi dan editing gambar berbasis AI menggunakan **Stable Diffusion** dengan antarmuka **Streamlit**.

---

## 📋 Deskripsi Proyek

**StudioAI** adalah aplikasi web interaktif yang memanfaatkan model **Stable Diffusion v1.5** untuk menghasilkan dan mengedit gambar secara kreatif. Aplikasi ini dibangun di atas Google Colab dengan GPU dan di-deploy melalui **Ngrok** tunnel.

### Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| ✨ **Text-to-Image** | Menghasilkan gambar dari prompt teks menggunakan Stable Diffusion |
| 🛠️ **Inpainting** | Mengedit bagian spesifik gambar menggunakan mask yang digambar langsung |
| 🔍 **Outpainting** | Memperluas gambar ke segala arah (zoom-out) |
| 🎛️ **Scheduler Selection** | Pilihan scheduler: Euler A, DPM++, DDIM |
| 📦 **Batch Generation** | Generate hingga 4 gambar sekaligus |
| 🧹 **Memory Management** | Flush GPU RAM untuk optimasi performa |

---

## 🤖 AI & Model yang Digunakan

- **Stable Diffusion v1.5** (`runwayml/stable-diffusion-v1-5`) — Model utama untuk Text-to-Image generation
- **Stable Diffusion Inpainting** (`runwayml/stable-diffusion-inpainting`) — Model khusus untuk Inpainting & Outpainting
- **Scheduler/Sampler**:
  - `EulerAncestralDiscreteScheduler` (Euler A)
  - `DPMSolverMultistepScheduler` (DPM++)
  - `DDIMScheduler` (DDIM)

---

## 🛠️ Teknologi & Library

| Teknologi | Kegunaan |
|-----------|----------|
| **Python** | Bahasa pemrograman utama |
| **PyTorch** | Framework deep learning (backend GPU) |
| **Hugging Face Diffusers** | Library untuk menjalankan model Stable Diffusion |
| **Hugging Face Transformers** | Tokenizer dan komponen NLP pendukung |
| **Streamlit** | Framework antarmuka web interaktif |
| **Streamlit Drawable Canvas** | Widget canvas untuk menggambar mask pada Inpainting |
| **Pillow (PIL)** | Manipulasi dan pemrosesan gambar |
| **NumPy** | Operasi array untuk pemrosesan mask |
| **Ngrok** | Tunneling untuk deploy aplikasi dari Google Colab |
| **Google Colab** | Platform eksekusi dengan GPU T4 |

---

## 📁 Struktur File

```
Submission/
├── Streamlit_submission_BFGAI_M Rislan Tristansyah.ipynb   # Notebook utama (Streamlit App)
├── pipeline-submission-bfgai-M Rislan Tristansyah.ipynb     # Notebook pipeline eksplorasi
├── video_demo_aplikasi_BFGAI.mp4                            # Video demo aplikasi
├── requirements.txt                                          # Daftar dependensi
└── README.md                                                 # Dokumentasi proyek
```

---

## ⚙️ Cara Menjalankan

### Prasyarat
- Akun **Google Colab** dengan akses GPU (T4 atau lebih tinggi)
- Akun **Ngrok** (gratis) untuk mendapatkan auth token

### Langkah-langkah

1. **Buka notebook** `Streamlit_submission_BFGAI_M Rislan Tristansyah.ipynb` di Google Colab
2. **Aktifkan GPU** — Runtime → Change runtime type → GPU (T4)
3. **Jalankan semua cell** secara berurutan:
   - Install dependencies
   - Tulis file `logic.py` dan `app.py`
   - Konfigurasi Ngrok auth token
   - Jalankan Streamlit server
4. **Akses aplikasi** melalui public URL yang dihasilkan Ngrok

---

## 📦 Dependencies

```
torch
diffusers
transformers
accelerate
safetensors
streamlit
streamlit_drawable_canvas==0.8.0
Pillow
numpy
matplotlib
pyngrok
huggingface_hub
```

Install semua dependensi:

```bash
pip install -r requirements.txt
```

---

## 🎯 Level Implementasi

Proyek ini mengimplementasikan tiga level kemampuan:

### 🟢 Basic
- Text-to-Image generation dengan parameter standar (prompt, negative prompt, seed, steps, CFG)

### 🟡 Skilled
- Memory flushing (`flush_memory`)
- Scheduler switching (Euler A, DPM++, DDIM)
- Batch image generation (hingga 4 gambar)

### 🔴 Advanced
- **Inpainting** — Edit objek dalam gambar menggunakan mask yang digambar manual
- **Outpainting** — Perluas canvas gambar 128px ke segala arah dengan background blur dan inpainting otomatis

---

## 👤 Author

**M Rislan Tristansyah**

---

## 📄 Lisensi

Proyek ini dibuat sebagai submission untuk program **Belajar Fundamental Generative AI (BFGAI)**.
