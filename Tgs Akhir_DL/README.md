# 🧠 Brain Tumor MRI Classification

Proyek klasifikasi gambar MRI otak untuk mendeteksi jenis tumor menggunakan **Convolutional Neural Network (CNN)** dengan **TensorFlow/Keras**.

## 📋 Deskripsi

Proyek ini membangun model deep learning yang mampu mengklasifikasikan citra MRI otak ke dalam **4 kelas**:

| Kelas | Deskripsi |
|-------|-----------|
| `glioma` | Tumor Glioma |
| `meningioma` | Tumor Meningioma |
| `notumor` | Tidak Ada Tumor |
| `pituitary` | Tumor Pituitari |

Dataset yang digunakan: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) dari Kaggle.

---

## Teknologi & AI yang Digunakan

### Model Architecture
- **Convolutional Neural Network (CNN)** custom dengan 5 blok konvolusi
- **Input**: Gambar MRI berukuran `224x224` piksel (3 channel RGB)
- **Output**: Probabilitas 4 kelas (`softmax`)

### Detail Arsitektur CNN
```
Layer (Block)          | Filter | Keterangan
-----------------------|--------|------------------
Block 1 - Conv2D       | 32     | Deteksi tepi dasar
Block 2 - Conv2D       | 64     | Ekstraksi tekstur
Block 3 - Conv2D       | 128    | Pola tumor
Block 4 - Conv2D       | 256    | Fitur kompleks
Block 5 - Conv2D       | 512    | Fitur abstrak
GlobalAveragePooling2D  | -      | Reduksi dimensi
Dense                   | 256    | Fully connected
Dense (Output)          | 4      | Klasifikasi akhir
```

### Teknik yang Digunakan
- **Data Augmentation**: `RandomFlip` (horizontal), `RandomRotation` (0.05)
- **Batch Normalization**: Stabilisasi training di setiap blok
- **Dropout**: Progresif (0.2 → 0.3 → 0.4 → 0.5) untuk mencegah overfitting
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Pembagian Dataset
- **Training Set**: 80%
- **Validation Set**: 10%
- **Test Set**: 10%
- Menggunakan **Stratified Split** untuk distribusi kelas yang seimbang

---

## Tech Stack

| Teknologi | Kegunaan |
|-----------|----------|
| **Python 3.x** | Bahasa pemrograman utama |
| **TensorFlow / Keras** | Framework deep learning |
| **NumPy** | Komputasi numerik |
| **Pandas** | Manipulasi data |
| **Matplotlib** | Visualisasi grafik training |
| **Seaborn** | Visualisasi confusion matrix |
| **Scikit-learn** | Evaluasi model & split dataset |
| **TensorFlow Lite** | Konversi model untuk deployment mobile |
| **TensorFlow.js** | Konversi model untuk deployment web |

---

## Struktur Project

```
submission_bundle/
├── m-rislan-tristansyah-submission-akhir.ipynb  # Notebook utama (training & evaluasi)
├── model.tflite                                 # Model TensorFlow Lite
├── saved_model/                                 # Model SavedModel format
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   ├── assets/
│   └── variables/
├── tfjs_model/                                  # Model TensorFlow.js
│   ├── model.json
│   ├── group1-shard1of2.bin
│   └── group1-shard2of2.bin
├── README.md
└── requirements.txt
```

---

## Format Model yang Tersedia

| Format | File | Kegunaan |
|--------|------|----------|
| **SavedModel** | `saved_model/` | Inference di server (TensorFlow Serving) |
| **TF-Lite** | `model.tflite` | Deployment di perangkat mobile (Android/iOS) |
| **TF.js** | `tfjs_model/` | Deployment di browser/web app |

---

## Hasil Pelatihan (Training)

| Parameter | Nilai |
|-----------|-------|
| Total Epoch | 27 (dari maks 50, dihentikan oleh EarlyStopping) |
| Best Epoch | 17 |
| Best Training Accuracy | ~95.91% |
| Best Validation Accuracy | **93.33%** |
| Best Validation Loss | 0.1881 |
| Total Parameters | 1,705,924 (6.51 MB) |
| Trainable Parameters | 1,703,428 (6.50 MB) |

### Ringkasan Proses Training
- Epoch 1–5: Akurasi naik cepat dari ~60% ke ~85% (training), validasi mulai stabil di ~75%
- Epoch 8: Learning rate diturunkan dari `0.001` → `0.0002` oleh ReduceLROnPlateau
- Epoch 9–11: Lonjakan performa validasi ke **88.61%** → **92.36%**
- Epoch 17: Mencapai puncak validasi **93.33%** (model terbaik disimpan)
- Epoch 27: Training dihentikan oleh EarlyStopping (tidak ada peningkatan selama 10 epoch)

---

## Hasil Evaluasi (Test Set)

### Akurasi Akhir
| Metrik | Nilai |
|--------|-------|
| **Test Accuracy** | **94.72%** |
| **Test Loss** | **0.1573** |

### Classification Report (Per Kelas)

| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Glioma** | 0.92 | 0.97 | 0.94 | 180 |
| **Meningioma** | 0.97 | 0.84 | 0.90 | 180 |
| **No Tumor** | 0.97 | 0.98 | 0.97 | 180 |
| **Pituitary** | 0.94 | 1.00 | 0.97 | 180 |
| | | | | |
| **Accuracy** | | | **0.95** | **720** |
| **Macro Avg** | 0.95 | 0.95 | 0.95 | 720 |
| **Weighted Avg** | 0.95 | 0.95 | 0.95 | 720 |

### Analisis Performa
- **Pituitary** memiliki recall sempurna (1.00) — model sangat baik mendeteksi tumor pituitari
- **No Tumor** memiliki performa tertinggi secara keseluruhan (F1 = 0.97)
- **Meningioma** memiliki recall terendah (0.84) — beberapa kasus salah diklasifikasikan
- Secara keseluruhan, model mencapai **akurasi 95%** dengan performa seimbang di semua kelas
