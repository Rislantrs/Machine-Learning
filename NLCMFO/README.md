# Brain Tumor Classification with Lightweight CNN Optimized by Nonlinear Lévy Chaotic Moth-Flame Optimization (NLCMFO)

Proyek ini merupakan **replikasi eksperimen** dari paper:

> *Lightweight convolutional neural networks using nonlinear Lévy chaotic moth flame optimisation for brain tumour classification via efficient hyperparameter tuning*  
> (Scopus ID: 105009608283)

Tujuan utama proyek ini adalah:
- Membangun model **Lightweight CNN** untuk klasifikasi tumor otak berbasis citra MRI.
- Mengoptimasi **hyperparameter CNN** menggunakan algoritma **Nonlinear Lévy Chaotic Moth-Flame Optimization (NLCMFO)**.
- Membandingkan dan mendekati performa yang dilaporkan pada paper, dengan keterbatasan komputasi (menggunakan Kaggle GPU).

---

## 1. Deskripsi Singkat Proyek

### 1.1. Dataset

Dataset yang digunakan dalam replikasi ini adalah:

> **Brain Tumor MRI Dataset (4 Classes)**  
> Dataset citra MRI otak dengan 4 kelas tumor (contoh umum: *glioma, meningioma, pituitary, no-tumor / normal*).

Dataset di-attach di Kaggle dengan struktur folder (contoh):

```text
/kaggle/input/tumorv2/
    Training/
        Class1/
        Class2/
        Class3/
        Class4/
    Testing/
        Class1/
        Class2/
        Class3/
        Class4/
```

Perbedaan penting dibanding paper asli:

- Paper asli juga menggunakan dataset MRI tumor otak, tetapi detail versi dataset, jumlah gambar, dan pembagian datanya bisa sedikit berbeda.
- Dalam replikasi ini digunakan **Brain Tumor MRI Dataset 4 kelas** sebagaimana tersedia di Kaggle, sehingga:
  - Distribusi jumlah gambar per kelas,
  - Variasi citra,
  - Dan kemungkinan pra-pemrosesan awal
  
  dapat berbeda dari dataset yang digunakan penulis paper, walaupun secara konsep sama-sama 4 kelas tumor otak.

Notebook:
- Menggabungkan **Training** dan **Testing** bawaan dataset.
- Melakukan **shuffle** global dan kemudian split manual:

```python
# 70% → df_train_full (untuk K-Fold + training final)
# 30% → df_test_final (untuk evaluasi akhir, unseen)
```

Sehingga metodologi pembagian data mengikuti konsep di paper, yaitu **split 70:30**.

---

### 1.2. Arsitektur Lightweight CNN

Model CNN yang digunakan merupakan **Lightweight CNN 6-layer** sesuai deskripsi di paper, diimplementasikan dalam fungsi:

```python
create_flexible_cnn(width=128, height=128, depth=3, classes=NUM_CLASSES, l2_reg=0.0001)
```

Karakteristik utama:
- Input: gambar RGB berukuran **128 × 128 × 3**.
- 3 blok konvolusi:
  - Conv2D (32 filter) → Conv2D (32) → BatchNorm → ReLU → MaxPooling2D → Dropout
  - Conv2D (64 filter) → Conv2D (64) → BatchNorm → ReLU → MaxPooling2D → Dropout
  - Conv2D (128 filter) → Conv2D (128) → BatchNorm → ReLU → MaxPooling2D → Dropout
- Bagian klasifikasi:
  - Flatten
  - Dense(512) + BatchNorm + ReLU + Dropout(0.5)
  - Dense(NUM_CLASSES) + Softmax

Model ini didesain agar:
- **Ringan (lightweight)** → cocok untuk komputasi terbatas.
- Tetap mampu mengekstraksi fitur yang cukup dalam untuk klasifikasi tumor otak multi-kelas.

---

### 1.3. NLCMFO – Hyperparameter Optimization

Hyperparameter yang dioptimasi dengan **NLCMFO**:

1. Learning rate (LR)
2. Momentum (SGD)
3. Jumlah epoch maksimum
4. Koefisien regularisasi L2

Batas bawah dan atas yang digunakan:

```python
lb = [0.001, 0.10, 10, 0.0001]   # LR, Momentum, Epoch, L2
ub = [0.100, 0.99, 50, 0.0100]
```

#### Objective Function

Fungsi objektif `objective_function_kfold(hyperparameters)`:

- Melakukan **5-Fold Cross Validation** pada `df_train_full`.
- Di setiap fold:
  - Membuat generator data dengan augmentasi (rotation, shift, flip).
  - Membangun model CNN dengan kombinasi hyperparameter tertentu.
  - Training dengan **EarlyStopping** (monitor `val_accuracy`, patience=5, `restore_best_weights=True`).
- Mengembalikan **error rate rata-rata** (1 − rata-rata `val_accuracy`) × 100.

NLCMFO kemudian mencari kombinasi hyperparameter yang **meminimalkan error rate** ini.

#### NLCMFO Core

Fungsi utama:

```python
NLCMFO(objf, lb, ub, dim=4, N, Max_iteration)
```

Konsep utama:
- **Moth (ngengat)** = kandidat solusi (vektor hyperparameter).
- **Flame** = solusi terbaik sementara.
- Menggunakan:
  - **Levy Flight** untuk eksplorasi langkah acak besar.
  - **Chaotic maps** (sine & chebyshev) untuk mengontrol dinamika eksplorasi.
  - **Spiral updating** (inti Moth-Flame Optimization) untuk menggerakkan Moth menuju Flame.
- Diimplementasikan dengan:
  - `levy_flight(beta=1.5)`  
  - `get_chaotic_value(map_type, x_old)` dengan dua map:
    - `sine map`
    - `chebyshev map`
  - Mekanisme **chaotic switching**:
    - Jika error stagnan beberapa iterasi, beralih dari sine ke chebyshev.

---

### 1.4. Early Stopping & Memory Management

Karena training berjalan di lingkungan terbatas (Kaggle T4 GPU):

- Setiap fold K-Fold:
  - `tf.keras.backend.clear_session()`
  - `gc.collect()`
- Digunakan **EarlyStopping** untuk:
  - Memotong training yang tidak lagi meningkatkan `val_accuracy`.
  - Menghemat waktu dan VRAM.
- Final training juga menggunakan:
  - `ModelCheckpoint` (save best) berdasarkan `val_accuracy`.
  - `EarlyStopping` dengan patience yang lebih longgar.

---

## 2. Perbedaan Pengaturan dibanding Paper Asli

Dalam paper asli, NLCMFO untuk tuning hyperparameter kemungkinan menjalankan **lebih banyak iterasi optimasi** dan/atau populasi yang lebih besar.

Pada replikasi ini, karena keterbatasan komputasi (Kaggle, GPU T4, waktu eksekusi terbatas), digunakan pengaturan:

```python
N_population = 10      # Jumlah populasi (jumlah ngengat)
Max_iterations = 1     # Iterasi optimasi NLCMFO
```

Artinya:
- **Populasi relatif kecil (10)** dan
- **Iterasi algoritma optimasi hanya 1 kali**.

Konsekuensi:
- Proses optimasi **jauh lebih ringan** dan bisa jalan di Kaggle.
- Tetapi pencarian hyperparameter **tidak sedalam** eksperimen di paper (kemungkinan kualitas optimum sedikit di bawah atau lebih variatif).

Lingkungan eksekusi:
- Notebook dijalankan di **Kaggle** dengan:
  - **GPU: NVIDIA Tesla T4** (CUDA-enabled).
  - TensorFlow versi `2.18.0`.
  - Python `3.11.13`.
- Terdapat beberapa penyesuaian minor demi stabilitas:
  - Manajemen memori yang agresif (`clear_session`, `gc.collect`).
  - Mengaktifkan / menonaktifkan logging GPU.

---

## 3. Alur Eksperimen di Notebook

Secara garis besar, `kpst-final.ipynb` berisi blok-blok utama berikut:

1. **Import library & cek GPU**  
   Mengimpor `TensorFlow`, `Keras`, `sklearn`, `OpenCV`, `matplotlib`, dll.  
   Mencetak versi TensorFlow dan status ketersediaan GPU.

2. **Persiapan dataset (4 kelas)**  
   - Crawl semua file dari:
     - `/kaggle/input/tumorv2/Training/...`
     - `/kaggle/input/tumorv2/Testing/...`
   - Simpan ke DataFrame `df` dengan kolom:
     - `filename` (path gambar)
     - `class` (label)
   - Shuffle, buang duplikat, lalu split 70:30:
     - `df_train_full` (70% → K-Fold + train final)
     - `df_test_final` (30% → test final)

3. **Visualisasi sampel dataset**  
   - Menggunakan `ImageDataGenerator` untuk menampilkan beberapa contoh gambar dari `df_train_full` dalam grid 3×3.
   - Memastikan gambar sudah dalam bentuk RGB (3 channel) dan ukuran 128×128.

4. **Helper Math & CNN**  
   - Implementasi:
     - `levy_flight`
     - `get_chaotic_value`
     - `create_flexible_cnn`
   - Fungsi-fungsi ini membentuk pondasi NLCMFO dan arsitektur Lightweight CNN.

5. **Objective Function + K-Fold + Early Stopping**  
   - Fungsi `objective_function_kfold`:
     - Menerima satu vektor hyperparameter.
     - Menjalankan 5-Fold Cross Validation.
     - Menghitung rata-rata error rate (1 − `val_accuracy`).

6. **Algoritma NLCMFO**  
   - Fungsi `NLCMFO`:
     - Inisialisasi populasi.
     - Evaluasi fitness dengan memanggil `objective_function_kfold`.
     - Update posisi Moth dengan spiral + Levy Flight + fitur chaotic.
   - Di akhir, mengembalikan:
     - `Best_flame_pos` → hyperparameter terbaik.
     - `Best_flame_score` → error rate terbaik.

7. **Eksekusi Utama (Optimasi + Final Training)**  
   - Menjalankan NLCMFO dengan:
     ```python
     N_population = 10
     Max_iterations = 1
     ```
   - Mencetak:
     - LR terbaik
     - Momentum terbaik
     - Epoch terbaik
     - L2 terbaik
   - Melatih model final pada `df_train_full`:
     - Generator dengan augmentasi ringan.
     - `ModelCheckpoint` + `EarlyStopping`.
   - Evaluasi final pada `df_test_final`.

8. **Eksekusi Final Manual (Skip Optimasi)**  
   - Ada satu blok untuk langsung training dengan **hyperparameter manual** yang diambil dari log terbaik sebelumnya:

     ```python
     MANUAL_LR     = 0.08066
     MANUAL_MOM    = 0.11968
     MANUAL_EPOCHS = 50
     MANUAL_L2     = 0.00180
     ```

   - Model dilatih dengan pengaturan tersebut sekali lagi, kemudian disimpan sebagai:
     - `Best_Model_Saved_Manual.h5`

9. **Evaluasi Lengkap + Confusion Matrix + ROC**  
   - Load bobot terbaik (`Best_Model_Saved_Manual.h5`).
   - Hitung:
     - Accuracy
     - Precision (macro)
     - Recall/Sensitivity (macro)
     - Specificity (rata-rata multi-class)
     - F1-Score
   - Tampilkan:
     - Tabel metrik global
     - Confusion matrix (heatmap dengan `seaborn`)
     - ROC curve multi-class (One-vs-Rest) dengan AUC per kelas.

10. **Analisis Performa per Kelas**  
    - Menggunakan `classification_report` dari `sklearn`.
    - Menyusun tabel per kelas:
      - Precision
      - Sensitivity (Recall)
      - Specificity
      - F1-Score
      - Jumlah data (support)
    - Menjelaskan makna sensitivity, specificity, dll.

11. **Grad-CAM Visualisation**  
    - Implementasi **Grad-CAM** untuk menjelaskan area mana di MRI yang paling berkontribusi terhadap prediksi.
    - Fungsi:
      - `make_gradcam_heatmap`
      - `display_gradcam_with_barplot(img_path)`
    - Output:
      - Gambar MRI + overlay heatmap (area merah = fitur kunci).
      - Bar plot probabilitas prediksi pada semua kelas.

---

## 4. Ringkasan Hasil Paper Asli

Paper membandingkan metode **CNN_NLCMFO** (metode utama) dengan beberapa arsitektur CNN lain:

- CNN_NLCMFO (metode yang direplikasi di sini)
- Darknet19
- EfficientNetB0
- Xception
- ResNet101
- InceptionResNetV2

Ringkasan hasil performa (dalam persen, AVG = rata-rata):

### 4.1. Accuracy

| Metode           | AVG    | Min    | Max    | STD    | P-Value    |
|------------------|--------|--------|--------|--------|------------|
| **CNN_NLCMFO**   | 97.40% | 96.75% | 97.80% | 0.004  | 3.57E−06   |
| Darknet19        | 96.41% | 96.10% | 96.80% | 0.522  | 3.03E−09   |
| EfficientNetB0   | 96.32% | 96.32% | 97.19% | 0.285  | 2.27E−01   |
| Xception         | 96.41% | 93.73% | 98.05% | 1.548  | 1.18E−09   |
| ResNet101        | 92.15% | 87.30% | 94.81% | 3.529  | 3.25E−07   |
| InceptionResNetV2| 95.63% | 94.16% | 96.97% | 0.899  | –          |

### 4.2. Recall (Sensitivity)

| Metode           | AVG    | Min    | Max    | STD    | P-Value    |
|------------------|--------|--------|--------|--------|------------|
| **CNN_NLCMFO**   | 96.00% | 95.50% | 96.70% | 0.0038 | 1.15E−03   |
| Darknet19        | 95.72% | 90.00% | 98.50% | 2.952  | 4.12E−01   |
| EfficientNetB0   | 96.08% | 93.00% | 99.00% | 2.052  | 4.65E−02   |
| Xception         | 95.50% | 88.30% | 98.50% | 3.739  | 1.24E−09   |
| ResNet101        | 88.88% | 82.70% | 94.60% | 4.774  | 4.71E−01   |
| InceptionResNetV2| 96.14% | 93.50% | 99.00% | 1.753  | –          |

### 4.3. Specificity

| Metode           | AVG    | Min    | Max    | STD    | P-Value    |
|------------------|--------|--------|--------|--------|------------|
| **CNN_NLCMFO**   | 98.60% | 97.95% | 98.90% | 0.0036 | 6.44E−04   |
| Darknet19        | 97.32% | 95.70% | 100.0% | 1.589  | 2.66E−04   |
| EfficientNetB0   | 96.70% | 95.00% | 99.10% | 1.596  | 2.24E−01   |
| Xception         | 97.86% | 94.60% | 99.60% | 1.713  | 3.25E−07   |
| ResNet101        | 95.44% | 91.60% | 98.50% | 2.787  | 2.66E−04   |
| InceptionResNetV2| 96.96% | 95.30% | 99.10% | 1.367  | –          |

### 4.4. Precision

| Metode           | AVG    | Min    | Max    | STD    | P-Value    |
|------------------|--------|--------|--------|--------|------------|
| **CNN_NLCMFO**   | 98.40% | 97.80% | 98.65% | 0.003  | 2.66E−04   |
| Darknet19        | 96.80% | 94.70% | 100.0% | 1.972  | 2.66E−04   |
| EfficientNetB0   | 96.04% | 93.70% | 99.10% | 2.108  | 1.45E−01   |
| Xception         | 97.48% | 93.50% | 99.50% | 2.078  | 1.13E−05   |
| ResNet101        | 94.80% | 90.30% | 98.60% | 3.409  | 2.66E−04   |
| InceptionResNetV2| 96.46% | 94.40% | 99.10% | 1.683  | –          |

### 4.5. F1-Score

| Metode           | AVG    | Min    | Max    | STD    | P-Value    |
|------------------|--------|--------|--------|--------|------------|
| **CNN_NLCMFO**   | 97.18% | 95.50% | 97.85% | 0.008  | 3.33E−01   |
| Darknet19        | 96.14% | 90.30% | 98.80% | 2.999  | 6.32E−01   |
| EfficientNetB0   | 96.52% | 93.50% | 99.20% | 1.977  | 6.32E−01   |
| Xception         | 96.04% | 89.50% | 98.80% | 3.417  | 2.20E−08   |
| ResNet101        | 89.56% | 82.80% | 95.60% | 4.902  | 7.56E−03   |
| InceptionResNetV2| 95.98% | 93.10% | 99.20% | 2.075  | –          |

Dari tabel-tabel di atas, dapat disimpulkan bahwa:

- **CNN_NLCMFO** konsisten memberikan:
  - **Akurasi sangat tinggi (~97.4%)**
  - **Precision dan Specificity tertinggi** di antara metode yang dibandingkan.
  - F1-Score tinggi dan stabil (STD rendah).
- Hal ini menunjukkan bahwa kombinasi:
  - **Lightweight CNN** +  
  - **NLCMFO untuk hyperparameter tuning**
  
  mampu menghasilkan model yang:
  - Akurat
  - Stabil
  - Lebih ringan dibanding arsitektur CNN besar seperti ResNet101 / InceptionResNetV2.

---

## 5. Hasil Replikasi (Notebook Ini)

Pada replikasi menggunakan:
- Populasi: `N_population = 10`
- Iterasi NLCMFO: `Max_iterations = 1`
- Lingkungan: **Kaggle – NVIDIA Tesla T4 (CUDA), TensorFlow 2.18.0**
- Dataset: **Brain Tumor MRI Dataset (4 kelas)** di Kaggle (`/kaggle/input/tumorv2`)

Model final dievaluasi pada **30% data test (unseen)**.  
Metrik yang diperoleh (tabel “TABEL PERFORMA LENGKAP” di notebook):

```text
Accuracy     : 97.01 %
Precision    : 96.91 %
Recall       : 96.90 %
Specificity  : 99.02 %
F1-Score     : 96.89 %
```

Interpretasi singkat:

- **Accuracy 97.01%** menunjukkan model hampir setara dengan rata-rata akurasi yang dilaporkan di paper (~97.4%), meskipun:
  - Dataset yang digunakan adalah **Brain Tumor MRI Dataset 4 kelas** dari Kaggle (bukan persis dataset penulis).
  - Pengaturan optimasi jauh lebih ringan (`N=10`, iterasi=1).
- **Specificity 99.02%** menunjukkan model sangat bagus dalam **menghindari false positive** (jarang salah menandai gambar sehat sebagai tumor atau salah jenis tumor).
- **Precision dan Recall sekitar 96.9%** menandakan keseimbangan yang baik antara kemampuan mendeteksi tumor dan menghindari prediksi salah.
- **F1-Score 96.89%** mengkonfirmasi bahwa performa stabil dan seimbang antara precision dan recall.

Selain itu:
- Ditampilkan **confusion matrix** untuk analisis kesalahan per kelas.
- Ditampilkan **ROC curve** multi-class beserta AUC per kelas.
- Ditampilkan **tabel performa per kelas** (Precision, Sensitivity, Specificity, F1-Score, jumlah sampel).
- Visualisasi **Grad-CAM** menunjukkan area MRI yang paling diperhatikan model saat memprediksi jenis tumor, memberikan interpretabilitas yang lebih baik.

---

## 6. Cara Menjalankan

1. Buka **Kaggle Notebook** dan attach dataset (misalnya `tumorv2`).
2. Pastikan `Accelerator` di-set ke **GPU (NVIDIA Tesla T4)**.
3. Upload file notebook `kpst-final.ipynb`.
4. Jalankan sel secara berurutan:
   - Import library & cek GPU.
   - Persiapan dataset (BAGIAN 1).
   - Helper math & CNN (BAGIAN 2).
   - Objective function & NLCMFO (BAGIAN 3 & 4).
   - Eksekusi utama (BAGIAN 7) **atau** eksekusi final manual (BAGIAN “EKSEKUSI FINAL LANGSUNG”).
   - Evaluasi akhir, confusion matrix, ROC, dan Grad-CAM.

---

## 7. Kesimpulan

- Proyek ini berhasil:
  - Membangun **Lightweight CNN** sesuai ide di paper.
  - Mengimplementasikan **Nonlinear Lévy Chaotic Moth-Flame Optimization (NLCMFO)** untuk hyperparameter tuning.
  - Menjalankan evaluasi lengkap: K-Fold, test final, confusion matrix, ROC, dan Grad-CAM.
- Karena keterbatasan komputasi (Kaggle T4, waktu eksekusi), konfigurasi optimasi disederhanakan:
  - `N_population = 10`
  - `Max_iterations = 1`
- Dataset yang digunakan adalah **Brain Tumor MRI Dataset 4 kelas** versi Kaggle, sehingga hasil tidak identik tetapi **sejalan** dengan tren performa yang dilaporkan di paper.
- Meskipun begitu, pipeline tetap:
  - Representatif terhadap metodologi paper asli.
  - Dapat diperluas (menambah populasi/iterasi atau mengganti dataset) bila komputasi memungkinkan.

---

## 8. Lisensi & Kredit

- Implementasi CNN dan NLCMFO ini adalah **replikasi akademik** untuk pembelajaran dan penelitian.
- Mohon selalu mengutip paper asli jika menggunakan metode ini dalam publikasi:

> *Lightweight convolutional neural networks using nonlinear Lévy chaotic moth flame optimisation for brain tumour classification via efficient hyperparameter tuning* (Scopus ID: 105009608283).
