# 🎵 Deep Learning for Spotify Track Popularity Prediction

## 📌 Tujuan

Proyek ini bertujuan untuk **memprediksi tingkat popularitas lagu di Spotify** menggunakan pendekatan **Deep Learning (Keras)** berbasis data metadata lagu seperti artis, album, durasi, dan tahun rilis.
Tujuan akhirnya adalah membangun model yang mampu memahami pola dari faktor-faktor tersebut untuk memperkirakan skor popularitas (0–100).

---

## ⚙️ Metodologi

### 1️⃣ Persiapan Data

Dataset utama: `spotify_data clean.csv`
Langkah yang dilakukan:

* Membersihkan data dan menangani nilai hilang (`NaN`).
* Membuat fitur baru:

  * `album_year` (tahun rilis album).
  * `explicit_bin` (lagu eksplisit atau tidak).
* Mengonversi fitur kategorikal (`artist_name`, `album_type`) menjadi **integer ID**.
* Menambahkan fitur statistik tambahan:

  * `artist_avg_popularity`: rata-rata popularitas artis berdasarkan data training.
  * `track_age`: selisih antara tahun sekarang dan tahun rilis.
  * `artist_track_count`: jumlah lagu artis di dataset (indikasi seberapa produktif atau dikenal).

### 2️⃣ Pembagian Data

Data dibagi menjadi:

* **Train:** 70%
* **Validation:** 10%
* **Test:** 20%

Target (`track_popularity`) diskalakan ke rentang **0–1** agar stabil dalam training.

### 3️⃣ Preprocessing

* Fitur numerik distandarkan menggunakan `StandardScaler`.
* Fitur kategorikal diubah ke ID integer untuk di-embedding.
* Output diaktifkan dengan **sigmoid** (hasil akhir tetap 0–1).

---

## 🧠 Arsitektur Model

| Komponen                | Deskripsi                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------- |
| **Embedding Layer**     | Untuk `artist_id` dan `albumtype_id` agar model mengenali representasi unik tiap artis dan album. |
| **Dense Layers**        | 256 → 128 → 64 neuron (ReLU activation) dengan L2 regularization.                                 |
| **Dropout**             | 0.25 untuk mencegah overfitting.                                                                  |
| **Batch Normalization** | Menstabilkan distribusi input antar-layer.                                                        |
| **Optimizer**           | Adam (learning rate = 3e-4, clipnorm = 1.0).                                                      |
| **Loss**                | Mean Squared Error (MSE) dan MAE sebagai metrik utama.                                            |
| **Callback**            | EarlyStopping, ReduceLROnPlateau, dan ModelCheckpoint.                                            |

---

## 📊 Hasil Evaluasi

| Model                  | MAE (Test) | MSE (Test) | Catatan                                           |
| ---------------------- | ---------- | ---------- | ------------------------------------------------- |
| **Model 1 (Original)** | 15.39      | 429.21     | Baseline, stabil namun masih bias artis.          |
| **Model 2 (Improved)** | 15.01      | 488.32     | MAE turun sedikit, MSE naik karena outlier besar. |

### 📈 Analisis Tambahan:

* Rata-rata kesalahan prediksi sekitar **±15 poin popularitas** (skala 0–100).
* Model cenderung **overestimate lagu-lagu artis terkenal atau klasik** (prediksi terlalu tinggi).
* Contoh error besar:
  `The Mystics (actual 2 → predicted 82.9)`
  `Ennio Morricone (actual 0 → predicted 79.6)`
  → Model mengira semua lagu artis legendaris itu masih populer.

---

## 🔍 Analisis Error

1. **Error dominan:** overpredict pada lagu lama & non-trending.
2. **Pola error:** residual negatif besar untuk artis lama atau legendaris.
3. **Kemungkinan penyebab:**

   * Model tidak mengenal konteks waktu/tren musik.
   * Tidak ada fitur audio (energi, tempo, valence, danceability).
   * Bias terhadap artis besar di data training.

---

## 🪄 Visualisasi yang Digunakan

* **Kurva MAE (train vs val)** → mengevaluasi overfitting.
* **Scatter Plot** `y_true vs y_pred` → melihat seberapa dekat garis diagonal.
* **Analisis Residual** → menemukan artis/lirik yang paling sering salah prediksi.

---

## 💡 Saran Pengembangan Selanjutnya

| Aspek                          | Rencana Perbaikan                                                                                     | Dampak                                                    |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Fitur Audio**                | Tambahkan kolom seperti `danceability`, `energy`, `tempo`, `valence`, `speechiness` dari Spotify API. | Model lebih “paham” karakter musik, bukan hanya metadata. |
| **Fitur Teks (Judul / Genre)** | Gunakan HashingVectorizer atau Word2Vec pada `track_name` dan `artist_genres`.                        | Model mengenali makna semantik dari lagu.                 |
| **Regularisasi Loss**          | Gunakan **Huber loss** agar robust terhadap outlier besar.                                            | Menurunkan MSE tanpa menaikkan MAE.                       |
| **Post-Processing**            | Kombinasikan prediksi dengan prior artis (`0.7*pred + 0.3*artist_avg_pop`).                           | Menstabilkan hasil pada artis legendaris.                 |
| **Tuning Hyperparameter**      | Optimasi batch size, learning rate, dan dropout.                                                      | Potensi peningkatan akurasi 5–10%.                        |
| **Visual Dashboard**           | Buat dashboard evaluasi (misal pakai Plotly).                                                         | Memudahkan eksplorasi hasil model.                        |

---

## 🧾 Kesimpulan

* Model deep learning berhasil mencapai **MAE sekitar 15** — baseline yang cukup kuat untuk data metadata Spotify tanpa fitur audio.
* Hasil menunjukkan bahwa model sudah mampu memahami pola umum popularitas, namun belum cukup tajam untuk kasus ekstrem (lagu klasik, lagu baru viral).
* Untuk hasil yang lebih akurat dan realistis, langkah berikutnya adalah memperkaya dataset dengan **fitur kontekstual (audio, genre, teks, waktu)** dan **loss function robust seperti Huber**.
