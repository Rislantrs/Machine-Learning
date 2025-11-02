# Proyek Prediksi Churn Pelanggan Telekomunikasi

## 1. Latar Belakang & Masalah Bisnis

Di industri telekomunikasi, biaya untuk mengakuisisi pelanggan baru (akuisisi) jauh lebih mahal daripada mempertahankan pelanggan yang sudah ada (retensi). *Customer Churn* (pelanggan berhenti berlangganan) adalah masalah utama yang menggerus profitabilitas.

**Tujuan Proyek:**
Membangun sebuah model *machine learning* klasifikasi yang mampu **mengidentifikasi pelanggan yang memiliki risiko tinggi untuk *churn***.

Dengan model ini, tim retensi dapat secara proaktif menghubungi pelanggan yang "berisiko" dan memberikan penawaran khusus, insentif, atau bantuan teknis untuk mencegah mereka pergi.

## 2. Sumber Data

Proyek ini menggunakan dataset `telco.csv` yang berisi **7.043 data pelanggan**.

IBM Team. (2024). Telco customer churn (11.1.3+) [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8360350

Dataset ini mencakup berbagai fitur, di antaranya:
* **Data Demografis:** (Gender, Age, Senior Citizen, Married, Dependents)
* **Layanan yang Digunakan:** (Phone Service, Internet Service, Online Security, Streaming TV, dll.)
* **Informasi Akun & Tagihan:** (Tenure, Contract, Payment Method, Monthly Charge, Total Charges)
* **Variabel Target:** `Churn Label` ('Yes' atau 'No')

## 3. Alur Kerja Proyek (Metodologi)

Proyek ini dijalankan dengan alur kerja *end-to-end* sebagai berikut:

1.  **Pembersihan Data:**
    * Menghapus kolom-kolom yang tidak relevan untuk prediksi (misal: `Customer ID`, `Country`, `City`, `Zip Code`).
    * Menghapus kolom yang menyebabkan **kebocoran data** (*data leakage*), yaitu kolom yang baru ada setelah pelanggan *churn* (misal: `Churn Category`, `Churn Reason`, `Customer Status`).

2.  **Eksplorasi Data (EDA):**
    * Mengidentifikasi bahwa data **tidak seimbang (*imbalanced*)**, dengan ~26% pelanggan *churn* dan 74% *stay*.
    * Menemukan korelasi kuat antara *churn* dengan:
        * `Contract`: Pelanggan `Month-to-Month` jauh lebih sering *churn*.
        * `Tenure in Months`: Pelanggan baru (tenure rendah) jauh lebih sering *churn*.
        * `Monthly Charge`: Pelanggan dengan tagihan bulanan tinggi lebih berisiko.

3.  **Preprocessing & Feature Engineering:**
    * Data dibagi menjadi set **Training (80%)** dan **Testing (20%)** menggunakan `stratify` untuk menjaga proporsi *churn* di kedua set.
    * **PENTING:** Semua *preprocessing* (scaling/encoding) di-*fit* **hanya** pada data *training* untuk mencegah *data leakage*.
    * **Scaling:** `StandardScaler` diterapkan pada semua fitur numerik kontinu.
    * **Encoding (Ordinal):** `OrdinalEncoder` digunakan untuk `Contract` (`Month-to-Month`: 0, `One Year`: 1, `Two Year`: 2) untuk menjaga informasi tingkatan.
    * **Encoding (Categorical):** `OneHotEncoder` digunakan untuk semua fitur kategorikal lainnya.
    * **Imputasi:** Nilai `NaN` pada `Offer` dan `Internet Type` diisi dengan 'None', karena `NaN` di sini adalah informasi yang bermakna (misal: "Tidak ada layanan internet").

4.  **Penanganan Imbalance (Data Training):**
    * Menggunakan teknik **SMOTE** (*Synthetic Minority Over-sampling Technique*) **hanya** pada data *training* untuk menyeimbangkan jumlah kelas *churn* dan *stay*.

5.  **Pemodelan (Baseline):**
    * Melatih 5 model baseline pada data *training* yang sudah di-SMOTE:
        1.  Logistic Regression
        2.  Decision Tree
        3.  Random Forest
        4.  K-Nearest Neighbors (KNN)
        5.  XGBoost
    * Evaluasi difokuskan pada **F1-Score** dan **Recall** (untuk Churn).

6.  **Hyperparameter Tuning (Optimasi):**
    * Model dengan performa baseline terbaik (`XGBoost`) dipilih untuk optimasi.
    * `GridSearchCV` digunakan untuk mencari kombinasi *hyperparameter* terbaik (misal: `max_depth`, `learning_rate`, `n_estimators`) untuk memaksimalkan **F1-Score**.

7.  **Evaluasi & Interpretasi Model Final:**
    * Model `XGBoost (Tuned)` terbaik dievaluasi pada data *test* (yang tidak pernah dilihat dan tidak di-SMOTE).
    * **SHAP** digunakan untuk menginterpretasi "kotak hitam" model dan memahami fitur apa yang paling mendorong prediksi.

## 4. Hasil Model Final

Model `XGBoost (Tuned)` dipilih sebagai model produksi. Performa model pada data tes adalah sebagai berikut:

| Metrik | Skor (untuk Kelas 'Churn') | Analisis Bisnis |
| :--- | :--- | :--- |
| **F1-Score** | **0.921 (92.1%)** | **Keseimbangan terbaik** antara Precision dan Recall. |
| **Recall** | **0.898 (89.8%)** | Model berhasil **menemukan ~9 dari 10** pelanggan yang akan *churn*. |
| **Precision** | **0.944 (94.4%)** | 94% pelanggan yang diprediksi *churn* **memang benar** akan *churn*. (Sangat efisien, tim retensi tidak buang waktu). |
| **ROC AUC** | 0.990 (99.0%) | Kemampuan model untuk membedakan kelas *churn* dan *stay* sangat tinggi. |

## 5. Output Proyek

Proyek ini menghasilkan dua *output* utama:

1.  **`best_xgboost_model.pkl`**:
    File model *machine learning* yang sudah dilatih, di-*tuning*, dan siap digunakan untuk memprediksi pelanggan baru.

2.  **`xgboost_predictions_analysis.csv`**:
    File CSV yang berisi hasil prediksi pada data tes. File ini adalah "daftar prioritas" untuk tim retensi. Kolom utamanya:
    * `Actual Churn`: Status *churn* asli (1 atau 0).
    * `Predicted Churn`: Prediksi dari model (1 atau 0).
    * `Churn Probability`: Skor probabilitas (0.0 - 1.0) pelanggan akan *churn*.
    * `Prediction Type`: Analisis (True Positive, False Positive, True Negative, False Negative).

## 6. Tools & Libraries yang Digunakan

* **Manipulasi Data:** `pandas`, `numpy`
* **Preprocessing & Pemodelan:** `scikit-learn` (sklearn)
* **Penanganan Imbalance:** `imblearn` (untuk SMOTE)
* **Model Utama:** `xgboost`
* **Interpretasi Model (XAI):** `shap`
* **Penyimpanan Model:** `joblib`
