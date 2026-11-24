# Sistem Prediksi Harga Saham PT Telekomunikasi Indonesia Tbk (TLKM.JK)

Proyek prediksi harga penutupan saham **TLKM.JK** berbasis *Time-Series Regression* (**Ridge Regression**, **Linear Regression / OLS**) dan **Holt's Linear Trend** dengan evaluasi perbandingan terhadap **Naive Baseline**, dilengkapi antarmuka web **Streamlit** serta notebook eksperimen **Google Colab / Jupyter**.

## Struktur Paket

| File / Folder | Fungsi |
| :--- | :--- |
| `notebook/TLKM_Stock_Prediction_Revisi.ipynb` | Alur lengkap Data Understanding, EDA, Preprocessing, Feature Engineering, Modeling, Evaluasi, dan ekspor `data/tlkm_clean.csv`. |
| `forecaster.py` | Modul inti dengan logika yang sama persis dengan notebook, dikemas sebagai fungsi untuk aplikasi dan CLI. |
| `app.py` | Aplikasi web Streamlit dengan 3 tab: **Prediksi**, **Data**, dan **Evaluasi**. |
| `prepare_data.py` | Alternatif notebook untuk membersihkan CSV mentah lewat satu perintah terminal. |
| `evaluate.py` | Alternatif notebook untuk mencetak tabel evaluasi pada data asli di terminal. |
| `requirements.txt` | Daftar pustaka Python yang diperlukan. |
| `.streamlit/config.toml` | Tema warna dan konfigurasi Streamlit. |
| `data/tlkm_clean.csv` | Dataset saham TLKM.JK yang telah dibersihkan (07/11/2019 – 06/11/2024). |
| `tests/test_forecaster.py` | Uji otomatis memakai data sintetis (tidak untuk dilaporkan sebagai hasil eksperimen). |

## Menjalankan Secara Lokal (Windows PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Perintah CLI Opsional

```powershell
# Membersihkan CSV mentah
python prepare_data.py --input "SAHAM - PT Telekomunikasi Indonesia Tbk (TLKM.JK) - Sheet1.csv" --output data/tlkm_clean.csv

# Evaluasi model di terminal untuk laporan
python evaluate.py --data data/tlkm_clean.csv --train-ratio 0.8 --alpha 10.0

# Menjalankan uji otomatis (data sintetis)
python -m unittest discover -s tests -v
```

> **Disclaimer:** Proyek ini dibuat untuk tujuan pembelajaran dan analisis data historis, bukan saran investasi atau rekomendasi transaksi saham.
