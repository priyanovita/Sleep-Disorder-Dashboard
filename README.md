# 🌙 Klasifikasi Gangguan Tidur — Sleep Disorder Classification

## 📦 File yang Dihasilkan
```
sleep_disorder_classification.py  ← Kode utama (jalankan di Google Colab)
app.py                             ← Auto-generated oleh script di atas
requirements.txt
```

## 🚀 Cara Pakai di Google Colab

### Step 1 — Install dependensi
```python
!pip install -r requirements.txt
```

### Step 2 — Jalankan semua cell di `sleep_disorder_classification.py`
File ini akan otomatis:
1. Download dataset dari Kaggle
2. EDA & visualisasi
3. Preprocessing + SMOTE
4. Training 7 model ML
5. Evaluasi & perbandingan
6. Simpan best_model.pkl + scaler + encoders
7. **Generate `app.py`** (Streamlit dashboard)

### Step 3 — Jalankan Streamlit
**Opsi A: pyngrok (paling mudah di Colab)**
```python
!pip install pyngrok
from pyngrok import ngrok
import subprocess, time

# Set token ngrok kamu (gratis di https://ngrok.com)
!ngrok authtoken YOUR_TOKEN_HERE

proc = subprocess.Popen(["streamlit","run","app.py","--server.port","8501","--server.headless","true"])
time.sleep(4)
url = ngrok.connect(8501)
print("Dashboard:", url)
```

**Opsi B: Deploy ke Streamlit Cloud (GRATIS, permanen)**
1. Push semua file ke GitHub (termasuk `app.py` dan semua `.pkl`)
2. Buka https://streamlit.io/cloud
3. Connect repo → deploy → dapat URL publik permanen!

## 🎯 Fitur Dashboard
| Halaman | Isi |
|---------|-----|
| 🏠 Beranda | Overview dataset, metric cards, pie chart distribusi |
| 📊 EDA | Histogram, boxplot, heatmap korelasi, scatter plot |
| 🤖 Prediksi | Form input lengkap → prediksi + probabilitas + saran |
| 📈 Evaluasi | Perbandingan 7 model, metodologi 10 pertemuan |

## 🏷️ Target Klasifikasi
- ✅ **Normal** — Tidak ada gangguan tidur
- ⚠️ **Insomnia** — Kesulitan tidur
- 🚨 **Sleep Apnea** — Gangguan napas saat tidur

> **Note:** Nilai `None` / `NaN` di dataset = **Normal** (tidak ada gangguan)
