# Analisis Sentimen Ulasan Pengguna DANA

## Tentang Proyek

Pada proyek ini saya melakukan analisis sentimen terhadap ulasan pengguna aplikasi DANA yang diperoleh dari Google Play Store. Tujuannya adalah untuk mengetahui apakah sebuah ulasan memiliki sentimen positif, netral, atau negatif.

Klasifikasi sentimen dilakukan berdasarkan rating yang diberikan pengguna:

* Rating 1-2 → Negatif
* Rating 3 → Netral
* Rating 4-5 → Positif

Dataset yang digunakan berjumlah 5000 ulasan pengguna.

## Dataset

Data dikumpulkan menggunakan library `google-play-scraper`.

Kolom yang digunakan pada proses analisis:

* `content` : isi ulasan pengguna
* `score` : rating yang diberikan pengguna

## Tahap Preprocessing

Sebelum digunakan untuk pelatihan model, data terlebih dahulu melalui beberapa tahapan preprocessing:

1. Menghapus data kosong (missing value)
2. Mengubah teks menjadi huruf kecil (case folding)
3. Membersihkan karakter yang tidak diperlukan
4. Tokenisasi
5. Stopword removal
6. Stemming

Hasil preprocessing disimpan dalam file:

```text
dataset_preprocessing.csv
```

## Model yang Digunakan

Model yang digunakan adalah Logistic Regression dengan representasi fitur TF-IDF.

Agar proses transformasi teks dan prediksi dapat dilakukan dalam satu model, saya menggunakan Scikit-Learn Pipeline.

## Experiment Tracking

Proses pelatihan model dicatat menggunakan MLflow. Informasi seperti parameter, metrik evaluasi, dan artefak model disimpan sehingga proses eksperimen dapat ditelusuri kembali dengan lebih mudah.

## Model Serving

Model yang telah dilatih kemudian dijalankan sebagai service menggunakan MLflow Model Serving.

Contoh menjalankan model menggunakan Docker:

```bash
docker run --name dana-sentiment -p 5005:8000 willyworksid/dana-sentiment:latest serve --host 0.0.0.0 --port 8000
```

Endpoint prediksi:

```text
POST http://localhost:5005/invocations
```

Contoh data yang dapat digunakan untuk pengujian:

```json
{
  "inputs": [
    "Aplikasi DANA sangat membantu dan mudah digunakan"
  ]
}
```

## Monitoring dan Alerting

Untuk memantau performa sistem, saya menggunakan Prometheus dan Grafana.

Beberapa metrik yang dipantau antara lain:

* Total request
* Request latency
* Throughput
* Penggunaan CPU
* Penggunaan RAM

Selain monitoring, Grafana juga dikonfigurasi untuk mengirimkan notifikasi ke Telegram ketika penggunaan CPU atau RAM melewati batas yang telah ditentukan.

## Kesimpulan

Model yang dibangun mampu melakukan klasifikasi sentimen terhadap ulasan pengguna DANA ke dalam kategori positif, netral, dan negatif. Seluruh proses mulai dari pelatihan model, model serving, monitoring, hingga alerting telah diimplementasikan pada proyek ini.

## Siswa

Willy Ganjar Saputra
