# Analisis Sentimen Ulasan Pengguna DANA

## Deskripsi Proyek

Proyek ini bertujuan untuk melakukan analisis sentimen terhadap ulasan pengguna aplikasi DANA yang diperoleh dari Google Play Store. Ulasan diklasifikasikan ke dalam tiga kategori sentimen, yaitu:

* Negatif (rating 1-2)
* Netral (rating 3)
* Positif (rating 4-5)

## Dataset

Dataset diperoleh melalui scraping Google Play Store menggunakan library `google-play-scraper`.

Jumlah data: 5000 ulasan

Atribut yang digunakan:

* `content` : isi ulasan pengguna
* `score` : rating yang diberikan pengguna


## Tahapan Preprocessing

Tahapan preprocessing yang dilakukan meliputi:

1. Missing Value Handling
2. Case Folding
3. Cleaning Text
4. Tokenizing
5. Stopword Removal
6. Stemming



Hasil preprocessing akan disimpan dalam file:

```text
dataset_preprocessing.csv
```
