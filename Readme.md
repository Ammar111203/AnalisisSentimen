# Proyek Analisis Sentimen Komentar

## Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis sentimen komentar-komentar yang ada dalam dataset. Dengan menggunakan teknik pemrosesan bahasa alami (NLP), setiap komentar akan dianalisis untuk menentukan apakah sentimen yang terkandung bersifat **positif**, **negatif**, atau **netral**. 

Selain itu, proyek ini juga menghasilkan visualisasi data seperti distribusi panjang komentar serta kata-kata yang paling sering muncul dalam komentar dengan menggunakan **WordCloud** dan **Histogram**.

## Bahan yang Diperlukan

Untuk menjalankan proyek ini, Anda memerlukan beberapa bahan berikut:

- **Python 3.8+**
- **Libraries** yang digunakan dalam proyek ini:
  - Pandas: Untuk manipulasi dan analisis data
  - Numpy: Untuk operasi numerik
  - NLTK (Natural Language Toolkit): Untuk pemrosesan teks dan analisis sentimen
  - Seaborn: Untuk visualisasi data
  - WordCloud: Untuk membuat visualisasi kata-kata yang sering muncul
  - Scikit-learn: Untuk klasifikasi data dan algoritma pembelajaran mesin

## Instalasi

Untuk menginstal semua dependensi yang dibutuhkan dalam proyek ini, Anda hanya perlu menjalankan satu perintah berikut di terminal atau command prompt:
```bash
pip install pandas numpy nltk seaborn wordcloud scikit-learn
```

## Proses

Proses utama dalam skrip ini dibagi menjadi beberapa tahap yang bertujuan untuk mempersiapkan data, melakukan analisis sentimen, pelatihan model, dan menghasilkan visualisasi yang informatif. Berikut adalah langkah-langkahnya:

### 1. **Pemrosesan Data**
   - **Pemuatan Dataset**: Skrip dimulai dengan memuat dataset dari file CSV (`youtube_comments.csv`) yang berisi komentar-komentar di YouTube.
   - **Normalisasi Slang**: Setiap komentar diproses untuk mengganti kata-kata slang dengan bentuk yang lebih formal menggunakan kamus slang. Jika file kamus slang tidak ditemukan, maka kamus default akan digunakan.
   - **Pembersihan Teks**: Setiap komentar akan dibersihkan dengan menghapus karakter yang tidak diperlukan (seperti tanda baca, angka, dll.), mengonversi teks menjadi huruf kecil, dan menghapus kata-kata umum (seperti 'saya', 'dan', 'untuk') yang tidak memberikan informasi penting untuk analisis.
   - **Penghapusan Stopword**: Setelah pembersihan awal, stopwords (kata-kata yang sering muncul namun tidak memberikan makna penting, seperti 'yang', 'dan', dll.) dihapus menggunakan pustaka `Sastrawi` yang mendukung Bahasa Indonesia.
   - **Stemming**: Selanjutnya, stemming dilakukan untuk mengubah kata-kata menjadi bentuk dasarnya. Misalnya, kata "memperoleh" menjadi "oleh".

### 2. **Analisis Sentimen**
   - **SentimentIntensityAnalyzer**: Dengan menggunakan pustaka `VADER` dari `nltk`, setiap komentar dianalisis untuk menilai sentimen secara otomatis, apakah komentar tersebut positif, negatif, atau netral. Skor polaritas VADER digunakan untuk mengklasifikasikan sentimen komentar:
     - **Positif**: Jika skor komposit lebih besar dari 0.05.
     - **Negatif**: Jika skor komposit kurang dari -0.05.
     - **Netral**: Jika skor komposit berada di antara -0.05 dan 0.05.

### 3. **Penciptaan Fitur untuk Klasifikasi**
   - **Bag of Words (BoW)**: Teknik Bag of Words digunakan untuk mengubah setiap komentar menjadi vektor numerik. Setiap kata dalam komentar menjadi fitur, dan vektor yang dihasilkan digunakan untuk pelatihan model pembelajaran mesin.
   - **Pembagian Data**: Dataset dibagi menjadi dua bagian: data latih (80%) dan data uji (20%) untuk memastikan model dilatih dengan data yang cukup dan diuji pada data yang tidak terlihat sebelumnya.

### 4. **Pelatihan Model Klasifikasi**
   - **KNN (K-Nearest Neighbors)**: Model klasifikasi KNN digunakan untuk mengklasifikasikan komentar ke dalam dua kelas (positif/negatif) berdasarkan fitur yang dihasilkan dari teks. Model ini dilatih dengan data latih dan diuji menggunakan data uji.

### 5. **Evaluasi Model**
   - **Akurasi**: Setelah model dilatih, akurasi dari model dievaluasi dengan menghitung tingkat keberhasilan prediksi pada data uji.
   - **Laporan Klasifikasi**: Laporan klasifikasi yang berisi metrik seperti precision, recall, f1-score untuk masing-masing kelas (positif, negatif, netral) dihasilkan untuk mengevaluasi kinerja model secara lebih rinci.

### 6. **Visualisasi**
   - **Distribusi Panjang Komentar**: Visualisasi distribusi panjang komentar setelah pembersihan menggunakan histogram.
   - **Word Cloud**: Membuat visualisasi kata-kata yang sering muncul dalam komentar untuk memberikan gambaran umum tentang kata-kata yang paling sering dibahas dalam komentar.
   - **Distribusi Sentimen**: Visualisasi distribusi sentimen komentar menggunakan diagram batang yang menunjukkan jumlah komentar dalam setiap kategori sentimen (positif, negatif, netral).

### 7. **Output**
   - Hasil analisis sentimen dan komentar yang telah diproses disimpan dalam file CSV baru (`preprocessed_youtube_comments.csv`) yang berisi kolom:
     - `Comment`: Komentar asli.
     - `processed_comment`: Komentar yang telah diproses.
     - `Sentiment`: Hasil klasifikasi sentimen (positif, negatif, netral).
   - Visualisasi berupa grafik distribusi panjang komentar, word cloud, dan distribusi sentimen disimpan sebagai file gambar (`distribusi_kata.png`, `wordcloud_reviews.png`, `sentiment_distribution.png`).


#   A n a l i s i s S e n t i m e n  
 