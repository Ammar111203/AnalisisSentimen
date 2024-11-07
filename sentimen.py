import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from itertools import islice
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

plt.ion()  # Mengaktifkan plotting interaktif

# Memuat dataset
df = pd.read_csv('youtube_comments.csv')  # Memuat dataset dari file CSV

# Mengonversi semua nilai di kolom 'Comment' menjadi string dan menangani nilai NaN
df['Comment'] = df['Comment'].astype(str).fillna('')

# Inisialisasi stemmer dan penghapus stopword
stemmer = StemmerFactory().create_stemmer()  # Stemmer untuk bahasa Indonesia
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()  # Penghapus stopword
additional_words = {'yang', 'saya', 'aja', 'nya', 'ini', 'itu', 'ya', 'yg', 'bang', 'dan', 'untuk', 'pada','gw','juga','jg','lha','lah','kan'}

# Fungsi normalisasi slang
def convert_to_slang(review):
    try:
        # Memuat kamus slang dari file jika ada
        with open("slangwords.txt") as f:
            slang_dict = eval(f.read())
    except FileNotFoundError:
        # Kamus slang default jika file tidak ditemukan
        slang_dict ={
            'gak': 'tidak',
            'gk' : 'tidak',
            'tdk' : 'tidak',
            'nggak': 'tidak',
            'kalo': 'kalau',
            'makasih': 'terima kasih',
            'sama': 'dengan',
            'tq': 'terima kasih',
            'dgn': 'dengan',
            'dlm': 'dalam',
            'bg': 'bang',
            'udh': 'sudah',
            'tp' :'tapi' , 
            'jgn' : 'jangan',
            'oon' : 'bodoh',
            'bnr' : 'benar',
            'mentri' : 'menteri',
        }
 
    # Mengganti kata slang dengan bentuk normal
    pattern = re.compile(r'\b(' + '|'.join(slang_dict.keys()) + r')\b')
    return pattern.sub(lambda x: slang_dict.get(x.group(), x.group()), review)

# Fungsi pembersihan teks
def clean_text(text):
    text = text.lower()  # Mengubah menjadi huruf kecil
    text = re.sub(r'[^a-z\s]', '', text)  # Menghapus karakter non-alfabet
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebihan
    words = text.split()
    return ' '.join([word for word in words if word not in additional_words])  # Menghapus kata tambahan umum

# Fungsi pembersihan teks lengkap
def preprocess_text(text):
    text = convert_to_slang(text)  # Normalisasi slang
    text = clean_text(text)  # Pembersihan teks
    text = stopword_remover.remove(text)  # Menghapus stopwords
    text = stemmer.stem(text)  # Stemming
    words = text.split()
    return ' '.join([word for word in words if word not in additional_words])  # Teks akhir yang telah dibersihkan

# Inisialisasi SentimentIntensityAnalyzer untuk analisis sentimen
sia = SentimentIntensityAnalyzer()

# Fungsi untuk mengklasifikasikan sentimen berdasarkan skor polaritas VADER
def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:  # Sentimen positif
        return 'Positif'
    elif sentiment_score['compound'] <= -0.05:  # Sentimen negatif
        return 'Negatif'
    else:  # Sentimen netral
        return 'Netral'

# Menerapkan preprocessing pada kolom 'Comment'
df['processed_comment'] = df['Comment'].apply(preprocess_text)

# Menerapkan analisis sentimen pada komentar yang telah diproses
df['Sentiment'] = df['processed_comment'].apply(get_sentiment)

# Mengecek apakah kolom Sentiment berhasil ditambahkan
print(df[['processed_comment', 'Sentiment']].head())

# 1. Informasi Dasar Dataset
print("1. Informasi Dataset:")
print(df.info())  # Menampilkan informasi dasar tentang dataset
print("\nJumlah komentar:", len(df))  # Jumlah komentar
print("Komentar unik:", df['Comment'].nunique())  # Menghitung komentar unik
print("\n" + "="*50 + "\n")

# 2. Menganalisis Panjang Komentar yang Telah Diproses
df['review_length'] = df['processed_comment'].str.len()  # Menghitung panjang karakter setiap komentar
df['word_count'] = df['processed_comment'].str.split().str.len()  # Menghitung jumlah kata setiap komentar

print("2. Statistik Panjang Review Setelah Preprocessing:")
print(df[['review_length', 'word_count']].describe())  # Menampilkan statistik tentang panjang komentar dan jumlah kata
print("\n" + "="*50 + "\n")

# 3. Visualisasi Distribusi Panjang Review
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='review_length', bins=30)  # Histogram distribusi panjang review
plt.title('Distribusi Panjang Review (Karakter) Setelah Preprocessing')
plt.xlabel('Jumlah Karakter')
plt.ylabel('Frekuensi')

plt.subplot(1, 2, 2)
sns.histplot(data=df, x='word_count', bins=30)  # Histogram distribusi jumlah kata
plt.title('Distribusi Jumlah Kata per Review Setelah Preprocessing')
plt.xlabel('Jumlah Kata')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.savefig('distribusi_kata.png')  # Menyimpan plot sebagai gambar
plt.close()

# Ekstraksi Fitur menggunakan Bag of Words (BoW)
vectorizer = CountVectorizer(max_features=5000)  # Membatasi fitur maksimal menjadi 5000 kata
X = vectorizer.fit_transform(df['processed_comment']).toarray()  # Mengubah komentar menjadi vektor fitur

# Asumsikan dataset memiliki kolom target 'Label' untuk diprediksi (sesuaikan ini dengan dataset Anda)
# Di sini, digunakan label dummy berdasarkan keberadaan kata-kata positif/negatif sebagai contoh
df['Label'] = df['processed_comment'].apply(lambda x: 1 if 'positif' in x else 0)  # Label berdasarkan kata kunci

y = df['Label']  # Variabel target

# Membagi dataset menjadi data latih dan uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Membuat model KNN dengan 5 tetangga terdekat
knn.fit(X_train, y_train)  # Melatih model

# Prediksi pada data uji
y_pred = knn.predict(X_test)

# Evaluasi kinerja model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy:.4f}")  # Menampilkan nilai akurasi
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))  # Menampilkan laporan klasifikasi

# Visualisasi Word Cloud dari komentar yang telah diproses
wordcloud = WordCloud(width=800, height=400, 
                      background_color='white', 
                      collocations=False,
                      max_words=100).generate_from_frequencies(Counter(' '.join(df['processed_comment']).split()))  # Membuat word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud dari Komentar yang Telah Diproses')
plt.savefig('wordcloud_reviews.png', bbox_inches='tight', dpi=300)  # Menyimpan gambar word cloud
print("\nWord Cloud Berhasil Dibuat")
# plt.show()
# plt.close()

# Menyimpan data yang telah diproses ke file CSV
output_path = 'preprocessed_youtube_comments.csv'
df.to_csv(output_path, index=False)  # Menyimpan DataFrame ke CSV

# Visualisasi Distribusi Sentimen
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df, palette='Set2')  # Countplot untuk distribusi sentimen
plt.title('Distribusi Sentimen Komentar')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Komentar')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')  # Menyimpan gambar distribusi sentimen
print("\nSentimen Berhasil Dibuat")
# plt.show()
# plt.close()
