import pandas as pd
import re
import nltk
import unicodedata

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('indonesian')) 
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def load_data(file_path):
    return pd.read_csv(file_path)

def create_sentiment_label(score):
    if score <= 2:
        return "negatif"
    elif score == 3:
        return "netral"
    else:
        return "positif"

def cleaning_text(text) :
    text = str(text)

    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def toknize(text):
    return text.split()

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stemming(tokens):
    sentence = " ".join(tokens)
    return stemmer.stem(sentence)

def preprocess_data(df):

    df = df.dropna(subset=['content'])
    df = df.drop_duplicates(subset=['content'])
    df['sentiment'] = df['score'].apply(create_sentiment_label)
    df['content_clean'] = df['content'].apply(cleaning_text)
    df['tokens'] = df['content_clean'].apply(toknize)
    df['tokens'] = df['tokens'].apply(remove_stopwords)
    df['text_final'] = df['tokens'].apply(stemming)

    df_final = df[['text_final', 'sentiment']]
    return df_final

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

def main():

    input_file = "../dana_reviews_raw.csv"
    output_file = "./preprocessing/dataset_preprocessing.csv"

    print("Memuat dataset...")
    df = load_data(input_file)

    print(f"Jumlah data awal: {len(df)}")

    df_final = preprocess_data(df)

    print(f"Jumlah data akhir: {len(df_final)}")

    save_data(df_final, output_file)

    print(f"Dataset berhasil disimpan ke: {output_file}")

    print("\nContoh hasil preprocessing:")
    print(df_final.head())

if __name__ == "__main__":
    main()