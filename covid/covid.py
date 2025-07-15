#  1. GEREKLİ KÜTÜPHANELERİ YÜKLE
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

#  2. VERİYİ YÜKLE
# Kaggle veri klasörüne göre ayarlanmıştır
df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv", low_memory=False)

# 🧪 3. VERİYİ TANI: head, tail, info
print("###################################################")
print("### Head ###")
print(df.head())

print("###################################################")
print("\n### Tail ###")
print(df.tail())

print("###################################################")
print("\n### Info ###")
print(df.info())

#  4. NA DEĞERLERİ DOLDUR
print("###################################################")
print("\n### DEĞERLERİ DOLDURMA ###")
df['abstract'] = df['abstract'].fillna("")
df['title'] = df['title'].fillna("")

# 🔧 5. ÖZNİTELİK MÜHENDİSLİĞİ (Feature Engineering)
print("###################################################")
print("\n### ÖZNİTELİK MÜHENDİSLİĞİ  ###")
df['abstract_len'] = df['abstract'].apply(len)
df['title_len'] = df['title'].apply(len)
df['abstract_wordcount'] = df['abstract'].apply(lambda x: len(x.split()))
df['title_wordcount'] = df['title'].apply(lambda x: len(x.split()))

#  6. BASİT ETİKET OLUŞTURMA
# Eğer abstract içinde "vaccine" kelimesi geçiyorsa etiket = 1, geçmiyorsa 0
print("###################################################")
print("\n### BASİT ETİKET OLUŞTURMA  ###")
df['label'] = df['abstract'].str.contains("vaccine", case=False).astype(int)

#  7. MODEL GİRİŞİ: TF-IDF + basit sayısal özellikler
features = ['abstract_len', 'title_len', 'abstract_wordcount', 'title_wordcount']
X_basic = df[features]
y = df['label']

# TF-IDF vektörü çıkar
tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
X_text = tfidf.fit_transform(df['abstract'])

# Sayısal özellikleri sparse formata çevirerek TF-IDF ile birleştir
X = hstack([X_text, X_basic])

#  8. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. MODEL EĞİTİMİ (Lojistik Regresyon)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔍 10. TAHMİN ve BAŞARI DEĞERLENDİRMESİ
y_pred = model.predict(X_test)

print("###################################################")
print("\n### Accuracy ###")
print(accuracy_score(y_test, y_pred))

print("###################################################")
print("\n### Classification Report ###")
print(classification_report(y_test, y_pred))

#  11. ÖRNEK
sample_text = [
    "The development of an effective vaccine for COVID-19 remains a top priority.",
    "This paper studies the transmission patterns of coronaviruses in urban areas."
]

sample_tfidf = tfidf.transform(sample_text)
sample_basic = np.array([[len(t), len(t.split()), 0, 0] for t in sample_text])
sample_combined = hstack([sample_tfidf, sample_basic])
sample_pred = model.predict(sample_combined)

print("\nSample Predictions:", sample_pred)
