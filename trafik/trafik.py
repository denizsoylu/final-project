#  1. GEREKLİ KÜTÜPHANELER
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

# 2. VERİ SETİNİ YÜKLE
df = pd.read_csv("/kaggle/input/2000-16-traffic-flow-england-scotland-wales/DfTRoadSafety_2000_2016.csv", low_memory=False)

# 3. VERİYİ TANI: head, tail, info, describe
print(df.head())
print(df.tail())
print(df.info())
print(df.describe(include='all'))

#  4. Temizlik & öznitelik mühendisliği
df['Accident_Severity'] = df['Accident_Severity'].astype(str)
df['Longitude'] = df['Longitude'].fillna(df['Longitude'].mean())
df['Latitude'] = df['Latitude'].fillna(df['Latitude'].mean())
df['Day'] = pd.to_datetime(df['Date'], dayfirst=True).dt.day
df['Month'] = pd.to_datetime(df['Date'], dayfirst=True).dt.month
df['Year'] = pd.to_datetime(df['Date'], dayfirst=True).dt.year

# 5. Basit etiket: ölümcül kaza mı?
df['is_fatal'] = (df['Accident_Severity'] == '1').astype(int)

#  6. Özelliklerin seçilmesi
features = ['Longitude', 'Latitude', 'Day', 'Month', 'Year', 'Number_of_Vehicles', 'Number_of_Casualties']
X = df[features]
y = df['is_fatal']

#  7. Modelleri hazırlayıp bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Lojistik regresyon
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#  9. Değerlendirme
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#  10. Örnek tahmin
sample = pd.DataFrame([{
    'Longitude': -0.1, 'Latitude': 51.5,
    'Day': 15, 'Month': 7, 'Year': 2010,
    'Number_of_Vehicles': 2, 'Number_of_Casualties': 1
}])
print("Sample fatal prediction:", model.predict(sample))
