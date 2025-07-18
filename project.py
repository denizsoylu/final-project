"""
Bu veri seti müşteri deneyimlerini anlamaya yönelik hazırlanmıştır.
Amaç; müşteri memnuniyetini etkileyen faktörleri analiz etmek, deneyim kalitesini ölçmek ve müşteri davranışlarını anlamaktır.
"""

"""
| Değişken                       | Açıklama                                             |
| ------------------------------ | ---------------------------------------------------- |
| CustomerID**                   | Müşteri Kimliği (benzersiz ID)                       |
| Gender**                       | Cinsiyet (Male / Female)                             |
| Age**                          | Yaş                                                  |
| Annual Income                  | Yıllık gelir                                         |
| Spending Score (1-100)         | Harcama Skoru (1=az, 100=çok)                        |
| Satisfaction Score (1-100)     | Memnuniyet Skoru                                          |
| Loyalty Status                 | Sadakat Durumu (Gold, Silver, Bronze)                |
| Feedback**                     | Müşteri geri bildirimi (Positive, Neutral, Negative) |
| Product Quality (1-10)         | Ürün Kalitesi Puanı                                  |
| Service Quality (1-10)         | Hizmet Kalitesi Puanı                                |
| Purchase Frequency**           | Satın Alma Sıklığı (aylık)                           |
| Response Time (minutes)        | Destek Yanıt Süresi (dakika)                         |
| Complain*                      | Şikayet (Yes / No)                                   |

"""

# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Tüm sütunları göstermek için
pd.set_option('display.max_columns', None)

# Tüm satırları göstermek için
pd.set_option('display.max_rows', None)
print("###############################################################")

# Grafik ayarları
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")
print("###############################################################")

# =============================================
# 1. Veri Setini Yükleme
# =============================================

# CSV dosyasının yolu (PyCharm için yerel yol)
file_path = r'C:\Users\HP\PycharmProjects\Final-Project1\customer_experience_data.csv'

print("###############################################################")

# Veri setini oku
df = pd.read_csv(file_path)
print(f"Veri {df.shape[0]} satır ve {df.shape[1]} sütundan oluşuyor.")

# İlk 5 satırı görüntüle
print(df.head())

# Sütun isimlerini yazdır
print("Sütunlar:", df.columns.tolist())

# Veri yapısı ve eksik değer durumu
print(df.info())
print("###############################################################")

# =============================================
# 2. Değişken Tiplerini Kontrol Etme
# =============================================
for col in df.columns:
    print(f"{col}: {df[col].dtype}")
print("###############################################################")

# =============================================
# 3. İstatistiksel Özet
# =============================================
print("###############################################################")

# Sayısal değişkenler için özet istatistikler
print(df.describe().T)
print("###############################################################")

# Kategorik değişkenlerin dağılımları
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    print(f"\n{col} Dağılımı:")
    print(df[col].value_counts())
print("###############################################################")

# =============================================
# 4. Eksik Değer Analizi
# =============================================

# Eksik değer sayısı ve yüzdesi
missing = df.isnull().sum()
print("\nEksik Değer Sayıları:\n", missing[missing > 0])
print("\nEksik Değer Yüzdesi:\n", (missing / len(df) * 100).round(2))
print("###############################################################")

# Eksik değer görselleştirme
sns.heatmap(df.isnull(), cbar=False)
plt.title('Eksik Değer Isı Haritası')
plt.show()
print("###############################################################")

# =============================================
# 5. Aykırı Değer Analizi
# =============================================

# Sayısal sütunları seç
num_cols = df.select_dtypes(include=[np.number]).columns
print("###############################################################")

# Z-score ile aykırı değer tespiti
z_scores = np.abs(stats.zscore(df[num_cols].dropna()))
outlier_mask = (z_scores > 3)
outlier_rows = np.where(outlier_mask.any(axis=1))[0]
print(f"Toplam aykırı gözlem sayısı: {len(outlier_rows)}")
print("###############################################################")

# Boxplot ile aykırı değer görselleştirme
df[num_cols].boxplot(rot=45)
plt.title('Aykırı Değer Boxplot')
plt.show()
print("###############################################################")

# =============================================
# 6. Görselleştirme
# =============================================

# Sayısal değişkenler için histogram
for col in num_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'{col} Dağılımı')
    plt.xlabel(col)
    plt.ylabel('Frekans')
    plt.show()
print("###############################################################")

# Kategorik-Nümerik İlişki Görselleştirme
for cat in cat_cols:
    for num in num_cols:
        plt.figure()
        sns.boxplot(x=cat, y=num, data=df)
        plt.title(f'{cat} göre {num}')
        plt.xlabel(cat)
        plt.ylabel(num)
        plt.show()
print("###############################################################")

# Korelasyon matrisi ve heatmap
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Sayısal Değişkenler Korelasyon Matrisi')
plt.show()
print("###############################################################")


