# 🚀 Müşteri Deneyimi Analizi Projesi

> Müşteri memnuniyetini artırmak için veri odaklı içgörüler.



## 🎯 Proje Amacı

Müşteri memnuniyetini etkileyen faktörleri veri analiziyle ortaya koymak, iş stratejilerine yön vermek.

---

## 📊 Veri Seti

Bu proje için kullanılan veri seti müşteri deneyimlerini ve memnuniyet faktörlerini ölçmek üzere hazırlanmıştır. İçerisinde hem sayısal hem kategorik değişkenler bulunmaktadır:

- **CustomerID:** Müşteri benzersiz kimliği  
- **Gender:** Cinsiyet (Male/Female)  
- **Age:** Yaş  
- **Annual Income :** Yıllık gelir (bin $ cinsinden)  
- **Spending Score (1-100):** Harcama skoru  
- **Satisfaction Score (1-100):** Memnuniyet skoru  
- **Loyalty Status:** Sadakat durumu (Gold, Silver, Bronze)  
- **Feedback:** Geri bildirim (Positive, Neutral, Negative)  
- **Product Quality (1-10):** Ürün kalitesi puanı  
- **Service Quality (1-10):** Hizmet kalitesi puanı  
- **Purchase Frequency:** Satın alma sıklığı (aylık)  
- **Response Time (minutes):** Destek yanıt süresi  
- **Complain:** Şikayet durumu (Yes/No)  

Aşağıdaki Python koduyla veri seti okunup ilk birkaç satır görüntülenebilir:

```python
import pandas as pd

df = pd.read_csv('data/customer_experience_data.csv')
print(f"Veri seti {df.shape[0]} satır ve {df.shape[1]} sütundan oluşuyor.")
print(df.head())
