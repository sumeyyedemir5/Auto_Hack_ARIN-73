import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from keras.layers import Dense
# Veri Toplama ve Analiz
def veri_toplama_ve_analiz():
    veri_kumesi_ilk = pd.read_csv('carsData.csv')
    veri_kumesi = pd.get_dummies(veri_kumesi_ilk, columns=['Car','Country'])
    return veri_kumesi

# Generative AI Modeli
def generative_ai_modeli(ozellik_sayisi):
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=ozellik_sayisi, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(layers.Dense(ozellik_sayisi, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Eğitim İşlemi
def egitim(model, veri_kumesi, epochs=5):
    ozellik_sayisi = len(veri_kumesi.columns)
    X = veri_kumesi.drop('Price', axis=1)  # 'price' sütunu tahmin edilecek hedef değişken
    y = veri_kumesi['Price']

    # NumPy dizilerine dönüştür
    X = np.array(X)
    y = np.array(y)
    model.fit(X, y, epochs=epochs)

# Sonuç Raporları ve Analiz
def sonuc_raporlari_ve_analiz(model, veri_kumesi):
    for index, tasarim in veri_kumesi.iterrows():
        tahmin = model.predict(tasarim.drop('Price'))  # 'price' sütunu tahmin edilecek hedef değişken
        print(f"Gerçek Fiyat: {tasarim['Price']}, Tahmini Fiyat: {tahmin}")

# Ana Program
ozellik_sayisi = 10
veri_kumesi = veri_toplama_ve_analiz()
model = generative_ai_modeli(ozellik_sayisi)

egitim(model, veri_kumesi)

sonuc_raporlari_ve_analiz(model, veri_kumesi)

model.save('generative_ai_modeli.h5')
