from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd
app = Flask(__name__)

# Modeli yükle
model = tf.keras.models.load_model('generative_ai_modeli.h5')  # Modelinizi kaydettiğiniz dosyanın adını ve yolunu belirtin

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_design', methods=['POST'])
def generate_design():
    hedefler = request.form['hedefler']
    stil_tercihleri = request.form['stil_tercihleri']
    teknik_gereksinimler = request.form['teknik_gereksinimler']

    kullanici_girdileri = np.array([[hedefler, stil_tercihleri, teknik_gereksinimler]])
    yeni_tasarim = model.predict(kullanici_girdileri)

    # Tasarımı kullanıcıya gösterme ve geri bildirim alınması
    geri_bildirim = "Yeni tasarım değerlendirildi."

    return render_template('index.html', yeni_tasarim=yeni_tasarim, geri_bildirim=geri_bildirim)

if _name_ == '_main_':
    app.run(debug=True)