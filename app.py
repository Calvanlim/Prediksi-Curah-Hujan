# File: app.py

import streamlit as st
import joblib
import pandas as pd

# Load model, scaler, dan features yang disimpan
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

# Judul Aplikasi
st.title("Prediksi Curah Hujan")
st.write("Aplikasi ini memprediksi curah hujan berdasarkan parameter cuaca yang Anda input.")

# Sidebar untuk input parameter cuaca dari pengguna
st.sidebar.header("Input Parameter Cuaca")

# Inisialisasi dictionary untuk menyimpan input pengguna
user_input = {}

# Ambil input pengguna untuk setiap fitur
for feature in features:
    user_input[feature] = st.sidebar.number_input(
        label=f"{feature}", 
        min_value=0.0,  # Menggunakan nilai minimum yang wajar 
        value=0.0  # Nilai default
    )

# Konversi input menjadi DataFrame
input_data = pd.DataFrame([user_input])

# Standarisasi input data menggunakan scaler yang disimpan
input_data_scaled = scaler.transform(input_data)

# Prediksi menggunakan model
prediction = model.predict(input_data_scaled)

# Tampilkan hasil prediksi
st.subheader("Prediksi Curah Hujan")
st.write(f"Curah hujan yang diprediksi adalah: {prediction[0]:.2f} mm")
