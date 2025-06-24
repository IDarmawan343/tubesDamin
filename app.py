import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# Memuat model dan scaler
# ================================
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# ================================
# Judul Halaman
# ================================
st.title("💹 Prediksi GDP per Kapita Negara")
st.subheader("Berdasarkan Indikator Ekonomi Global")

st.markdown("""
Masukkan nama negara dan nilai indikator ekonominya, lalu klik tombol **Prediksi** untuk melihat estimasi GDP per kapita negara tersebut.
""")

# ================================
# Input Pengguna
# ================================
negara = st.text_input("🌍 Nama Negara", value="Indonesia")

inflasi = st.number_input("Inflasi (CPI %)", min_value=-50.0, max_value=100.0, value=2.0, step=0.1)
pengangguran = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
pertumbuhan_gdp = st.number_input("Pertumbuhan GDP Tahunan (%)", min_value=-20.0, max_value=20.0, value=2.5, step=0.1)

# ================================
# Proses Prediksi
# ================================
if st.button("🔮 Prediksi"):
    # Siapkan data input
    data_input = np.array([[inflasi, pengangguran, pertumbuhan_gdp]])
    data_terskalakan = scaler.transform(data_input)

    # Lakukan prediksi
    hasil_prediksi = model.predict(data_terskalakan)

    # Tampilkan hasil
    st.success(f"💰 Perkiraan GDP per Kapita untuk **{negara}** adalah **${hasil_prediksi[0]:,.2f} USD**")

    st.markdown("---")
    st.markdown("📌 **Catatan**: Prediksi ini didasarkan pada model Linear Regression yang dilatih menggunakan data World Bank tahun 2010–2025.")

# ================================
# Footer
# ================================
st.markdown("---")
st.caption("📘 Dibuat untuk Tugas Besar Mata Kuliah Data Mining 2025 - Kelompok Anda")
