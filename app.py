import streamlit as st 
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model_regresi.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Prediksi Harga Mobil BMW dan Segmentasi Pasar")
st.markdown("""
MEMPREDIKSI HARGA MOBIL BMW DI UNITED KINGDOM (UK) (EU) Europa Union
""")

st.sidebar.header("Masukkan Fitur Mobil")

year = st.sidebar.slider("Tahun", 1990, 2020, 2015)
mileage = st.sidebar.number_input("Jarak Tempuh (Mileage)", value=30000)
tax = st.sidebar.number_input("Pajak (£)", value=150)
l_per_100km = st.sidebar.number_input("Efisiensi BBM (Liter per 100km)", value=7.0)
engineSize = st.sidebar.number_input("Ukuran Mesin (CC)", value=2.0)
transmission = st.sidebar.selectbox("Transmisi", ('Manual', 'Automatic', 'Semi-Auto', 'Other'))
fuelType = st.sidebar.selectbox("Jenis Bahan Bakar", ('Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other'))


mpg = 282.481 / l_per_100km

transmission_dict = {'Manual': 2, 'Automatic': 0, 'Semi-Auto': 3, 'Other': 1}
fuel_dict = {'Petrol': 4, 'Diesel': 1, 'Hybrid': 2, 'Electric': 0, 'Other': 3}

transmission_enc = transmission_dict.get(transmission, 1)
fuel_enc = fuel_dict.get(fuelType, 3)

input_data = pd.DataFrame([[year, mileage, tax, mpg, engineSize, transmission_enc, fuel_enc]],
                          columns=['year', 'mileage', 'tax', 'mpg', 'engineSize', 'transmission', 'fuelType'])

input_numerik = input_data[['year', 'mileage', 'tax', 'mpg', 'engineSize']]
input_kategori = input_data[['transmission', 'fuelType']]

input_scaled_numerik = scaler.transform(input_numerik)

input_final = np.concatenate([input_scaled_numerik, input_kategori.values], axis=1)

predicted_price_gbp = model.predict(input_final)[0]

kurs = 20000
predicted_price_idr = predicted_price_gbp * kurs

st.subheader("Hasil Prediksi Harga Mobil")
st.write(f" Harga Perkiraan (poundsterling): **£ {predicted_price_gbp:,.2f}**")
st.write(f" Harga Perkiraan (Rupiah): **Rp {predicted_price_idr:,.0f}**")

# Footer
st.markdown("---")
st.markdown("Tugas Besar Data Mining")
