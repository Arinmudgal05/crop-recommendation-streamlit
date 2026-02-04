import streamlit as st
import numpy as np
import joblib  
import sklearn

# -------------------------------
# Load model files
# -------------------------------

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="centered")

st.title("🌾 Crop Recommendation System")
st.markdown("Predict the **best crop** based on soil and climate conditions.")

st.divider()

# -------------------------------
# Input fields
# -------------------------------
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=40)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=40)
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=60.0)
ph = st.number_input("Soil pH", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=100.0)

st.divider()

# -------------------------------
# Prediction
# -------------------------------
if st.button("🌧️ Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    crop = label_encoder.inverse_transform(prediction)

    st.success(f"✅ Recommended Crop: **{crop[0]}** 🌱")

