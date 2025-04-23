import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Cement Strength Predictor", layout="wide")

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🧱 Cement Strength Prediction Dashboard")

with st.form("input_form"):
    st.subheader("Input Cement Mix Components")
    cement = st.slider("Cement (kg/m³)", 100, 600, 300)
    slag = st.slider("Slag (kg/m³)", 0, 300, 50)
    fly_ash = st.slider("Fly Ash (kg/m³)", 0, 200, 30)
    water = st.slider("Water (kg/m³)", 120, 250, 160)
    superplasticizer = st.slider("Superplasticizer (kg/m³)", 0, 30, 5)
    coarse_agg = st.slider("Coarse Aggregate (kg/m³)", 800, 1200, 1000)
    fine_agg = st.slider("Fine Aggregate (kg/m³)", 500, 800, 600)
    age = st.selectbox("Age of Concrete (days)", [1, 3, 7, 14, 28, 90, 180, 365])

    submitted = st.form_submit_button("Predict Strength")

if submitted:
    input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    st.success(f"Predicted Cement Strength: {prediction[0]:.2f} MPa")