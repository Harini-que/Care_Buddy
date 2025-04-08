import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and encoders
model = load_model("risk_model.h5")
scaler = joblib.load("scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

st.set_page_config(page_title="Care Buddy", page_icon="🩺", layout="centered")

st.title("💖 Care Buddy: Health Risk Checker")

st.markdown("Give your vitals, and let Care Buddy guide you toward better well-being! 🌈")

# User input fields
gender = st.selectbox("Gender", ["Male", "Female"])
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=75)
temperature = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=36.7)
resp_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=40, value=18)
bmi = st.number_input("Derived BMI", min_value=10.0, max_value=50.0, value=22.0)
hrv = st.number_input("Derived HRV", min_value=0.0, max_value=0.2, value=0.08)
stress = st.number_input("Stress Index", min_value=0.0, max_value=10.0, value=3.0)

if st.button("🩺 Analyze My Risk"):
    input_data = {
        "Gender": gender,
        "Heart Rate": heart_rate,
        "Body Temperature": temperature,
        "Respiratory Rate": resp_rate,
        "Derived_BMI": bmi,
        "Derived_HRV": hrv,
        "Stress Index": stress
    }

    # Encode Gender
    input_data["Gender"] = gender_encoder.transform([input_data["Gender"]])[0]

    # Convert to DataFrame and scale
    input_df = pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input, verbose=0)[0][0]
    predicted_label = "🟥 High Risk" if prediction > 0.5 else "🟩 Low Risk"

    # Feedback
    feedback = []
    if hrv > 0.1:
        feedback.append("🧘 You seem calm and well-regulated (HRV is healthy).")
    elif hrv > 0.07:
        feedback.append("😌 You’re doing okay, but some stress may be present.")
    else:
        feedback.append("⚠️ High stress detected. Try relaxation or deep breathing.")

    if bmi < 18.5:
        feedback.append("🍽️ You're underweight. Consider nutritional support.")
    elif bmi > 30:
        feedback.append("🩺 High BMI – it may be beneficial to consult a health professional.")
    else:
        feedback.append("✅ BMI is within a healthy range.")

    if temperature < 36.1:
        feedback.append("🥶 Low body temperature – possible hypothermia.")
    elif temperature > 37.5:
        feedback.append("🌡️ High body temperature – possible fever.")
    else:
        feedback.append("🌈 Body temperature is within normal range.")

    if heart_rate < 60:
        feedback.append("💓 Heart rate is low (bradycardia) – monitor closely.")
    elif heart_rate > 100:
        feedback.append("💔 Heart rate is high (tachycardia) – possible stress or illness.")
    else:
        feedback.append("💖 Heart rate is within normal range.")

    # Show results
    st.subheader("📊 Predicted Risk")
    st.success(predicted_label)

    st.subheader("🧠 Sensor-Based Feedback")
    for f in feedback:
        st.write(f)
