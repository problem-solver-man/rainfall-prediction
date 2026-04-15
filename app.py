import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------------
# Load model and scaler
# -------------------------------

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(page_title="Rainfall Predictor",
                   page_icon="🌧️", layout="centered")

st.title("🌧️ Rainfall Prediction App")
st.markdown("Predict whether it will rain based on weather conditions")

# -------------------------------
# Sidebar Inputs
# -------------------------------

st.sidebar.header("🌡️ Input Weather Data")

pressure = st.sidebar.number_input(
    "Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1013.0)
temperature = st.sidebar.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
dewpoint = st.sidebar.number_input(
    "Dew Point (°C)", min_value=-10.0, max_value=40.0, value=20.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
cloud = st.sidebar.slider("Cloud Cover (%)", 0, 100, 40)
sunshine = st.sidebar.number_input(
    "Sunshine (hours)", min_value=0.0, max_value=15.0, value=7.0)
winddirection = st.sidebar.slider("Wind Direction (°)", 0, 360, 180)
windspeed = st.sidebar.number_input(
    "Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)

# -------------------------------
# Main Panel
# -------------------------------

st.subheader("📋 Input Summary")

input_dict = {
    "Pressure": pressure,
    "Temperature": temperature,
    "Dew Point": dewpoint,
    "Humidity": humidity,
    "Cloud": cloud,
    "Sunshine": sunshine,
    "Wind Direction": winddirection,
    "Wind Speed": windspeed
}

st.write(input_dict)

# -------------------------------
# Prediction Button
# -------------------------------

if st.button("🔍 Predict Rainfall"):

    # Convert input to array (IMPORTANT: same order as training)
    # input_data = np.array([[pressure, temperature, dewpoint,
    #                       humidity, cloud, sunshine, winddirection, windspeed]])
    input_data = pd.DataFrame([{
        "pressure": pressure,
        "temparature": temperature,
        "dewpoint": dewpoint,
        "humidity": humidity,
        "cloud": cloud,
        "sunshine": sunshine,
        "winddirection": winddirection,
        "windspeed": windspeed
    }])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    # Progress bar
    st.progress(int(probability * 100))

    # Output result
    if prediction == 1:
        st.error(f"🌧️ Rain Expected")
        st.write(f"Confidence: {probability:.2%}")
    else:
        st.success(f"☀️ No Rain Expected")
        st.write(f"Confidence: {(1 - probability):.2%}")

    # Additional Visualization
    st.subheader("📈 Probability Breakdown")

    st.bar_chart({
        "No Rain": [1 - probability],
        "Rain": [probability]
    })


# -------------------------------
# Footer
# -------------------------------

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
