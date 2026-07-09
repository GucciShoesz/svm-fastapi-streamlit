import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="wide"
)

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Breast Cancer Prediction")

st.write("Masukkan 30 fitur untuk melakukan prediksi.")

inputs = []

for i in range(30):
    value = st.number_input(
        f"Feature {i+1}",
        value=0.0,
        format="%.6f"
    )
    inputs.append(value)

if st.button("Predict"):

    x = np.array(inputs).reshape(1, -1)

    x = scaler.transform(x)

    prediction = model.predict(x)[0]

    probability = model.predict_proba(x).max()

    if prediction == 1:
        hasil = "Benign (Jinak)"
    else:
        hasil = "Malignant (Ganas)"

    st.success(f"Prediction : {hasil}")

    st.metric(
        "Probability",
        f"{probability:.2%}"
    )