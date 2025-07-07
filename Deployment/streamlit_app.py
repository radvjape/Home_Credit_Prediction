import streamlit as st
import pandas as pd
import requests

st.title("Model Prediction Demo")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Input data:", df.head())

    # Reset the pointer of the uploaded file to start (important!)
    uploaded_file.seek(0)

    # Send file content as bytes, set filename explicitly
    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}

    response = requests.post(
        "https://my-fastapi-app-503260321931.europe-west3.run.app/predict/",
        files=files
    )

    if response.status_code == 200:
        preds = response.json().get("predictions", [])
        st.write("Predictions:")
        st.table(preds)
    else:
        st.error(f"Prediction failed: {response.text}")
