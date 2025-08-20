import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="ML Prediction App", layout="wide")
st.title("ML Prediction Web App")

st.markdown("""
Upload your dataset (Parquet file) and get predictions from the backend ML model.
""")

uploaded_file = st.file_uploader("Choose a Parquet file", type="parquet")

if uploaded_file is not None:
    df = pd.read_parquet(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        try:
            # Backend URL
            backend_url = "http://backend:8000/predict"  
            response = requests.post(backend_url)
            if response.status_code == 200:
                preds = response.json()["predictions"]
                st.subheader("Predictions")
                st.write(preds[:20])  
            else:
                st.error(f"Backend error: {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
