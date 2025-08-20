from fastapi import FastAPI
import pandas as pd
import joblib
import os
import numpy as np  # əlavə etdik

# Backend FastAPI server
app = FastAPI(title="ML Prediction API")

MODEL_PATH = os.path.join("models", "model.pkl")

model = joblib.load(MODEL_PATH)

# Root endpoint
@app.get("/")
def root():
    return {"message": "ML API is running"}

# Predict endpoint
@app.post("/predict")
def predict():
    df = pd.read_parquet("data/external/multisim_dataset.parquet")
    
    df.drop(columns=["telephone_number"], inplace=True, errors="ignore")
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    if "tenure" in df.columns:
        df["tenure_years"] = df["tenure"] / 365
    if "age_dev" in df.columns and "tenure" in df.columns:
        df["age_dev"] = pd.to_numeric(df["age_dev"], errors="coerce")
        df["device_age_ratio"] = df["age_dev"] / (df["tenure"] + 1)
    if "dev_man" in df.columns and "device_os_name" in df.columns:
        df["device_man_os"] = df["dev_man"].astype(str) + "_" + df["device_os_name"].astype(str)

    preds = model.predict(df)

    # NaN və inf dəyərləri təmizləyirik
    preds_clean = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0).tolist()

    return {"predictions": preds_clean}
