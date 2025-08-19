import joblib
import pandas as pd


def main():
    model = joblib.load("models/model.pkl")

    df = pd.read_parquet("data/external/multisim_dataset.parquet")
    df.drop(columns=["telephone_number", "target"], errors="ignore", inplace=True)

    df["tenure_years"] = df["tenure"] / 12
    df["device_age_ratio"] = df["age_dev"]
    df["device_man_os"] = df["dev_man"] + "_" + df["device_os_name"]

    preds = model.predict(df)
    print("sample predictions:", preds[:20])


if __name__ == "__main__":
    main()
