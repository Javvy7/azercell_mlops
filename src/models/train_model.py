import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

df = pd.read_parquet("data/external/multisim_dataset.parquet")
df.drop(columns=["telephone_number"], inplace=True)

# Feature Engineering
df["tenure_years"] = df["tenure"] / 365
df["age_dev"] = pd.to_numeric(df["age_dev"], errors="coerce")
df["device_age_ratio"] = df["age_dev"] / (df["tenure"] + 1)
df["device_man_os"] = df["dev_man"].astype(str) + "_" + df["device_os_name"].astype(str)

X = df.drop(columns=["target"])
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
num_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()


num_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[("num", num_transformer, num_cols), ("cat", cat_transformer, cat_cols)]
)

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", xgb_clf)])

# Hyperparameter tuning
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.05, 0.1, 0.2],
    "classifier__subsample": [0.8, 1],
    "classifier__colsample_bytree": [0.8, 1],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", verbose=2, n_jobs=1)

grid_search.fit(X_train, y_train)
print("✅ Best Parameters:", grid_search.best_params_)

joblib.dump(grid_search.best_estimator_, "models/model.pkl")
print("✅ Model saved to models/model.pkl")
