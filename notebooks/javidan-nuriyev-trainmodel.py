#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix




bucket_name = 'dataminds-warehouse'
s3_file_key = 'multisim_dataset.parquet'            # e.g. 'folder/myfile.txt'
local_file_path = 'multisim_dataset.parquet'        # Local destination

# Create an S3 client (remove `bucket_name` here — not a valid argument for boto3.client)
s3 = boto3.client(
    's3',
    region_name='us-east-1'
    # aws_access_key_id='your_access_key',
    # aws_secret_access_key='your_secret_key'
)

# Download the file
try:
    s3.download_file(bucket_name, s3_file_key, local_file_path)
    print(f"✅ File downloaded successfully from s3://{bucket_name}/{s3_file_key} to {local_file_path}")
except Exception as e:
    print("❌ Error downloading file:", e)


# In[3]:


df = pd.read_parquet(local_file_path)


# In[4]:


df


# In[5]:


# eda
print("shape:", df.shape)


# In[6]:


print("\nMissing Values:\n", df.isna().sum().sort_values(ascending=False).head(15))


# In[7]:


sns.countplot(x='target', data=df)
plt.title("Target distribution")
plt.show()


# In[8]:


print(list(df.columns))


# In[9]:


df.isnull().sum()


# In[10]:


df.drop(columns=['telephone_number'], inplace=True)


# In[11]:


# separate categorical and numeric columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(exclude=['object']).columns.tolist()
num_cols.remove('target')


# In[12]:


# FE
df['tenure_years'] = df['tenure'] / 365


# In[13]:


df['age_dev'] = pd.to_numeric(df['age_dev'], errors='coerce')


# In[14]:


# device age to tenure ratio
df['device_age_ratio'] = df['age_dev'] / (df['tenure'] + 1)  # +1 to avoid division by zero


# In[15]:


# combine device manufacturer + OS
df['device_man_os'] = df['dev_man'].astype(str) + "_" + df['device_os_name'].astype(str)


# In[16]:


X = df.drop(columns=['target'])
y = df['target']





# In[18]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)


# In[19]:


cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()



# In[21]:


# Pipelines
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])



# In[23]:


# full pipeline with XGBoost
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb_clf)
])




# In[25]:


# hyperparameter tuning with GridSearchCV
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.05, 0.1, 0.2],
    "classifier__subsample": [0.8, 1],
    "classifier__colsample_bytree": [0.8, 1]
}


# In[26]:


grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


# In[27]:


print("Best Parameters:", grid_search.best_params_)


# In[29]:


final_model = XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42
)


# In[30]:


# applying fit_transform for preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

final_model.fit(X_train_processed, y_train)
y_pred = final_model.predict(X_test_processed)




# In[32]:


print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[33]:


best_model = grid_search.best_estimator_


# In[34]:


xgb_model = best_model.named_steps['classifier']
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title("Top 15 Feature Importances")
plt.show()


# In[35]:



# Replace with your actual credentials and info
bucket_name = 'dataminds-homeworks'
s3_file_key = 'javidan-nuriyev-fe2.ipynb' 
local_file_path = 'javidan-nuriyev-fe2.ipynb'

# Create an S3 client
s3 = boto3.client(
    's3'
)

# Upload the file
try:
    s3.upload_file(local_file_path, bucket_name, s3_file_key)
    print(f"File uploaded successfully to s3://{bucket_name}/{s3_file_key}")
except Exception as e:
    print("Error uploading file:", e)

 


# In[ ]:




