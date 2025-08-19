#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import boto3
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score





bucket_name = 'dataminds-warehouse'
s3_file_key = 'ramen-ratings.csv'            # e.g. 'folder/myfile.txt'
local_file_path = 'ramen-ratings.csv'        # Local destination

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


df = pd.read_csv("ramen-ratings.csv")


# In[4]:


df = df[df["Stars"] != "Unrated"]
df["Stars"] = df["Stars"].astype(float)


# In[5]:


df["Top_Ten_Binary"] = df["Top Ten"].notna().astype(int)


# In[6]:


X = df[["Brand", "Style", "Country", "Top_Ten_Binary"]]
y = df["Stars"]




# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


categorical_cols = ["Brand", "Style", "Country"]
numeric_cols = ["Top_Ten_Binary"]



# In[11]:


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", CatBoostEncoder())
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])







# In[13]:


preprocessor = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_cols),
    ("num", numeric_pipeline, numeric_cols)
])


# In[14]:


# In[15]:


model_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", CatBoostRegressor(verbose=0, random_state=42))
])


# In[16]:


model_pipeline.fit(X_train, y_train)



# In[19]:


y_pred = model_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# In[21]:


print(r2)
print(rmse)


# In[22]:




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





