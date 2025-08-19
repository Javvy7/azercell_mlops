#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV



# In[3]:


import boto3
bucket_name = 'dataminds-homeworks'
s3_file_key = 'data_usage_production.parquet'            # e.g. 'folder/myfile.txt'
local_file_path = 'data_usage_production.parquet'        # Local destination

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


# In[4]:


df = pd.read_parquet("data_usage_production.parquet")


# In[5]:


df


# In[6]:


df.columns


# In[7]:


df.dtypes.value_counts()


# In[8]:


df.isna().sum()


# In[9]:


list(df.isna().sum()) # to see all of the null values


# In[10]:


# Sample 15,000 rows randomly
df_sample = df.sample(n=15000, random_state=42)


# In[11]:


# Set index as required
df_sample.set_index("telephone_number", inplace=True)


# In[12]:




# In[13]:


# Split into X and y
X = df_sample.drop(columns=["data_compl_usg_local_m1"])
y = df_sample["data_compl_usg_local_m1"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


categorical_features = [
    'tariff_desc', 'customer_status', 
    'lasttariff_m2', 'lasttariff_m3', 'lasttariff_m4', 'lasttariff_m5', 'lasttariff_m6'
]

numerical_features = [col for col in X.columns if col not in categorical_features]


# In[15]:





# In[16]:


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[17]:



# In[18]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


# In[19]:


model.fit(X_train, y_train)


# In[20]:


model.score(X_test, y_test)



# In[22]:


y_pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))


# In[23]:


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()


# In[24]:


model.named_steps["regressor"].feature_importances_


# In[28]:


param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10]
}

search = RandomizedSearchCV(model, param_distributions=param_grid,
                            n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=42)

search.fit(X_train, y_train)

print("Best R² score:", search.best_score_)
print("Best parameters:", search.best_params_)


# In[27]:



# Replace with your actual credentials and info
bucket_name = 'dataminds-homeworks'
s3_file_key = 'javidan-nuriyev-fe1.ipynb' 
local_file_path = 'javidan-nuriyev-fe1.ipynb'

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




