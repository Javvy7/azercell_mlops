#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import boto3
bucket_name = 'dataminds-warehouse'
s3_file_key = 'HousingPrices-Amsterdam-August-2021.csv'            # e.g. 'folder/myfile.txt'
local_file_path = 'HousingPrices-Amsterdam-August-2021.csv'        # Local destination

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


df = pd.read_csv(s3_file_key)


# In[4]:


df


# In[5]:


df.head()        #to look first 5 rows of df


# In[6]:


df.shape         # to know shape of df (it`s also written at the bottom when we write 'df' command


# In[7]:


df.dtypes        # data types of each columns


# In[8]:


df.isna().sum()      # to know if there is null values


# In[9]:


df.duplicated().sum()    # to know if there are duplicate values


# In[10]:


plt.figure(figsize=(8,5))
sns.histplot(df['Price'], bins=30)
plt.title("Price Distribution")
plt.show()

#visualizes the distribution of the 'Price' column from the dataset


# In[11]:


sns.scatterplot(data = df, x = df['Price'], y = df['Room'])
#creates a scatter plot to show the relationship between the 'Price' and 'Room' columns


# In[12]:


sns.scatterplot(data = df, x = df['Price'], y = df['Area'])
#creates a scatter plot to show the relationship between the 'Price' and 'Area' columns


# In[16]:


pd.pivot_table(df, values='Price', 
                                index='Room',  
                                aggfunc=np.mean)

#creates a pivot table that shows the average price for each room category


# In[37]:


pd.pivot_table(df, values='Price', 
                                index='Area',  
                                aggfunc=np.mean)
#creates a pivot table that shows the average price for each unique value in the "Area" column


# In[39]:


sns.pairplot(data=df)
#creates a pair plot to visualize relationships and distributions between all numeric columns


# In[46]:


sns.barplot(
    data = df,x=df['Area'][:10],
    y=df['Price'][:10],
    estimator = np.median)
#creates a bar plot showing the median price for the first 10 area values 


# In[47]:


plt.figure(figsize= (12,6))

ax = sns.barplot(data = df,x=df['Area'],y=df['Price'],estimator = np.median)
ax.set_xlabel('Area',fontsize = 15)
ax.set_ylabel('Price', fontsize = 15)
ax.set_title('Bivariate analysis of Price and Area', fontsize= 20)

#creates a bar plot for the bivariate analysis of "Area" and "Price" using the median as the estimator


# In[50]:


sns.boxplot(x=df['Price'])
#to display outliers of the 'Price' column


# In[51]:


df.sort_values('Price', ascending=False).head(5)
#sorts the DataFrame by the "Price" column in descending order and displays the top 5 rows


# In[17]:



# Replace with your actual credentials and info
bucket_name = 'dataminds-homeworks'
s3_file_key = 'javidan-nuriyev-eda.ipynb' 
local_file_path = 'javidan-nuriyev-eda.ipynb'

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




