#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary dependencies

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Data collection and preprocessing

# In[2]:


raw_df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Spam mail prediction/mail_data.csv')


# In[3]:


raw_df


# In[4]:


# replace the null values with a null string
mail_data = raw_df.where((pd.notnull(raw_df)), '')


# In[5]:


# PRINT THE FIRST 5 ROWS OF THE DATASET
mail_data.head()


# In[6]:


# shape of the dataset
mail_data.shape


# Label Encoding

# In[7]:


mail_data.replace({'Category' : {'ham' : 0, 'spam' : 1}}, inplace = True)


# In[8]:


mail_data


# Separating the features and labels

# In[9]:


X = mail_data['Message']
Y = mail_data['Category']


# In[10]:


print(X)


# In[11]:


print(Y)


# Splitting the data into training and testing data

# In[12]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2, stratify = Y, random_state = 2)


# In[13]:


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


# Feature extraction

# In[14]:


# Transform the text data to feature vector that can be used as input
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = 'True')


# In[15]:


x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)


# In[16]:


#converting the y_train and y-test as integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[17]:


print(x_train_features)


# Model training:
# 
# Logistic regression

# In[18]:


model =LogisticRegression()


# In[19]:


model.fit(x_train_features, y_train)


# # Model evaluation

# In[20]:


#on training data
training_prediction = model.predict(x_train_features)
#accuracy of the model
training_accuracy = accuracy_score(y_train, training_prediction)

print('ACCURACY OF THE MODEL ON TRAINING DATA : ', training_accuracy)


# In[21]:


#on testing data
testing_prediction = model.predict(x_test_features)
#accuracy of the model
testing_accuracy = accuracy_score(y_test, testing_prediction)

print('ACCURACY OF THE MODEL ON TESTING DATA : ', testing_accuracy)


# # Building the predictive model

# In[27]:


input_mail = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]

# convert text to feature vector
input_feature = feature_extraction.transform(input_mail)

#making prediction
prediction = model.predict(input_feature)
if prediction == 1:
    print('SPAM')
else:
    print('HAM')


# In[ ]:




