#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[2]:


# import the data
df = pd.read_csv('mail_data.csv')


# In[3]:


df.head()


# In[4]:


# shape of the dataset
df.shape


# In[5]:


# checking the distribution of target vriable
df['Category'].value_counts()


# In[6]:


# checking for any missing values
df.isnull().sum()


# In[7]:


# Cleaning the dataset
stemmer = PorterStemmer()
corpus = []
for i in range(0 , len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['Message'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[8]:


# Vectorizing the dataset and creating the bag of words
vect = TfidfVectorizer()


# In[9]:


X = vect.fit_transform(corpus).toarray()


# In[10]:


# converting the textual target to numerical target
df.replace({'Category' : {'ham' : 0, 'spam' : 1 }}, inplace = True)


# In[11]:


Y = df['Category']


# In[12]:


print(X)
print(Y)


# In[13]:


# train test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state =2)


# In[14]:


# training model using naive bayes classifier
model = MultinomialNB()


# In[15]:


model.fit(x_train, y_train)


# In[16]:


y_pred = model.predict(x_test)


# In[20]:


from sklearn.metrics import confusion_matrix


# In[21]:


matrix = confusion_matrix(y_test, y_pred)


# In[22]:


matrix


# In[23]:


from sklearn.metrics import accuracy_score


# In[24]:


accuracy = accuracy_score(y_test, y_pred)
print('ACCURACY IS :',accuracy)


# In[ ]:




