#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[4]:


wine_data=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv")
wine_data


# In[5]:


wine_data.head()


# In[6]:


wine_data.shape


# In[7]:


wine_data.tail()


# In[8]:


wine_data.isnull().sum()


# In[10]:


wine_data.describe()


# In[12]:


sns.catplot(x='quality',data=wine_data,kind='count')


# In[59]:


plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y = 'volatile acidity',data=wine_data)


# In[60]:


plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y ='citric acid',data=wine_data)


# In[61]:


plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='residual sugar',data=wine_data)


# In[62]:


plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='chlorides',data=wine_data)


# In[63]:


plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='free sulfur dioxide',data=wine_data)


# In[64]:


plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='total sulfur dioxide',data=wine_data)                


# In[65]:


correlation=wine_data.corr()


# In[66]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.if', annot=True, annot_kws={'size':8},cmap ='Blues')


# In[67]:


x=wine_data.drop('quality',axis=1)


# In[68]:


print(x)


# In[69]:


y= wine_data['quality'].apply(lambda y_value:1 if y_value>=7 else 0)


# In[70]:


print(y)


# In[71]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=3)


# In[72]:


print(y.shape,y_train.shape, y_test.shape)


# In[77]:


model=RandomForestClassifier()


# In[78]:


model.fit(x_train,y_train)


# In[79]:


x_train_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_train_prediction,y_test)


# In[80]:


print('Accuracy', test_data_accuracy)


# In[87]:


input_data=(7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')


# In[ ]:




