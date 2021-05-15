#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


os.chdir('F:\Pallav\da\default')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set


# In[4]:


raw_data = pd.read_csv ('Default_Fin.csv')


# In[5]:


raw_data.head()


# In[8]:


df1 = raw_data.rename(columns = {'Index':'index','Employed':'employed','Bank Balance': 'bank_balance', 'Annual Salary': 'annual_salary','Defaulted?':'defaulted' })


# In[9]:


df1.head()


# In[11]:


df1.info()


# In[16]:


sns.distplot(df1['bank_balance'])


# In[17]:


sns.distplot(df1['annual_salary'])


# In[18]:


df1.describe()


# In[59]:


plt.scatter(df1['bank_balance'],df1['annual_salary'])


# In[24]:


df2 = df1.copy()


# In[26]:


df2['employed'] = df2['employed'].astype('bool')
df2['defaulted'] = df2['defaulted'].astype('bool')
df2.info()


# In[27]:


df2.head()


# In[28]:


from sklearn.model_selection import train_test_split


# In[33]:


df2['employed'] = pd.get_dummies(df2['employed'])
df2['defaulted'] = pd.get_dummies(df2['defaulted'])
df2


# In[35]:


y = df2['defaulted']
x = df2[{'employed','bank_balance','annual_salary'}]


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(x,y)


# In[39]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[40]:


lr.fit(x_train,y_train)


# In[41]:


lr.score(x_train,y_train)


# In[44]:


predict = lr.predict(x_test)
predict


# In[46]:


score = lr.score(x_test,y_test)
score


# In[49]:


from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test,predict)
print(confusion_matrix)


# In[51]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predict))


# In[64]:


ConfusionMatrix = pd.DataFrame(confusion_matrix, columns = ('False', 'True'), index = ('Negative','Positve'))
ConfusionMatrix


# In[ ]:




