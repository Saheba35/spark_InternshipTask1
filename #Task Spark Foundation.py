#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation - Data Science & Business Analytics Internship

# ## TASK 1- Prediction using Supervised Machine Learning 
# In this task we are going to predict the percentage marks of an student based on the no. of study hours using simple linear regression algorithm.

# ### STEP 1- Importing all the required libraries 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')


# ### STEP 2-  Data Preprocessing

# In[3]:


data = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[4]:


#1 understanding the data
data.head()


# In[5]:


data.tail()


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data.nunique()


# In[9]:


data.isnull().sum()


# ### STEP 3- Data Exploration 

# In[10]:


data.info()


# In[11]:


data.describe()


# ### STEP 4- Data Visualisation 

# In[12]:


plt.xlabel('Hours')
plt.ylabel('Scores')
plt.scatter(data.Hours,data.Scores, color='red',marker='+')


# In[13]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Scores', size=12)
plt.xlabel('Hours', size=12)
plt.show()
print(data.corr())


# ### STEP 5- Training The Model

# In[14]:


#Splitting the data
# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[15]:


#Fitting the data into the model
reg = LinearRegression()
reg.fit(train_X, train_y)


# In[16]:


#Predicting the Marks percentage
pred_y = reg.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# # Comparing the Predicted Marks with the Actual Marks

# In[17]:


comparing_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
comparing_scores


# ### Visually Comparing the Predicted Marks with the Actual Marks 

# In[18]:


df1 = comparing_scores.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# ### STEP-6  Evaluating The Model 

# In[19]:


# Error Matrics
# Calculating the accuracy of the model
print('Mean Absolute Error:', metrics.mean_absolute_error(val_y, pred_y))  
print('Mean Squared Error:', metrics.mean_squared_error(val_y, pred_y))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(val_y, pred_y)))


# # the predicted score of a student if he/she studies for 9.25 hrs/ day?

# In[20]:


hours = [9.25]
answer = reg.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# In[ ]:





# In[ ]:




