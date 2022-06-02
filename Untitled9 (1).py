#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Ques 3


# In[2]:


# Loading the dataset


# In[3]:


from sklearn.datasets import load_diabetes


# In[4]:


diabetics=load_diabetes()


# In[5]:


print(diabetics.DESCR)


# In[6]:


# Data pattern


# In[7]:


print('Data shape', diabetics.data.shape)


# In[8]:


print('Target shape', diabetics.target.shape)


# In[9]:


# Attributes


# In[10]:


print(diabetics.feature_names)


# In[11]:


# Preprocessing 


# In[12]:


import pandas as pd


# In[ ]:


# Building data frame


# In[13]:


DF= pd.DataFrame(diabetics.data, columns = diabetics.feature_names)
print(DF.head())


# In[ ]:





# In[ ]:


DF['db'] = diabetics.target


# In[ ]:


pd.set_option('precision' ,4)
pd.set_option('max_columns' ,11)
pd.set_option('display.width' ,None)


# In[44]:


DF


# In[45]:


# importing the necessary library


# In[46]:


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt 
from sklearn import metrics
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[47]:


# Visualize using heatmap


# In[48]:


plt.figure(figsize=(12,10))
cor= DF.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[49]:


# Correlation amonge features


# In[50]:


cor_target = abs(cor['db'])
print(cor_target)


# In[51]:


# Selection of best features ( Ans of 3 A)


# In[76]:


selected_features= cor_target[cor_target > 0.5]
print('selected_features are: ', selected_features)


# In[53]:


# Correlation among the best features


# In[77]:


print(DF[['bmi', 's5']].corr())


# In[78]:


# Data frame using the best features


# In[111]:


selected_features=DF.loc[:,['bmi','s5', 'db']]
selected_features


# In[127]:


x=selected_features.drop(labels='db',axis=1)


# In[114]:


x


# In[115]:


y=selected_features['db']
y


# In[ ]:


# Split of training and testing data


# In[116]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, random_state=11)
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)


# In[ ]:


# Linear regrssion model


# In[117]:


from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(X=x_train, y=y_train)
print(LR.get_params())


# In[118]:


print('Linear regression intercept = ', LR.intercept_)


# In[121]:


print('Linear regression coef:\n')
for i, name in enumerate(selected_features.columns[0:2]):
      print(f'{name:>10}: {LR.coef_[i]}')


# In[ ]:


# R2 Score


# In[134]:


from sklearn import metrics
print('R2 score:', metrics.r2_score(predicted, expected))


# In[131]:


print('Training set score: ', LR.score(x_train, y_train))
print('Testing set score: ', LR.score(x_test, y_test))


# In[132]:


# Cross validation [c]
# Ridge and Lasso regression [D]


# In[135]:


from sklearn.linear_model import Lasso, Ridge
estimators = {'Linear Regression': LR,
               'Lasso': Lasso(),
               'Ridge': Ridge()
              }


# In[139]:


from sklearn.model_selection import cross_val_score
for estimator_name, estimator_object in estimators.items():
    fold = KFold(n_splits=10, shuffle=True, random_state=11)
    scores = cross_val_score(estimator = estimator_object, X=diabetics.data, y=diabetics.target, cv=fold, scoring='neg_mean_squared_error')
    print(f'{estimator_name:>17}: '  + f'Mean squared error={scores.mean():.3f}')


# In[ ]:





# In[ ]:





# In[ ]:




