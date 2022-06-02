#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ques 3(F)


# In[185]:


# Implementation of wrapper technique


# In[186]:


from sklearn.datasets import load_diabetes


# In[187]:


diabetics=load_diabetes()


# In[203]:


import pandas as pd


# In[204]:


DF= pd.DataFrame(diabetics.data, columns = diabetics.feature_names)
print(DF.head())


# In[205]:


DF['db'] = diabetics.target


# In[206]:


pd.set_option('precision' ,4)
pd.set_option('max_columns' ,11)
pd.set_option('display.width' ,None)


# In[192]:


DF


# In[193]:


# Loading necessary labraries


# In[200]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt 
from sklearn import metrics
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[201]:


pip install mlxtend


# In[202]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression


# In[199]:


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1=plot_sfs(sfs1.get_metric_dict(), kind='std_dev')
plt.grid()
plt.show()


# In[182]:


selected_features=DF.loc[:,['bmi','s5','bp','s4','s3', 's6', 'db']]
selected_features


# In[183]:


x=selected_features.drop(labels='db',axis=1)


# In[184]:


x


# In[146]:


y=selected_features['db']
y


# In[207]:


# Splitting into trin and test data


# In[147]:


from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test= train_test_split(selected_features.data, selected_features.target, random_state=11)
x_train, x_test, y_train, y_test= train_test_split(x, y, random_state=11)
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)


# In[ ]:


# Linear regression 


# In[148]:


from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(X=x_train, y=y_train)
print(LR.get_params())


# In[149]:


print('Linear regression intercept = ', LR.intercept_)


# In[152]:


print('Linear regression coef:\n')
for i, name in enumerate(selected_features.columns[0:5]):
      print(f'{name:>10}: {LR.coef_[i]}')


# In[209]:


from sklearn import metrics
print('Mean squared error:', metrics.mean_squared_error(predicted, expected))


# In[ ]:


# Cross validation and Lasso vs Ridge


# In[156]:


from sklearn.linear_model import Lasso, Ridge
estimators = {'Linear Regression': LR,
               'Lasso': Lasso(),
               'Ridge': Ridge()
              }


# In[211]:


from sklearn.model_selection import cross_val_score
for estimator_name, estimator_object in estimators.items():
    fold = KFold(n_splits=10, shuffle=True, random_state=11)
    scores = cross_val_score(estimator = estimator_object, X=diabetics.data, y=diabetics.target, cv=fold, scoring='neg_mean_squared_error')
    print(f'{estimator_name:>17}: '  + f'Mean squared error={scores.mean():.3f}')


# In[ ]:




