#!/usr/bin/env python
# coding: utf-8

# In[232]:


# Ans of Ques 4A


# In[ ]:


##loading necessary modules and dataset
 
from sklearn.datasets  import load_breast_cancer
from sklearn.decomposition import PCA
import pandas as pd
 
cancer = load_breast_cancer()


# In[285]:


#reading the description
print(cancer.DESCR)


# In[287]:


## laoding into an dataframe
 
df =  pd.DataFrame(cancer.data, columns = cancer.feature_names)
df['target'] = cancer.target
 
pd.set_option('max_columns', 31)
pd.set_option('display.width', None)
 
df.head()
 
##separating the features and target
 
features =['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension']
x = df.loc[:, features].values
y = df.loc[:, ['target']].values
 


# In[288]:


# Normalization
 
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
 


# In[297]:


## selecting the number of attributes for >98% total variance ratio


pca = PCA(n_components=12)
pca_components = pca.fit_transform(x)
print(sum(pca.explained_variance_ratio_))

pca = PCA(n_components=13)
pca_components = pca.fit_transform(x)
print(sum(pca.explained_variance_ratio_))

pca = PCA(n_components=14)
pca_components = pca.fit_transform(x)
print(sum(pca.explained_variance_ratio_)


# In[ ]:




