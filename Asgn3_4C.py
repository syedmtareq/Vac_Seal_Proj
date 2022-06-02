#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Ans of Ques 4C


# In[14]:


# Uploading necessary modules and dataset
 
from sklearn.datasets  import load_breast_cancer
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()


# In[21]:


# Building dataframe
 
df =  pd.DataFrame(cancer.data, columns = cancer.feature_names)
df['target'] = cancer.target
 
pd.set_option('max_columns', 31)
pd.set_option('display.width', None)
 
df.head()


# In[25]:


# separating features and target
 
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


# In[26]:


# Normalization
 
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
 


# In[28]:


# selecting the number of attributes for >98% total variance ratio
 
def variance_ratio_calculator(n):
    
         
    pca = PCA(n_components=n)
    pca_components = pca.fit_transform(x)
    return sum(pca.explained_variance_ratio_)*100
 
for i in range(30):
    if variance_ratio_calculator(i+1) > 98:
        # print("Number of principle components required for more than 98% total variance ratio:", i+1)
        break


# In[29]:


#  New dataset from the principle components
 
pca = PCA(n_components=i+1)
pca_components = pca.fit_transform(x)
pca_DF = pd.DataFrame(data = pca_components)
new_data = pd.concat([pca_DF, df[['target']]], axis = 1)
new_data.head()
 
new_x = new_data.iloc[:,:14]
new_y = new_data.iloc[:,14]
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_x, new_y, random_state=11, test_size=.3)
 
 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
 
estimators = {
    'KNeighborsClassifier' : KNeighborsClassifier( n_neighbors= 10),
    'GaussianNB' : GaussianNB()
}
 
for estimator_name, estimator_object in estimators.items():
    kfold = KFold(n_splits=10, random_state = 11, shuffle = True)
    scores = cross_val_score(estimator = estimator_object, X = X_train, y = y_train, cv = kfold)
    print('Estimator = {0:<20}\tMean Accuracy = {1:.2%}\t\tStandard Deviation = {2:.2}'.format(estimator_name,scores.mean(), scores.std()))


# In[ ]:





# In[ ]:




