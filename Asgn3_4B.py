#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Ans of ques 4B


# In[16]:


# Uploading modules and dataset
 
from sklearn.datasets  import load_breast_cancer
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
 


# In[17]:


cancer = load_breast_cancer()


# In[6]:


# Building dataframe
df =  pd.DataFrame(can.data, columns = cancer.feature_names)
df['target'] = cancer.target
 
pd.set_option('max_columns', 31)
pd.set_option('display.width', None)
 
df.head()


# In[19]:


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


# In[20]:


#  Normalization
 
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)


# In[24]:


#  selecting the number of attributes with variance ratio > 98% 
 
def variance_ratio_calculator(n):
    
       
    pca = PCA(n_components=n)
    pca_components = pca.fit_transform(x)
    return sum(pca.explained_variance_ratio_)*100
 
for i in range(30):
    if variance_ratio_calculator(i+1) > 98:
        # print("Number of principle components required for more than 98% total variance ratio:", i+1)
        break


# In[25]:


#  making a new dataset from the principle components
 
pca = PCA(n_components=i+1)
pca_components = pca.fit_transform(x)
pca_DF = pd.DataFrame(data = pca_components)
new_data = pd.concat([pca_DF, df[['target']]], axis = 1)
new_data.head()
 
new_x = new_data.iloc[:,:14]
new_y = new_data.iloc[:,14]
 
## splititing the dataset
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_x, new_y, random_state=11, test_size=.3)
 


# In[11]:


## training the model for different k value
 
from sklearn.model_selection import KFold, cross_val_score
 
for k in range (2,15):
    kfold = KFold(n_splits=10, random_state = 11, shuffle = True)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = kfold)
    print('K = {0:<4}\tMean Accuracy = {1:.2%}\t\tStandard Deviation = {2:.2}'.format(k,scores.mean(), scores.std()))
    
 
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X = X_train, y = y_train)
predicted = knn.predict(X_test)
expected = y_test
 
print('\n Mismatched Prdiction: ')
wrong = [(p,e) for (p,e) in zip (predicted,expected) if p!=e]
print(wrong,"\n\n")
 


# In[26]:


# Confusion matrix
 
from sklearn.metrics import confusion_matrix
 
confusion  = confusion_matrix(y_true = expected, y_pred = predicted)
confusion_df= pd.DataFrame(confusion, index = range(2), columns = range(2))
sns.heatmap(confusion_df, annot= True, cmap = 'Reds')
plt.show()
 


# In[27]:


# Classification report 
 
 
from sklearn.metrics import classification_report
 
print('\nClassification Report: \n')
print(classification_report(expected, predicted, target_names = cancer.target_names))


# In[28]:


df =  pd.DataFrame(cancer.data, columns = can.feature_names)
df['target'] = cancer.target
 
pd.set_option('max_columns', 31)
pd.set_option('display.width', None)
 
df.head()


# In[ ]:





# In[ ]:





# In[ ]:




