#!/usr/bin/env python
# coding: utf-8

# # CPSC 5240-Project(Part3)

# In[36]:


## Library upload

import  pandas as pd
import numpy as np


# In[37]:


## Data upload 

Air_Quality = pd.read_csv('D:\Data_A\Project\Real_Combine.csv')
print(Air_Quality)


# # Data preprocessing 

# In[38]:


## Data cleaning 

Air_Quality.isnull().sum()

Air_Quality[Air_Quality['PM 2.5'].isnull()]


# In[39]:


## Data update

Air_Quality_Updated = Air_Quality.interpolate()
display(Air_Quality_Updated) 


# In[40]:


## Statistical analysis of data 

Air_Quality_Updated.describe()


# In[41]:


## Data visualization 

from matplotlib import pyplot as plt 
import seaborn as sns
sns.heatmap(Air_Quality_Updated.isnull(), yticklabels=False)


# In[42]:


## Pairplot

from matplotlib import pyplot as plt 
import seaborn as sns
sns.pairplot(Air_Quality_Updated)
plt.savefig("pairplot.png")


# In[43]:


## Data outlayers using boxplot

import matplotlib.pyplot as plt

a4_dims=(11.7, 8.27)
flg, ax= plt.subplots(figsize=a4_dims)
g=sns.boxplot(data=Air_Quality_Updated, linewidth=2.5, ax=ax)
g.set_yscale("log")
plt.savefig("boxplot.png")
plt.show()


# In[72]:


## Variance or co-variance
#print(Air_Quality_Updated.corr())


# In[73]:


## Data normalization 

target = Air_Quality_Updated['PM 2.5']
Air_Quality_Updated = Air_Quality_Updated.iloc[:,:-1].apply(lambda x: (x-x.min(axis=0)) / (x.max(axis=0)-x.min(axis=0)))
Air_Quality_Updated['PM 2.5'] = target
new_data = pd.DataFrame(data = Air_Quality_Updated)
print(new_data)


# In[74]:


## Variance or co-variance
print(Air_Quality_Updated.corr())


# In[75]:


## Feature reduction by PCA technique


# In[76]:


## separate dataset into attributes and target
cols = ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'] 
x = new_data.loc[:, cols].values
y = new_data.loc[:, ['PM 2.5']].values


# In[77]:


from sklearn.decomposition import PCA    
for i in range(2,8):
    pca = PCA(n_components = i)
    pca_components = pca.fit_transform(x)
    print('For', i, 'components', 'total variance: ', sum(pca.explained_variance_ratio_))
    #pca_components = pca.fit_transform(x)


# In[78]:


## We can select 6 components

pca = PCA(n_components = 6)
pca_components = pca.fit_transform(x)
pca_new_data = pd.DataFrame(data = pca_components, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
pca_new_data = pd.concat([pca_new_data, new_data[['PM 2.5']]], axis = 1)


# # Application of Model 

# In[79]:


## Decision tree (1st model)


# In[80]:


# Spliting data into train and test 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pca_new_data.iloc[:,:-1], pca_new_data.iloc[:,-1], random_state=10)


# In[81]:


X_train


# In[82]:


X_test


# In[83]:


## Training the model

from sklearn.tree import DecisionTreeRegressor
reg_decision_model = DecisionTreeRegressor(max_depth=6)


# In[84]:


## Fit independent varaibles to the dependent variables

reg_decision_model.fit(X_train,y_train)
predicted = reg_decision_model.predict(X_test)
expected  = y_test


# In[85]:


## Check for overfit
print("Training set score: ", reg_decision_model.score(X_train,y_train))
print("Testing set score: ", reg_decision_model.score(X_test,y_test))


# In[86]:


## Decision Tree Model Evaluation
from sklearn import metrics
print("Mean Squared Error: ", metrics.mean_squared_error(expected, predicted))
print("Mean Absloute Error: ", metrics.mean_absolute_error(expected,predicted))
print("R2 Score: ", metrics.r2_score(expected,predicted))


# In[87]:


## Decision Tree Model Evaluation


# In[88]:


prediction=reg_decision_model.predict(X_test)
# checking difference between labled y and predicted y
sns.distplot(y_test-prediction)


# In[89]:


# #checking predicted y and labeled y using a scatter plot.
plt.scatter(y_test,prediction)
plt.xlabel("Expected")
plt.ylabel("Predicted")


# In[ ]:





# # Analysis using KNN model  

# In[90]:


## Change the target column to categorical so that the problem becomes classification problem

pca_new_data['target'] = pd.qcut(pca_new_data['PM 2.5'], 3, labels=list('ABC'))
print(pca_new_data.head())


# In[91]:


## Dropping the old target column: PM 2.5 
new_data = pca_new_data.drop(labels = ['PM 2.5'], axis  = 1)
print(pca_new_data.head())


# In[92]:


## New data set 

new_x = new_data.iloc[:,:6]
new_y = new_data.iloc[:,6]


# In[93]:


## Splitting the dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_x, new_y, random_state=10)


# In[94]:


# Uploading the necessary library 

from sklearn import preprocessing
from sklearn import utils
from sklearn.datasets  import load_breast_cancer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score


# In[95]:


# training the model for different k value

for k in range (2,7):
    kfold = KFold(n_splits=10, random_state = 10, shuffle = True)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = kfold)
    print('K = {0:<4}\tMean Accuracy = {1:.2%}\t\tStandard Deviation = {2:.2}'.format(k,scores.mean(), scores.std()))
    


# In[96]:


## Training the model for n_neighbors = 2

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X = X_train, y = y_train)
predicted = knn.predict(X_test)
expected = y_test


# In[97]:


## Printing the mismatched prediction
print('\n Mismatched Prdiction: ')
wrong = [(p,e) for (p,e) in zip (predicted,expected) if p!=e]
print(wrong,"\n\n")
print('number of mismatched prediction: ' + str(len(wrong)))


# In[98]:


## Plotting the confusion matrix
from sklearn.metrics import confusion_matrix
confusion  = confusion_matrix(y_true = expected, y_pred = predicted)
confusion_df= pd.DataFrame(confusion, index = range(3), columns = range(3))
sns.heatmap(confusion_df, annot= True, cmap = 'Reds')
plt.show()


# In[99]:


# Classification report 
from sklearn.metrics import classification_report
print('\nClassification Report: \n')
print(classification_report(expected, predicted))


# In[ ]:





# # Use model random forest 

# In[100]:


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


# In[101]:


# define the model
model = RandomForestClassifier()


# In[102]:


model.fit(X = X_train, y = y_train)
predicted = model.predict(X_test)
expected = y_test


# In[103]:


print('\n Mismatched Prdiction: ')
wrong = [(p,e) for (p,e) in zip (predicted,expected) if p!=e]
print(wrong,"\n\n")



from sklearn.metrics import confusion_matrix
 
confusion  = confusion_matrix(y_true = expected, y_pred = predicted)
confusion_df= pd.DataFrame(confusion, index = range(3), columns = range(3))
sns.heatmap(confusion_df, annot= True, cmap = 'Reds')
plt.show()


# Classification report 
 
 
from sklearn.metrics import classification_report
 
print('\nClassification Report: \n')
print(classification_report(expected, predicted))


# In[ ]:





# In[ ]:




