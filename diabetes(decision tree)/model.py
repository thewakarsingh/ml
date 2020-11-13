#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[ ]:





# In[ ]:


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

pima = pd.read_csv("diabetes.csv", header=None, names=col_names)
pima=pd.DataFrame(pima)
pima=pima.drop(pima.index[0])
pima.head()


# In[ ]:


#correlation_matrix = pima.corr().round(2)
#sns.heatmap(data=correlation_matrix, annot=True)


# In[ ]:


feature_cols = ['pregnant', 'insulin', 'bmi','age', 'glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# In[ ]:


clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf = clf.fit(X_train,y_train)


# In[ ]:

filename="modeldb"
fileobj=open(filename,'wb')
pickle.dump(clf,fileobj)
fileobj.close()

# In[ ]:


print(X_test.head())


# In[ ]:





# In[ ]:


y_pred = clf.predict(X_test)



