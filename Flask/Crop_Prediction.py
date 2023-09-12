#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd


# In[18]:


crop = pd.read_csv("Crop_recommendation.csv")
crop.head()


# In[19]:


crop.shape


# In[20]:


crop.tail(6)


# In[21]:


crop.columns


# In[22]:


crop.info()


# In[23]:


crop['no_label'] = pd.Categorical(crop['label']).codes


# In[24]:


crop.columns


# In[25]:


crop.duplicated().sum()


# In[26]:


crop.isnull().sum()


# In[27]:


crop.describe()


# In[28]:


crop.nunique()


# In[29]:


crop['label'].unique()


# In[30]:


crop['label'].value_counts()


# In[31]:


# Visual Analysis


# In[32]:


import matplotlib.pyplot as plt  
import seaborn as sns


# In[33]:


#Nitrogen
plt.figure(figsize=(8,7))
sns.histplot(x='N',data=crop,color='b');
plt.title("Nitrogen for crops",{'fontsize':20});


# In[34]:


#Potassium
plt.figure(figsize=(8,7))
sns.histplot(x='K',data=crop,color='b');
plt.title("Potassium for crops",{'fontsize':20});


# In[35]:


#Phosphorus
plt.figure(figsize=(8,7))
sns.histplot(x='P',data=crop,color='b');
plt.title("Phosphorus for crops",{'fontsize':20});


# In[36]:


#Temperature
plt.figure(figsize=(10,6))
sns.boxplot(x=crop.temperature);


# In[37]:


#Humidity
plt.figure(figsize=(10,6))
sns.boxplot(x=crop.humidity);


# In[38]:


#PH value
plt.figure(figsize=(8,7))
sns.histplot(x='ph',data=crop,color='b');
plt.title("PH for crops",{'fontsize':20});


# In[39]:


#Rainfall
plt.figure(figsize=(8,7))
sns.histplot(x='rainfall',data=crop,color='b');
plt.title("Rainfall feature",{'fontsize':20});


# In[40]:


#Distplot
sns.distplot(crop['ph'])


# In[ ]:





# In[41]:


sns.scatterplot(x='temperature',y='no_label',data=crop)


# In[42]:


sns.FacetGrid(crop,hue="no_label",height=5).map(plt.scatter,"N","humidity").add_legend()


# In[43]:


#get correlations of each feature in dataset
corrmat=crop.corr()
corrmat.style.background_gradient('coolwarm')


# In[44]:


top_corr_features=['N','P','K','temperature','humidity','ph','rainfall']


# In[45]:


#plot heat map
g=sns.heatmap(crop[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[46]:


sns.pairplot(crop,hue='no_label',height=3)


# In[50]:


#Splitting data into training and testing
#sklearn
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[51]:


#X consists of independent variables
X = crop.drop(['label','no_label'],axis=1)
#y consistes of dependent variable "label"
y = pd.Categorical(crop.label)


# In[52]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:





# In[53]:


#K-nearest - Neighbors Classifier
knnclassifier = KNeighborsClassifier(n_neighbors=9)
knnclassifier.fit(X_train,y_train)
print("The accuracy of K Nearest Neighbors Classifier is",knnclassifier.score(X_train,y_train),knnclassifier.score(X_test,y_test))
knn=[knnclassifier.score(X_train,y_train),knnclassifier.score(X_test,y_test)]


# In[54]:


#support Vector Machines Classifier
svm = SVC()
svm.fit(X_train,y_train)
print("the accuracy of svm is",svm.score(X_train,y_train),svm.score(X_test,y_test))
svm=[svm.score(X_train,y_train),svm.score(X_test,y_test)]


# In[55]:


#Decision Tree
dtclassifier=DecisionTreeClassifier(max_depth=7)
dtclassifier.fit(X_train,y_train)
print("The accuracy of decision tree classsifer is ",dtclassifier.score(X_train,y_train),dtclassifier.score(X_test,y_test))
dt=[dtclassifier.score(X_train,y_train),dtclassifier.score(X_test,y_test)]


# In[56]:


#Random forest
rfclassifier=RandomForestClassifier()
rfclassifier.fit(X_train,y_train)
print("The accuracy of random forest classifier is ",rfclassifier.score(X_train,y_train),rfclassifier.score(X_test,y_test))
rf=[rfclassifier.score(X_train,y_train),rfclassifier.score(X_test,y_test)]


# In[ ]:


#Testing Model With multiple evaluationmatrix


# In[61]:


#KNN Classifier
knnclassifier = KNeighborsClassifier()
knnclassifier.fit(X_train,y_train)
y_pred=knnclassifier.predict(X_test)
print(classification_report(y_test,y_pred))


# In[62]:


dtclassifier = DecisionTreeClassifier()
dtclassifier.fit(X_train,y_train)
y_pred=dtclassifier.predict(X_test)
print(classification_report(y_test,y_pred))


# In[63]:


svm = SVC()
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(classification_report(y_test,y_pred))


# In[65]:


rfclassifier=RandomForestClassifier()
rfclassifier.fit(X_train,y_train)
y_pred=rfclassifier.predict(X_test)
print(classification_report(y_test,y_pred))


# In[67]:


import pickle


# In[68]:


#Open a file,where you want to store the data
pickle.dump(knnclassifier,open('model.pkl','wb'))


# In[ ]:




