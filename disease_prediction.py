#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


# In[2]:


heart_data=pd.read_csv(r'C:\Users\Pranitha\OneDrive\Desktop\heart.csv')


# In[3]:


heart_data.head()


# In[4]:


x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']


# In[5]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=45)


# In[6]:


model=LogisticRegression()
model.fit(x_train,y_train)


# In[7]:


x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
input_data_str=['62','0','0','140','268','0','0','160','0','3.6','0','2','2']
input_data=np.array(input_data_str,dtype=float).reshape(1,-1)


# In[8]:


prediction=model.predict(input_data)
print(prediction)
if prediction[0]==0:
    print('The person doesnot have a heart disease')
else:
    print('The person has heart disease')


# In[9]:


print(training_data_accuracy)


# In[10]:


import pickle
filename='heart_disease_model1.sav'
pickle.dump(model,open(filename,'wb'))
loaded_model=pickle.load(open('heart_disease_model1.sav','rb'))
for column in x.columns:
    print(column)


# # Diabetes 

# In[11]:


diabetes_dataset=pd.read_csv(r'C:\Users\Pranitha\OneDrive\Desktop\diabetes.csv')


# In[12]:


diabetes_dataset.head()


# In[13]:


x=diabetes_dataset.drop(columns='Outcome',axis=1)
y=diabetes_dataset['Outcome']


# In[14]:


diabetes_dataset.info()


# In[16]:


import matplotlib.pyplot as plt
p=diabetes_dataset[diabetes_dataset['Outcome']==1].hist(figsize=(20,20))
plt.title('Diabetes patient')


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[19]:


print(x.shape,x_train.shape,x_test.shape)


# In[23]:


classifier=svm.SVC(kernel='linear')


# In[25]:


classifier.fit(x_train,y_train)


# In[26]:


x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[27]:


print('Accuracy score of the training data: ',training_data_accuracy)


# In[28]:


input_data=(5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=classifier.predict(input_data_reshaped)
print(prediction)
if prediction[0]==0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")


# In[29]:


import pickle


# In[39]:


filename=r'C:\Users\Pranitha\OneDrive\Desktop\diabetes_model.sav'
pickle.dump(classifier,open(filename,'wb'))


# In[40]:


loaded_model = pickle.load(open(r'C:\Users\Pranitha\OneDrive\Desktop\diabetes_model.sav', 'rb'))


# In[41]:


input_data=(5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if prediction[0]==0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")


# In[43]:


for column in x.columns:
    print(column)


# # Parkisons  

# In[45]:


parkinsons_data=pd.read_csv(r'C:\Users\Pranitha\OneDrive\Desktop\parkinsons.csv')


# In[46]:


parkinsons_data.head()


# In[47]:


parkinsons_data.shape


# In[49]:


x=parkinsons_data.drop(columns=['name','status'],axis=1)
y=parkinsons_data['status']


# In[50]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[51]:


model=svm.SVC(kernel='linear')


# In[52]:


model.fit(x_train,y_train)


# In[53]:


x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(y_train,x_train_prediction)
print('Accuracy score of the training data: ',training_data_accuracy)


# In[54]:


x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(y_test,x_test_prediction)
print('Accuracy score of the test data: ',test_data_accuracy)


# In[62]:


input_data=(197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.00339,26.77500,0.42)
input_data_as_numpy_array=np.asarray(input_data)
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data_reshaped)
prediction = model.predict(input_data_scaled)
print(prediction)
if (prediction[0]==0):
    print("The person is doesnot have parkinsons Disease")
else:
    print("The person has Parkinsons")


# In[63]:


import pickle
filename='parkinsons_model.sav'
pickle.dump(model,open(filename,'wb'))
loaded_model=pickle.load(open('parkinsons_model.sav','rb'))
for column in x.columns:
    print(column)


# In[ ]:




