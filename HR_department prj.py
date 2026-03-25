#!/usr/bin/env python
# coding: utf-8

# 

# In[83]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[84]:


os.getcwd()


# In[85]:


os.chdir('C:\\Users\\adima\\OneDrive\\Documents\\Cohort 128_ML_ Day 47 Projects\\Project -4 HR Department')


# In[86]:


df=pd.read_csv('Human_Resources.csv')


# In[87]:


display(df)
display(df.shape)
display(df.info())
display(df.describe())


# In[88]:


display(df.columns)


# In[89]:


display(df.isnull().sum())


# In[90]:


display(df.Attrition)


# In[91]:


df.Attrition=df['Attrition'].map({'Yes':0,'No':1})


# In[92]:


display(df.OverTime)


# In[93]:


df.OverTime=df['OverTime'].map({'Yes':0,'No':1})


# In[94]:


display(df.Over18)


# In[95]:


df.Over18=df['Over18'].map({'Yes':0,'No':1})


# In[96]:


df.hist(bins=30,figsize=(20,50),color='r')
plt.show()


# In[97]:


df.drop(['EmployeeCount', 'StandardHours', 'Over18'],axis=1,inplace=True)


# In[98]:


display(df)


# In[99]:


left_df=df[df.Attrition==0]
stayed_df=df[df.Attrition==1]
display(left_df)
display(stayed_df)


# In[100]:


print("Total No. of emp",len(df))
print("Total No. of emp left",len(left_df))
print("Percentage of emp left",round(1.*len(left_df)/len(df)*100,2),'%')
print("Total No. of emp stayed",len(stayed_df))
print("Percentage of emp stayed",round(1.*len(stayed_df)/len(df)*100,2),'%')


# In[101]:


display(stayed_df.describe())
display(left_df.describe())


# In[171]:


cdf=df.copy()
cdf.drop(['BusinessTravel','Department','Gender','EducationField','MaritalStatus','JobRole'],axis=1,inplace=True)
correlations = cdf.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)
plt.show()


# In[105]:


plt.figure(figsize=[20,10])
sns.countplot(x='Age',hue='Attrition',data=df)
plt.show()


# In[106]:


plt.figure(figsize=[20,20])
plt.subplot(411)
sns.countplot(x='JobRole' ,hue='Attrition', data=df)
plt.subplot(412)
sns.countplot(x='MaritalStatus',hue='Attrition', data=df)
plt.subplot(413)
sns.countplot(x='JobInvolvement' ,hue='Attrition', data=df)
plt.subplot(414)
sns.countplot(x='JobLevel' ,hue='Attrition', data=df)
plt.show()


# In[104]:


display(df.columns)


# In[107]:


plt.figure(figsize=[20,20])
plt.subplot(211)
sns.countplot(x='DistanceFromHome',hue='Attrition', data=df)
plt.show()


# In[109]:


plt.figure(figsize=[12,7])
sns.kdeplot(left_df['DistanceFromHome'],label='Employees who left',shade=True,color='r')
sns.kdeplot(stayed_df['DistanceFromHome'],label='Employees who stayed',shade=True,color='b')
plt.show()


# In[111]:


plt.figure(figsize=[12,7])
sns.kdeplot(left_df['YearsWithCurrManager'],label='Employees who left',shade=True,color='r')
sns.kdeplot(stayed_df['YearsWithCurrManager'],label='Employees who stayed',shade=True,color='b')
plt.show()


# In[116]:


plt.figure(figsize=[12,7])
sns.kdeplot(left_df['TotalWorkingYears'],label='Employees who left',shade=True,color='r')
sns.kdeplot(stayed_df['TotalWorkingYears'],label='Employees who stayed',shade=True,color='b')
plt.xlabel='Total Working Years'
plt.show()


# In[119]:


plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'Gender', data = df)
plt.show()


# In[120]:


plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = df)
plt.show()


# In[122]:


x_obj=df[['BusinessTravel','Department','Gender','EducationField','MaritalStatus','JobRole']]
print(x_obj)


# In[126]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder()
x=onehotencoder.fit_transform(x_obj).toarray()
display(x)


# In[127]:


x=pd.DataFrame(x)
display(x)


# In[136]:


x_numerical=df[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',	'TotalWorkingYears'	,'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	,'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]
display(x_numerical)


# In[137]:


x_all=pd.concat([x,x_numerical],axis=1)
display(x_all.info())


# In[138]:


x_all.columns=x_all.columns.astype(str)
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()
X=minmax.fit_transform(x_all)
display(pd.DataFrame(X))


# In[139]:


y=df['Attrition']
display(y)


# In[141]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
display (X_train.shape)
display (X_test.shape)
display (y_train.shape)
display (y_test.shape)


# In[155]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
display(y_pred)
acc=r2_score(y_test,y_pred)
display(acc)


# In[158]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

print(accuracy_score(y_test, y_pred))


# In[164]:


cm=confusion_matrix(y_pred,y_test)
display(cm)
plt.figure(figsize=[5,5])
sns.heatmap(cm,annot=True)
plt.show()


# In[165]:


print(classification_report(y_pred,y_test))


# In[166]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[169]:


plt.figure(figsize=[5,3])
sns.heatmap(confusion_matrix(y_pred,y_test),annot=True)
plt.show()


# 

# In[172]:


from xgboost import XGBClassifier
xg=XGBClassifier()
xg.fit(X_train,y_train)
y_pred=xg.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))
plt.figure(figsize=[5,3])
sns.heatmap(confusion_matrix(y_pred,y_test),annot=True)
plt.show()


# In[177]:


import tensorflow as tf
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500,activation='relu',input_shape=(50,)))
model.add(tf.keras.layers.Dense(units=500,activation='relu'))
model.add(tf.keras.layers.Dense(units=500,activation='relu'))
model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
display(model.summary())


# In[182]:


model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])


# In[183]:


df['Attrition'].value_counts()


# In[197]:


epochs_hist=model.fit(X_train,y_train,epochs=100,batch_size=50)


# In[190]:


y_pred=model.predict(X_train)
y_pred=(y_pred>0.5)
display(y_pred)


# In[193]:


display(epochs_hist.history.keys())


# In[198]:


plt.plot(epochs_hist.history['loss'])
plt.title('Model loss during training')
plt.xlabel=('epoch')
plt.ylabel('Training loss')
plt.legend('Training loss')
plt.show()


# In[199]:


plt.plot(epochs_hist.history['accuracy'])
plt.title('Model accuracy')
plt.xlabel=('epoch')
plt.ylabel('accuracy')
plt.legend('accuracy')
plt.show()


# In[204]:


cm=confusion_matrix(y_pred,y_train)
display(cm)
plt.figure(figsize=[5,3])
sns.heatmap(cm,annot=True)
plt.show()


# In[205]:


print(classification_report(y_pred,y_train))


# In[206]:


y_pred=model.predict(X_test)
y_pred=(y_pred>0.5)
display(y_pred)


# In[207]:


display(epochs_hist.history.keys())


# In[208]:


plt.plot(epochs_hist.history['loss'])
plt.show()


# In[209]:


plt.plot(epochs_hist.history['accuracy'])
plt.show()


# In[212]:


print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))
plt.figure(figsize=[5,3])
sns.heatmap(confusion_matrix(y_pred,y_test) ,annot=True)
plt.show()


# In[ ]:




