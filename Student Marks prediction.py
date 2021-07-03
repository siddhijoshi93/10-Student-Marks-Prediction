# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import matplolib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:/Users/ADMIN/Desktop/Siddhi/StudentsPerformance.csv")

#EDA
data.columns
data.shape
data.info()
data.describe()
#checking missing values
data.isna().sum()

data['TotalScore'] = data[['math score', 'reading score','writing score']].sum(axis=1)
gender = pd.get_dummies(data['gender'],drop_first=True)

#adding changed gender coloumn to main dtaframe
df = pd.concat([df,gender],axis=1)

#checking other oclumns
data['race/ethnicity'].value_counts
data['parental level of education'].value_counts
data['lunch'].value_counts
data['test preparation course']

#adding 0,1 as a yes andno in column race,parental level of eductaion,lunch and test prepration
group = pd.get_dummies(data['race/ethnicity'],drop_first=True)
education = pd.get_dummies(data['parental level of education'],drop_first=True)
lunch = pd.get_dummies(data['lunch'],drop_first=True)
course = pd.get_dummies(data['test preparation course'],drop_first=True)

#combining updated above columns in main dataframe
data = pd.concat([data,group,education,lunch,course],axis=1)
 
data.head(5)

#removing columns from the data
data.drop(['gender','race/ethnicity','parental level of education','lunch','test preparation course'],axis=1,inplace=True)

#Data Visualiztion
 #making a heatmap
 plt.figure(figsize=(14,14))
sns.heatmap(data.corr(),annot=True)

#model making
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

data=pd.DataFrame(scaler.fit_transform(data),columns=data.columns)
data.head(5)

X=data.drop('TotalScore', axis=1)
y=data['TotalScore']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#linear Regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)
y_predLR = LR.predict(X_test)
actualy=y_test
#accuracy
from sklearn.metrics import r2_score,mean_squared_error
r2_score(y_predLR,y_test)
#accuracy= 1.0


#visualizing the prediction
plt.scatter(y_test,y_predLR)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_predLR))
print('MSE:', metrics.mean_squared_error(y_test, y_predLR))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predLR)))
#MAE: 5.388166764615695e-16
#MSE: 4.791311312678572e-31
#RMSE: 6.921929870114672e-16

#Random Forst
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
y_predrfr=rfr.predict(X_test)
actualy_rfr=y_test

#accurcay
r2_score(y_predrfr,y_test)
#0.9970686991083776

#visualizing the prediction
plt.scatter(y_test,y_predrfr)

print('MAE:', metrics.mean_absolute_error(y_test, y_predrfr))
print('MSE:', metrics.mean_squared_error(y_test, y_predrfr))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predrfr)))
#MAE: 0.035605749234374755
#MSE: 0.0024712061671489536
#RMSE: 0.049711227777524805
#Decision Tree
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
y_preddtr=dtr.predict(X_test)
actualy_dtc=y_test

#accuracy
r2_score(y_preddtr,y_test)
#0.987377417290908

#visulaizing the prediction
plt.scatter(y_test,y_preddtr)

print('MAE:', metrics.mean_absolute_error(y_test, y_preddtr))
print('MSE:', metrics.mean_squared_error(y_test, y_preddtr))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_preddtr)))
#MAE: 0.0778936679845404
#MSE: 0.011021664392180339
#RMSE: 0.104984114951611111114561

#Conclusion- All the three models have shown above 95% accuracy hence all te tree are good odels for the data