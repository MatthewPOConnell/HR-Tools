# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:36:23 2019

@author: SEANKurian
"""
#Was using train test split on last commit, found cross val score is more accurate
#Plan on using hyper tuning to determine optimal number of neighbors, to be implemented
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Turning all values into scaled numbers after reading in a spreadsheet and dropping irrelevant data
df = pd.read_csv(r'C:\Users\SEANKurian\Desktop\attritionSheet.csv')
df_encoded = df.apply(preprocessing.LabelEncoder().fit_transform)
df_encoded = df_encoded.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])

predictors = df_encoded.drop(columns=['Attrition'])
target = df_encoded['Attrition'].values
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=0.25, random_state=1)

#Determines optimal number of neighbors 
model = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 25)}
modelGSCV = GridSearchCV(model, param_grid, cv=5)
modelGSCV.fit(pred_train, tar_train)

#Optimal accuracy of ~85.0% is achieved with 12 neighbours within the same set
print(modelGSCV.best_params_)
print(modelGSCV.best_score_)

#When using the data we set aside for predictions and comparing to actual values, we find the prediction
# is correct ~81.8% of the time
print(modelGSCV.predict(pred_test)[0:25])
print(tar_test[0:25])
print(modelGSCV.score(pred_test,tar_test))