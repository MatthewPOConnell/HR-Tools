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

#Turning all values into scaled numbers after reading in a spreadsheet
df = pd.read_csv(r'C:\Users\SEANKurian\Desktop\attritionSheet.csv')
df_encoded = df.apply(preprocessing.LabelEncoder().fit_transform)

predictors = df_encoded.drop(columns=['Attrition'])
target = df_encoded['Attrition'].values
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=0.25, random_state=1)

#Determines optimal number of neighbors 
model = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 25)}
modelGSCV = GridSearchCV(model, param_grid, cv=5)
modelGSCV.fit(pred_train, tar_train)

#Optimal accuracy of ~84.6% is achieved with 20 neighbours
print(modelGSCV.best_params_)
print(modelGSCV.best_score_)