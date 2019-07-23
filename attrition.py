# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:36:23 2019

@author: SEANKurian
"""
#Was using train test split on last commit, found cross val score is more accurate
#Plan on using hyper tuning to determine optimal number of neighbors, to be implemented
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Turning all values into scaled numbers after reading in a spreadsheet and dropping irrelevant data
def loadFile(csv):
    df = pd.read_csv(csv)
    df_encoded = df.apply(preprocessing.LabelEncoder().fit_transform)
    df_encoded = df_encoded.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])
    df_pred = df_encoded.drop(columns=['Attrition'])
    df_tar = df_encoded['Attrition'].values
    return df_pred, df_tar, df

def trainModel(predTrain, tarTrain):
    #Determines optimal number of neighbors 
    model = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 25)}
    modelGSCV = GridSearchCV(model, param_grid, cv=5)
    modelGSCV.fit(predTrain, tarTrain)
    return modelGSCV

#1 signifies attrition
def predictAttrition(trainedModel, realPredictors):
    return trainedModel.predict(realPredictors)

def bestParams(model): 
   return model.best_params_

def bestScore(model):
    return model.best_score_

def score(model, predTestData, tarTestData):
    return model.score(predTestData, tarTestData)

def corrMatrix(data):
    data = data.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])
    names = data.columns.values
    data = data.apply(preprocessing.LabelEncoder().fit_transform)
    correlations = data.corr()
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,31,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation = 'vertical')
    ax.set_yticklabels(names)
    plt.show()
        
    
predictors, target, df = loadFile(r'C:\Users\SEANKurian\Desktop\attritionSheet.csv')
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=0.25, random_state=1)
trainedModel = trainModel(pred_train, tar_train)
predictAttrition(trainedModel, pred_test)
corrMatrix(df)
#print (bestScore(trainedModel))
#print (bestParams(trainedModel))
#print (score(trainedModel, pred_test, tar_test))





