# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:46:17 2019

@author: SEANKurian
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import os

#Dropping uncessary columns and preprocessing the dataset
PATH_TO_DATA = 'ec'
NAME_OF_DF = 'employee-compensation.csv'
df = pd.read_csv(os.path.join(PATH_TO_DATA, NAME_OF_DF))
df = df.drop(columns=['Organization Group Code', 'Department Code', 'Union Code', 'Job Family Code', 'Job Code'])
df['Employee Identifier'] = df['Employee Identifier'].astype('object') 
df2 = df.apply(preprocessing.LabelEncoder().fit_transform)

#Removing N/A values
df2.apply(lambda x: x.isna().sum()/len(x)*100, axis=0)
df2 = df2.dropna()
columns_float = [i for i in df2.columns if (df2[i].dtypes == np.float) | (df2[i].dtypes == np.int)]

#Setting predictors/target variables (x and y) and splitting training and test data
predictors = df2.drop(columns=['Total Salary', 'Salaries', 'Overtime', 'Other Salaries'])
target = df2['Total Salary'].values
p_train, p_test, t_train, t_test = train_test_split(predictors, target, test_size=0.25, random_state=1)

#Creating the Linear Regression model and fitting it to the training data
regr = LinearRegression()
regr.fit(p_train, t_train)

#Model had an accuracy of 99%
print(regr.score(p_test, t_test))
