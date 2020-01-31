#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:38:28 2020

@author: suman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Expander_data.csv")

from sklearn import preprocessing

min_val = df["Discharge Pressure (psig)"].min()
max_val = df["Discharge Pressure (psig)"].max()

file_name = input("Enter the file name: ")

df1 = pd.read_csv(file_name)

col = list(df1.iloc[:,1:-1].columns) 
x = df1.iloc[:,1:-1].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (min_val, max_val))
x = min_max_scaler.fit_transform(x)
df1 = pd.DataFrame(x, columns=col)

y = df1["Discharge Pressure (psig)"].values
X = df1.drop("Discharge Pressure (psig)", axis=1).values

print("There are 3 trained models.\nEnter 1 for Linear Regression.\nEnter 2 for Random Forest.\nEnter 3 for Artifitial Neural Network.")
c = int(input("Enter your choice: "))


import pickle
from sklearn.metrics import mean_squared_error 

if c==1:
    from sklearn.linear_model import LinearRegression
    linreg = pickle.load(open("lin_reg.sav", 'rb'))
    pred_y = linreg.predict(X) 
    print ("The Minimum mean square error is: " + str(mean_squared_error(y, pred_y)))
    
elif c==2:
    from sklearn.ensemble import RandomForestRegressor
    rfreg = pickle.load(open("rf_reg.sav", 'rb'))
    pred_y = rfreg.predict(X)
    print ("The Minimum mean square error is: " + str(mean_squared_error(y, pred_y)))
    
elif c==3:
    from keras.models import load_model
    regressor = load_model("ANN.h5")
    y_pred = regressor.predict(X)
    print ("The Minimum mean square error is: " + str(mean_squared_error(y, y_pred)))
else:
    print ("You have entered a wrong option. Please try again.")


