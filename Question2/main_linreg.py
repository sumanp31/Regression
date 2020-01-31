#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 06:23:12 2020

@author: suman
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import mode


df = pd.read_csv("Expander_final.csv", index_col=0)

y = df["Discharge Pressure (psig)"].values
X = df.drop("Discharge Pressure (psig)", axis=1).values

del df


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)

import pickle
pickle.dump(linreg, open("lin_reg.sav", 'wb'))


linreg = pickle.load(open("lin_reg.sav", 'rb'))
pred_y = linreg.predict(X_test)

from sklearn.metrics import mean_squared_error 
print ("The Minimum mean square error is: " + str(mean_squared_error(y_test, pred_y)))


from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=4)
rfreg.fit(X_train, y_train)
train_pred = rfreg.predict(X_train)
pickle.dump(rfreg, open("rf_reg.sav", 'wb'))


rfreg = pickle.load(open("rf_reg.sav", 'rb'))
pred_y = rfreg.predict(X_test)
print ("The Minimum mean square error is: " + str(mean_squared_error(y_test, pred_y)))


from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
regressor.compile(optimizer = "adam", loss =  'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 200, batch_size = 10)
regressor.save("ANN.h5")

from keras.models import load_model
regressor = load_model("ANN.h5")
y_pred = regressor.predict(X_test)
print ("The Minimum mean square error is: " + str(mean_squared_error(y_test, y_pred)))


















