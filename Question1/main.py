#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:00:06 2020

@author: suman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("weather_cleaned.csv", index_col = 0)

step = 360
df1 = df[step:].copy()
for i in range(step):
    df1["temp_"+str(i)] = df.loc[i:len(df1)+i-1," _tempm"].values
    
del df
df1.drop([" _tempm", "datetime_utc", " _hail", " _snow", " _tornado", " _wdird", " _wspdm"], axis=1, inplace=True)
df1.dropna(inplace=True)

df1.to_csv("Weather_final.csv" )

df1 = pd.read_csv("Weather_final.csv", index_col = 0)

y = df1["temp"].values
df1.drop(["temp"], axis=1, inplace=True)
X = np.reshape(df1.values, (df1.values.shape[0], df1.values.shape[1], 1)) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

del df1

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the R
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 60, batch_size = 64)

predicted_temp = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error 
print ("The mean squared error is: " + str(mean_squared_error(y_test, predicted_temp)))

plt.plot(y_test, color = 'red', label = 'Real temp')
plt.plot(predicted_temp, color = 'blue', label = 'Predicted temp')
plt.title('Delhi temperature prediction')
plt.xlabel('day')
plt.ylabel("temperature")
plt.legend()
plt.show()


import keras.models
regressor.save("model.h5")


