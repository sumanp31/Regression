#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:48:11 2020

@author: suman
"""

import numpy as np
import pandas as pd
import collections
from scipy import stats


df = pd.read_csv("Weather_data.csv")

file_name = input("Enter the file name: ")

df_test = pd.read_csv(file_name, index_col=0)
df["temp"] = df[" _tempm"].values


df = df[:-len(df_test)]
df = pd.concat([df, df_test], sort=False)
df.reset_index(drop=True, inplace = True)


print ("The data is preprocessing.")

df = df[[" _tempm", "temp"]]

miss_temp = list(df[df[" _tempm"].isna()].index)
for i in miss_temp:
    df.loc[i, " _tempm"] = int(df.loc[[i-1,i+1], " _tempm"].dropna().values.mean())


df.info()

step = 360
df1 = df[step:].copy()
for i in range(step):
    df1["temp_"+str(i)] = df.loc[i:len(df1)+i-1," _tempm"].values
    

df_final = df1[-len(df_test):]

y = df_final["temp"].values
df_final.drop(["temp", " _tempm"], axis=1, inplace=True)
X = np.reshape(df_final.values, (df_final.values.shape[0], df_final.values.shape[1], 1)) 

print ("Data preprocessing is complete.")

from keras.models import load_model

regressor = load_model("model1.h5")
predicted_temp = regressor.predict(X)

from sklearn.metrics import mean_squared_error 
#print ("The mean squared error is: " + str(mean_squared_error(y, predicted_temp)))

df_test = df_test[["datetime_utc", " _tempm"]]

df_test[" _tempm"] = predicted_temp

d = input("Enter the date you are interested in(same format as in the dataset): ")

print ("The predicted temperature for " + d + " is: " + str(df_test[df_test["datetime_utc"] == d][" _tempm"].values[0]))