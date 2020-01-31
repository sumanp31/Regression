#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:21:57 2020

@author: suman
"""

import numpy as np
import pandas as pd
import collections
from scipy import stats


df = pd.read_csv("Weather_data.csv")

file_name = input("Enter the file name: ")

df_test = pd.read_csv(file_name, index_col=0)

df = df[:-len(df_test)]
df = pd.concat([df, df_test], sort=False)
df.reset_index(drop=True, inplace = True)


print ("The data is preprocessing.")
df["temp"] = df[" _tempm"].values

miss_cond = list(df[df[" _conds"].isna()].index)
for i in miss_cond:
    df.loc[i, " _conds"] = (collections.Counter(df.loc[i-7:i+7, " _conds"].values).most_common())[0][0]
    
miss_dewptm = list(df[df[" _dewptm"].isna()].index)

for i in miss_dewptm:
    df.loc[i, " _dewptm"] = int(df.loc[[i-1,i+1], " _dewptm"].dropna().values.mean())
    

miss_hum = list(df[df[" _hum"].isna()].index)
for i in miss_hum:
    df.loc[i, " _hum"] = int(df.loc[[i-1,i+1], " _hum"].dropna().values.mean())
    

q = df[" _pressurem"].quantile(0.999)
miss_press = list((df.loc[df[" _pressurem"] > q, " _pressurem"]).index)
miss_press.extend(list(df[df[" _pressurem"].isna()].index))
for i in miss_press:
    df.loc[i, " _pressurem"] = int(df.loc[[i-1,i+1], " _pressurem"].dropna().values.mean())

miss_temp = list(df[df[" _tempm"].isna()].index)
for i in miss_temp:
    df.loc[i, " _tempm"] = int(df.loc[[i-1,i+1], " _tempm"].dropna().values.mean())


q = df[" _vism"].quantile(0.999)
miss_vis = list((df.loc[df[" _vism"] > q, " _vism"]).index)
miss_vis.extend(list(df[df[" _vism"].isna()].index))

for i in miss_vis:
    df.loc[i, " _vism"] = int(df.loc[[i-1,i+1], " _vism"].dropna().values.mean())

miss_wdird = list(df[df[" _wdird"].isna()].index)
for i in miss_wdird[:-1]:
    df.loc[i, " _wdird"] = int(df.loc[[i-1,i+1], " _wdird"].dropna().values.mean())
df.loc[miss_wdird[-1], " _wdird"] = df.loc[miss_wdird[-2], " _wdird"]

miss_wdire = list(df[df[" _wdire"].isna()].index)
for i in miss_wdire:
    df.loc[i, " _wdire"] = df.loc[i-1, " _wdire"]
    
q = df[" _wspdm"].quantile(0.99)
miss_wsp = list(df[df[" _wspdm"].isna()].index)
miss_wsp.extend(list((df.loc[df[" _wspdm"] > q, " _wspdm"]).index))

for i in miss_wsp:
    df.loc[i, " _wspdm"] = int(df.loc[[i-1,i+1], " _wspdm"].dropna().values.mean())
    
#drop precipm as it has only nan values. _windchillm and _wgustm because they have negligible data

df.drop([" _heatindexm", " _precipm", " _windchillm", " _wgustm"], axis=1, inplace=True)

df = pd.get_dummies(df, columns=[" _conds", " _wdire"], dummy_na=False, drop_first=True)


from sklearn import preprocessing

min_val = df["temp"].min()
max_val = df["temp"].max()

col = list(df.iloc[:,1:].columns) 
x = df.iloc[:,1:].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (min_val, max_val))
x = min_max_scaler.fit_transform(x)
df1 = pd.DataFrame(x, columns=col)
df1["datetime_utc"] = df["datetime_utc"]
df1["temp"] = df["temp"]

df1.to_csv("weather_1.csv")

df = pd.read_csv("weather_1.csv", index_col = 0)

step = 360
df1 = df[step:].copy()
for i in range(step):
    df1["temp_"+str(i)] = df.loc[i:len(df1)+i-1," _tempm"].values
    
del df
df1.drop([" _tempm", "datetime_utc", " _hail", " _snow", " _tornado", " _wdird", " _wspdm"], axis=1, inplace=True)
df1.dropna(inplace=True)

df1.to_csv("Weather_final1.csv" )

df1 = pd.read_csv("Weather_final1.csv", index_col = 0)

df_final = df1[-len(df_test):]

y = df_final["temp"].values
df_final.drop(["temp"], axis=1, inplace=True)
X = np.reshape(df_final.values, (df_final.values.shape[0], df_final.values.shape[1], 1)) 

print ("Data preprocessing is complete.")

from keras.models import load_model

regressor = load_model("model.h5")
predicted_temp = regressor.predict(X)

from sklearn.metrics import mean_squared_error 
print ("The mean squared error is: " + str(mean_squared_error(y, predicted_temp)))

df_test = df_test[["datetime_utc", " _tempm"]]

df_test[" _tempm"] = predicted_temp

d = input("Enter the date you are interested in(same format as in the dataset): ")

print ("The predicted temperature for " + d + " is: " + str(df_test[df_test["datetime_utc"] == d][" _tempm"].values[0]))






























#print("There are 2 models.\nEnter 1 for past temperature based model.\nEnter 2 for maultivariate model")
#c = int(input("Enter your choice: "))
#y = df_final["temp"].values
#
#from sklearn.metrics import mean_squared_error 
#if c==1:
#    pass
#    
#elif c==2:
#
#    df_final.drop(["temp"], axis=1, inplace=True)
#    X = np.reshape(df_final.values, (df_final.values.shape[0], df_final.values.shape[1], 1)) 
#    from keras.models import load_model
#    regressor = load_model("model.h5")
#    y_pred = regressor.predict(X)
#    print ("The Minimum mean square error is: " + str(mean_squared_error(y, y_pred)))
#    
#else:
#    print ("You have entered a wrong option. Please try again.")
