#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 05:25:38 2020

@author: suman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Expander_data.csv")

from sklearn import preprocessing

min_val = df["Discharge Pressure (psig)"].min()
max_val = df["Discharge Pressure (psig)"].max()

col = list(df.iloc[:,1:-1].columns) 
x = df.iloc[:,1:-1].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (min_val, max_val))
x = min_max_scaler.fit_transform(x)
df1 = pd.DataFrame(x, columns=col)

info = df1.describe()

plt.plot(df1["Suction Pressure (psig)"].values, color = 'red', label = 'Suction Pressure (psig)')
plt.plot(df1["Discharge Pressure (psig)"].values, color = 'blue', label = 'Discharge Pressure (psig)')
plt.title('Suction Pressure (psig) vs Discharge Pressure (psig)   ')
plt.xlabel('day')
plt.legend()
plt.show() 


df1.plot.scatter(x= "Suction Pressure (psig)", y = "Discharge Pressure (psig)")
plt.show()


plt.plot(df1["Suction Temperature (F)"].values, color = 'red', label = 'Suction Temperature (F)')
plt.plot(df1["Discharge Pressure (psig)"].values, color = 'blue', label = 'Discharge Pressure (psig)')
plt.title('Suction Temperature (F)  vs Discharge Pressure (psig)   ')
plt.xlabel('day')
plt.legend()
plt.show() 

df1.plot.scatter(x= "Suction Temperature (F)", y = "Discharge Pressure (psig)")
plt.show()


plt.plot(df1["Total Flow (gpm)"].values, color = 'red', label = 'Total Flow (gpm)')
plt.plot(df1["Discharge Pressure (psig)"].values, color = 'blue', label = 'Discharge Pressure (psig)')
plt.title('Total Flow (gpm)  vs Discharge Pressure (psig)   ')
plt.xlabel('day')
plt.legend()
plt.show() 

df1.plot.scatter(x= "Total Flow (gpm)", y = "Discharge Pressure (psig)")
plt.show()


plt.plot(df1["Speed (rpm)"].values, color = 'red', label = 'Speed (rpm)')
plt.plot(df1["Discharge Pressure (psig)"].values, color = 'blue', label = 'Discharge Pressure (psig)')
plt.title('Speed (rpm)  vs Discharge Pressure (psig)   ')
plt.xlabel('day')
plt.legend()
plt.show() 

df1.plot.scatter(x= "Speed (rpm)", y = "Discharge Pressure (psig)")
plt.show()

plt.plot(df1["By-pass Valve Position (%)"].values, color = 'red', label = 'By-pass Valve Position (%)')
plt.plot(df1["Discharge Pressure (psig)"].values, color = 'blue', label = 'Discharge Pressure (psig)')
plt.title('By-pass Valve Position (%)  vs Discharge Pressure (psig)   ')
plt.xlabel('day')
plt.legend()
plt.show() 

df1.plot.scatter(x= "By-pass Valve Position (%)", y = "Discharge Pressure (psig)")
plt.show()


plt.plot(df1["Discharge Temperature (F)"].values, color = 'red', label = 'Discharge Temperature (F)')
plt.plot(df1["Discharge Pressure (psig)"].values, color = 'blue', label = 'Discharge Pressure (psig)')
plt.title('Discharge Temperature (F)  vs Discharge Pressure (psig)   ')
plt.xlabel('day')
plt.legend()
plt.show() 

df1.plot.scatter(x= "Discharge Temperature (F)", y = "Discharge Pressure (psig)")
plt.show()

"""
As all the features seems to be varying with discharge pressure, I am selecting all the features
"""

df1.to_csv("Expander_final.csv")