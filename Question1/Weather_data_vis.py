#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 07:25:06 2020

@author: suman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("weather_cleaned.csv", index_col=0)

plt.plot(df[" _dewptm"].values, color = 'red', label = 'dew')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('dew  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show() 


plt.plot(df[" _fog"].values, color = 'red', label = 'fog')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('fog  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show() 

plt.plot(df[" _hail"].values, color = 'red', label = 'hail')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('hail  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show() 

plt.plot(df[" _hum"].values, color = 'red', label = 'humidity')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('humidity  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show()

plt.plot(df[" _pressurem"].values, color = 'red', label = 'pressure')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('pressure  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show()

plt.plot(df[" _rain"].values, color = 'red', label = 'rain')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('rain  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show()

plt.plot(df[" _snow"].values, color = 'red', label = 'snow')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('snow  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show()

plt.plot(df[" _thunder"].values, color = 'red', label = 'thunder')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('thunder  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show()

plt.plot(df[" _tornado"].values, color = 'red', label = 'tornado')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('tornado  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show()

plt.plot(df[" _vism"].values, color = 'red', label = 'vism')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('vism  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show()

plt.plot(df[" _wdird"].values, color = 'red', label = 'wdird')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('wdird  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show()

plt.plot(df[" _wspdm"].values, color = 'red', label = 'wspdm')
plt.plot(df["temp"].values, color = 'blue', label = 'temperature')
plt.title('wspdm  vs temp   ')
plt.xlabel('day')
plt.legend()
plt.show()

df = pd.read_csv("Weather_data.csv", index_col=0)
df = df.loc[:, [" _tempm", " _wdire"]]
gf = df.groupby(" _wdire").mean()
plt.plot(gf, color = "blue", label = 'wind direction')
plt.title('wind direction vs temp')
plt.xlabel('wind direction')
plt.ylabel('average tempreture')
plt.legend()
plt.show()