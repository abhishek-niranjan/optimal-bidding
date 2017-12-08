#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:40:16 2017

@author: bromance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# loading dataframes from csv(S)

solar_train = 'Solar_Train.csv'
price_train = 'Price_Train.csv'
demand_train = 'Demand_Train.csv'
df_demand = pd.read_csv(demand_train)
df_solar = pd.read_csv(solar_train)
df_price = pd.read_csv(price_train)

# ploting weekly 
'''
for k in range(20):
    a = np.array([])
    for i in range((k * 7), ((k * 7) + 7)):
        a = np.append(a, df_price.iloc[i].values)
    x_axis = [(j+1) for j in range(168)]
    plt.plot(x_axis, a, label='price')
    plt.xlabel('hour number')
    plt.ylabel('Price')
    plt.title('week_{}'.format(str(k + 1)))
    #saving the plot as jpg
    fig_name = 'week_{}'.format(str(k + 1))
    plt.savefig(fig_name, dpi=600)
    plt.gcf().clear()  
'''
# maximum values

max1 = df_demand.values.max()
max2 = df_price.values.max()
max3 = df_solar.values.max()

# plotting demand & price for each day

for i in range(20):
    #plotting
    #df_price.shape[0]
    #lw = 2
    row1 = df_demand.iloc[i].divide(max1)
    row2 = df_price.iloc[(i)].divide(max2)
    row3 = df_solar.iloc[(i)].divide(max3)
    row1.plot(kind='line', label='demand(t)', legend=True)
    row2.plot(kind='line', label='price(t)', legend=True)
    row3.plot(kind='line', label='solar(t)', legend=True)
    plt.xlabel('hour')
    plt.ylabel('Solar/Price/Demand')
    plt.title('day_{}'.format(str(i + 1)))
    #saving the plot as jpg
    fig_name = 'day_together_{}'.format(str(i + 1))
    plt.savefig(fig_name, dpi=600)
    plt.gcf().clear()
'''
plt.show()

# drawing roc curve
#plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='AUROC = {:0.5f})'.format(auroc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('hour')
plt.ylabel('Demand/Price')
plt.title('day_{}'.format(str(i + 1)))
plt.legend(loc="lower right")

# Saving the plot as jpg
fig_name = 'day_{}'.format(str(i + 1))
plt.savefig(fig_name, dpi=600)
#plt.show()
'''