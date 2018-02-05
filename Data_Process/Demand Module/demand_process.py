#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:58:47 2017

@author: bromance
"""
import pandas as pd
import numpy as np

# importing dataset


# for training data (900 days)
# file = '../../DataSets/All_Vector.csv'
# file2 = '../../DataSets/Demand_Train_Pred.csv'


# for public leadeboard data (50 days)
# file = '../../DataSets/Data_LeaderBoard/All_Vector.csv'
# file2 = '../../DataSets/Data_LeaderBoard/Demand_LB_pred.csv'



file = '../../DataSets/Data_TestSet_PrivateEvaluation/All_Vector.csv'
file2 = '../../DataSets/Data_TestSet_PrivateEvaluation/Demand_Test_pred.csv'

dataset = pd.read_csv(file)
dataset2 = pd.read_csv(file2)

lag = 2
columns = []
t_dict = {}
t_dict['demand_td0'] = []
t_dict['day'] = []
t_dict['hour_of_day'] = []
# t_dict['demand_act'] = []

for i in range(1, lag + 1):
    t_dict['demand_t-'+str(i)] = []
    t_dict['demand_d-'+str(i)] = []
    t_dict['demand_t+'+str(i)] = []
    t_dict['demand_d+'+str(i)] = []

numberofrows=dataset.shape[0]
numberofdays=dataset2.shape[0]

for i in range(numberofrows):
    # hour columns
    t_dict['demand_td0'].append(dataset['demand_pred'][i])
    if(i>= lag) and (i < (numberofrows - lag)):
        for j in range(0, lag):
            t_dict['demand_t-'+str(j + 1)].append(dataset['demand_pred'][i - j - 1])
            t_dict['demand_t+'+str(j + 1)].append(dataset['demand_pred'][i + j + 1])
    elif (i < lag):
        for j in range(0, lag):
            if(j < i):
                t_dict['demand_t-'+str(j + 1)].append(dataset['demand_pred'][i - j - 1])
            else:
                t_dict['demand_t-'+str(j + 1)].append(dataset['demand_pred'][0])
        for j in range(0, lag):
             t_dict['demand_t+'+str(j + 1)].append(dataset['demand_pred'][i + j + 1])
    elif (i >= (numberofrows-lag)):
        for j in range(0, lag):
            t_dict['demand_t-'+str(j + 1)].append(dataset['demand_pred'][i - j - 1])
        for j in range(0, lag):
            if j < (numberofrows - i - 1):
              t_dict['demand_t+'+str(j + 1)].append(dataset['demand_pred'][i + j + 1])
            else:
              t_dict['demand_t+'+str(j + 1)].append(dataset['demand_pred'][numberofrows - 1])
    # day, day of hour columns, demand actual
    t_dict['day'].append(dataset['day'][i])
    t_dict['hour_of_day'].append(dataset['hourofday'][i])
    # t_dict['demand_act'].append(dataset['demand_act'][i])
    
    # day columns
    name='hour'+str((i%24)+1)
    #t_dict['demand_td0'].append(dataset2[name][i])
    day = int(i / 24)
    if(day>= lag) and (day < (numberofdays - lag)):
        for j in range(0, lag):
            t_dict['demand_d-'+str(j + 1)].append(dataset2[name][day - j - 1])
            t_dict['demand_d+'+str(j + 1)].append(dataset2[name][day + j + 1])
    elif (day < lag):
        for j in range(0, lag):
            if(j < day):
                t_dict['demand_d-'+str(j + 1)].append(dataset2[name][day - j - 1])
            else:
                t_dict['demand_d-'+str(j + 1)].append(dataset2[name][0])
        for j in range(0, lag):
             t_dict['demand_d+'+str(j + 1)].append(dataset2[name][day + j + 1])
    elif (day >= (numberofdays-lag)):
        for j in range(0, lag):
            t_dict['demand_d-'+str(j + 1)].append(dataset2[name][day - j - 1])
        for j in range(0, lag):
            if j < (numberofdays - day - 1):
              t_dict['demand_d+'+str(j + 1)].append(dataset2[name][day + j + 1])
            else:
              t_dict['demand_d+'+str(j + 1)].append(dataset2[name][numberofdays - 1])

'''
for i in range(1, lag + 1):
    print('demand_d-'+str(i))
    print(len(t_dict['demand_d-'+str(i)]))
    print('demand_d+'+str(i))
    print(len(t_dict['demand_d+'+str(i)]))
print('demand_td0')
print(len(t_dict['demand_td0']))
 '''   
# # constructing required modded dataset
mod_df = pd.DataFrame.from_dict(t_dict)
# mod_df.to_csv('../../DataSets/Demand_beforeModel.csv', sep=',', index=False)
# mod_df.to_csv('../../DataSets/Data_LeaderBoard/Demand_beforeModel.csv', sep=',', index=False)
mod_df.to_csv('../../DataSets/Data_TestSet_PrivateEvaluation/Demand_beforeModel.csv', index=False)