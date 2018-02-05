#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:58:47 2017

@author: bromance
"""
import pandas as pd
import numpy as np

# importing dataset

file = '../../DataSets/All_Vector.csv'
dataset = pd.read_csv(file)

# dropping first stray column

#dataset = dataset.drop(dataset.columns[0], axis=1)

# building required dictionary

t_dict = {}

for count, i in enumerate(range(dataset.shape[0])):
    tt_dict = {}
     #demand input (hour-wise)
    temp = np.array([])
    if (i == 0):
        temp = np.append(temp, [dataset['demand_pred'][i], dataset['demand_pred'][i]])
        temp = np.append(temp, dataset['demand_pred'][:(i + 2 + 1)].values)
        #print([dataset['demand_pred'][i], dataset['demand_pred'][i]])
        print(temp)
    elif (i == 1):
        temp = np.append(temp, [dataset['demand_pred'][i - 1], dataset['demand_pred'][i - 1]])
        temp = np.append(temp, dataset['demand_pred'][:(i + 2 + 1)].values)
    elif (i == ((dataset.shape[0] - 1) - 1)):
        temp = np.append(temp, dataset['demand_pred'][(i - 2):].values)
        temp = np.append(temp, [dataset['demand_pred'][i + 1], dataset['demand_pred'][i + 1]])
    elif (i == ((dataset.shape[0] - 1) - 0)):
        temp = np.append(temp, dataset['demand_pred'][(i - 2):].values)
        temp = np.append(temp, [dataset['demand_pred'][i], dataset['demand_pred'][i]])
    else:
        temp = dataset['demand_pred'][(i - 2):(i + 2 + 1)].values
    #temp = dataset['demand_pred'][(i - 2):(i + 2 + 1)].values
    tt_dict['demand_t-2'] = temp[0]
    tt_dict['demand_t-1'] = temp[1]
    tt_dict['demand_t'] = temp[2]
    tt_dict['demand_t+1'] = temp[3]
    tt_dict['demand_t+2'] = temp[4]
    tt_dict['hourOfDay'] = dataset['hourofday'][i]
    tt_dict['Day'] = dataset['day'][i]
    tt_dict['demand_act'] = dataset['demand_act'][i]
    if(i > 24):
        temp3 = dataset['demand_pred'][(i - 24)]
        tt_dict['demand_d-1'] = temp3
    else:
        tt_dict['demand_d-1'] = dataset['demand_pred'][:(i + 1)].values[0]
    if(i > 48):
        temp3 = dataset['demand_pred'][(i - 48)]
        tt_dict['demand_d-2'] = temp3
    else:
        tt_dict['demand_d-2'] = dataset['demand_pred'][:(i + 1)].values[0]
    if((dataset.shape[0] - 1 - i) > 24):
        temp3 = dataset['demand_pred'][(i + 24)]
        tt_dict['demand_d+1'] = temp3
    else:
        tt_dict['demand_d+1'] = dataset['demand_pred'][i:].values[-1]
    if((dataset.shape[0] - 1 - i)  > 48):
        temp3 = dataset['demand_pred'][(i + 48)]
        tt_dict['demand_d+2'] = temp3
    else:
        tt_dict['demand_d+2'] = dataset['demand_pred'][i:].values[-1]
    t_dict[count] = tt_dict

# constructing required modded dataset

mod_df = pd.DataFrame.from_dict(t_dict, orient='index')
mod_df.to_csv('../../DataSets/Demand_beforeModel.csv', sep=',', index=False)
