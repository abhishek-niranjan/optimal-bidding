#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:58:47 2017

@author: bromance
"""
import pandas as pd
import numpy as np

# importing dataset

# file = '../../DataSets/All_Vector.csv'              #### For Training 
file = '../../DataSets/Data_LeaderBoard/All_Vector.csv'                  ##### For Leaderboard
dataset = pd.read_csv(file)

# dropping first stray column

#dataset = dataset.drop(dataset.columns[0], axis=1)

# building required dictionary

t_dict = {}

for count, i in enumerate(range(dataset.shape[0])):
    tt_dict = {}

    #price input (hour-wise)
    temp = np.array([])
    if (i == 0):
        temp = np.append(temp, [dataset['price_pred'][i], dataset['price_pred'][i]])
        temp = np.append(temp, dataset['price_pred'][:(i + 2 + 1)].values)
        #print([dataset['price_pred'][i], dataset['price_pred'][i]])
        #print(temp)
    elif (i == 1):
        temp = np.append(temp, [dataset['price_pred'][i - 1], dataset['price_pred'][i - 1]])
        temp = np.append(temp, dataset['price_pred'][:(i + 2 + 1)].values)
    elif (i == ((dataset.shape[0] - 1) - 1)):
        temp = np.append(temp, dataset['price_pred'][(i - 2):].values)
        temp = np.append(temp, [dataset['price_pred'][i + 1], dataset['price_pred'][i + 1]])
    elif (i == ((dataset.shape[0] - 1) - 0)):
        temp = np.append(temp, dataset['price_pred'][(i - 2):].values)
        temp = np.append(temp, [dataset['price_pred'][i], dataset['price_pred'][i]])
    else:
        temp = dataset['price_pred'][(i - 2):(i + 2 + 1)].values
    
    #solar input
    temp2 = np.array([])

    if (i == 0):
        temp2 = np.append(temp2, [dataset['solar_pred'][i], dataset['solar_pred'][i], dataset['solar_pred'][i]])
        temp2 = np.append(temp2, dataset['solar_pred'][:(i + 3 + 1)].values)
    elif (i == 1):
        temp2 = np.append(temp2, [dataset['solar_pred'][i - 1], dataset['solar_pred'][i - 1], dataset['solar_pred'][i - 1]])
        temp2 = np.append(temp2, dataset['solar_pred'][:(i + 3 + 1)].values)
    elif (i == 2):
        temp2 = np.append(temp2, [dataset['solar_pred'][i - 2], dataset['solar_pred'][i - 2], dataset['solar_pred'][i - 1]])
        temp2 = np.append(temp2, dataset['solar_pred'][:(i + 3 + 1)].values)
    elif (i == ((dataset.shape[0] - 1) - 2)):
        temp2 = np.append(temp2, dataset['solar_pred'][(i - 3):].values)
        temp2 = np.append(temp2, [dataset['solar_pred'][i + 1], dataset['solar_pred'][i + 2], dataset['solar_pred'][i + 2]])
    elif (i == ((dataset.shape[0] - 1) - 1)):
        temp2 = np.append(temp2, dataset['solar_pred'][(i - 3):].values)
        temp2 = np.append(temp2, [dataset['solar_pred'][i + 1], dataset['solar_pred'][i + 1], dataset['solar_pred'][i + 1]])
    elif (i == ((dataset.shape[0] - 1) - 0)):
        temp2 = np.append(temp2, dataset['solar_pred'][(i - 3):].values)
        temp2 = np.append(temp2, [dataset['solar_pred'][i], dataset['solar_pred'][i], dataset['solar_pred'][i]])
    else:
        temp2 = dataset['solar_pred'][(i - 3):(i + 3 + 1)].values
    
    tt_dict['price_t-2'] = temp[0]
    tt_dict['price_t-1'] = temp[1]
    tt_dict['price_t'] = temp[2]
    tt_dict['price_t+1'] = temp[3]
    tt_dict['price_t+2'] = temp[4]
    tt_dict['sol_t-3'] = temp2[0]
    tt_dict['sol_t-2'] = temp2[1]
    tt_dict['sol_t-1'] = temp2[2]
    tt_dict['sol_t'] = temp2[3]
    tt_dict['sol_t+1'] = temp2[4]
    tt_dict['sol_t+2'] = temp2[5]
    tt_dict['sol_t+3'] = temp2[6]
    if(i > 24):
        temp3 = dataset['price_pred'][(i - 24)]
        tt_dict['price_d-1'] = temp3
    else:
        tt_dict['price_d-1'] = dataset['price_pred'][:(i + 1)].values[0]
    if(i > 48):
        temp3 = dataset['price_pred'][(i - 48)]
        tt_dict['price_d-2'] = temp3
    else:
        tt_dict['price_d-2'] = dataset['price_pred'][:(i + 1)].values[0]
    if((dataset.shape[0] - 1 - i) > 24):
        temp3 = dataset['price_pred'][(i + 24)]
        tt_dict['price_d+1'] = temp3
    else:
        tt_dict['price_d+1'] = dataset['price_pred'][i:].values[-1]
    if((dataset.shape[0] - 1 - i)  > 48):
        temp3 = dataset['price_pred'][(i + 48)]
        tt_dict['price_d+2'] = temp3
    else:
        tt_dict['price_d+2'] = dataset['price_pred'][i:].values[-1]
    tt_dict['hourOfDay'] = dataset['hourofday'][i]
    tt_dict['Day'] = dataset['day'][i]
    # tt_dict['price_act'] = dataset['price_act'][i]                    ######### Uncomment while generating training data
    t_dict[count] = tt_dict

# constructing required modded dataset

mod_df = pd.DataFrame.from_dict(t_dict, orient='index')
# mod_df.to_csv('../../DataSets/Price_beforeModel.csv', sep=',', index=False)       ### Training

mod_df.to_csv('../../DataSets/Data_LeaderBoard/Price_beforeModel.csv', sep=',', index=False)            ### Leaderboard
