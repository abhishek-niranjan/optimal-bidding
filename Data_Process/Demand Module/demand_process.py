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

temp = np.array([])
t_dict = {}

for count, i in enumerate(range(2, (dataset.shape[0] - 2))):
    tt_dict = {}
    temp = dataset['demand_pred'][(i - 2):(i + 2 + 1)].values
    tt_dict['demand_t-2'] = temp[0]
    tt_dict['demand_t-1'] = temp[1]
    tt_dict['demand_t'] = temp[2]
    tt_dict['demand_t+1'] = temp[3]
    tt_dict['demand_t+2'] = temp[4]
    tt_dict['hourOfDay'] = dataset['hourofday'][i]
    tt_dict['Day'] = dataset['day'][i]
    tt_dict['demand_act'] = dataset['demand_act'][i]
    t_dict[count] = tt_dict

# constructing required modded dataset

mod_df = pd.DataFrame.from_dict(t_dict, orient='index')
mod_df.to_csv('../../DataSets/Demand_Train_Pred_processed.csv', sep=',', index=False)
