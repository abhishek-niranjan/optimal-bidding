#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:58:47 2017

@author: bromance
"""
import pandas as pd
import numpy as np

# importing dataset

file = 'dataset_vec.csv'
dataset = pd.read_csv(file)

# dropping first stray column

dataset = dataset.drop(dataset.columns[0], axis=1)

# building required dictionary

temp = np.array([])
t_dict = {}

for count, i in enumerate(range(3, (dataset.shape[0] - 3))):
    tt_dict = {}
    temp = dataset['price_pred'][(i - 2):(i + 2 + 1)].values
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
        tt_dict['prevDay_price'] = temp3
    else:
        tt_dict['prevDay_price'] = -1
    tt_dict['hourOfDay'] = dataset['hourofday'][i]
    tt_dict['Day'] = dataset['day'][i]
    tt_dict['price_act'] = dataset['price_act'][i]
    t_dict[count] = tt_dict

# constructing required modded dataset

mod_df = pd.DataFrame.from_dict(t_dict, orient='index')
mod_df.to_csv('price_train_processed2.csv', sep=',', index=False)
