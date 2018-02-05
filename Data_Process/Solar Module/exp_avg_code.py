#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:23:57 2017

@author: bromance
"""
import pandas as pd

# importing data to dataframe

# solar_pred = '../../DataSets/Solar_Train_Pred.csv'
solar_pred = '../../DataSets/Data_TestSet_PrivateEvaluation/Solar_Test_pred.csv'  ### For leaderboard

df = pd.read_csv(solar_pred)

# creating row wise sum column

df['row_sum'] = df.sum(axis=1).values

# calculating exponential moving average

df['exp_ma'] = df['row_sum'].ewm(halflife=20, adjust=False).mean()
#pd.ewma(df['row_sum'], span=20, adjust=False)

# dumping processed dataframe

# file_name = '../../DataSets/Solar_ema.csv'
file_name = '../../DataSets/Data_TestSet_PrivateEvaluation/Solar_ema.csv'  ### For leaderboard

df.to_csv(file_name, sep=',', index=False)
