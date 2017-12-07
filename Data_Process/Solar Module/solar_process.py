# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:12:31 2017

@author: DELL-PC
"""
import os
import pandas as pd
import pickle
import numpy as np
import scipy as sp


solar_pred = '../../DataSets/Solar_Train_Pred.csv'
solar_act = '../../DataSets/Solar_Train.csv'

pred_df = pd.read_csv(solar_pred)
act_df = pd.read_csv(solar_act)
#print(pred_df.head(), act_df.head())


### Make a training dataframe
train_df = pd.DataFrame(columns = ['t-1','t-2','t-3','t-4','t-5','t-6','t-7','exp_ma','hour','slope','curr_pred','actual'])
#print(train_df.head(), type(train_df))


#def get_exp_ma(df):
        

t1 = t2 = t3 = t4 = t5 = t6 = t7 = [0]*1
curr = hour = slope = actual = [0]*1
exp_ma = [0]*1
for index, row  in pred_df.iterrows():
    if(index<7):
        continue
    if(index==40):
        break
    else:
        #exp_ma = get_exp_ma()
        for i in range(6, 18):
            t1, t2, t3, t4 = [pred_df.iloc[index-1,i]],[pred_df.iloc[index-2,i]], [pred_df.iloc[index-3,i]], [pred_df.iloc[index-4,i]]
            t5, t6, t7 = [pred_df.iloc[index-5,i]], [pred_df.iloc[index-6,i]], [pred_df.iloc[index-7,i]]
            curr = [pred_df.iloc[index,i]]
            hour = [i]
            slope = [pred_df.iloc[index,i] - pred_df.iloc[index,i-1]]
            slope2 = [act_df.iloc[index,i]-act_df.iloc[index,i-1]]
            actual = [act_df.iloc[index,i]]
            exp_ma = [0]#get_exp_ma(pred_df[:index])
            temp = list(zip(t1,t2,t3,t4,t5,t6,t7,exp_ma,hour,slope,curr, actual))
            print(slope[0], slope2[0])
            temp = pd.DataFrame(temp, columns = ['t-1','t-2','t-3','t-4','t-5','t-6','t-7','exp_ma','hour','slope','curr_pred','actual'])
           # print(temp)
            train_df = train_df.append(temp, ignore_index=True)
print(train_df.iloc[0:20], train_df.shape)
            
            
            
            
            
        



