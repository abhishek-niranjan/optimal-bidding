import os
import pandas as pd
import pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting

from functools import partial # to reduce df memory consumption by applying to_numeric

import sklearn 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
import graphlab as gl 
from sklearn.model_selection import train_test_split
import xgboost as xgb 


# test_file = '../../DataSets/Demand_beforeModel.csv' 							###### From Training 
# test_file = '../../DataSets/Data_LeaderBoard/Demand_beforeModel.csv' 			######## From Leaderboard Dataset
test_file = '../../DataSets/Data_TestSet_PrivateEvaluation/Demand_beforeModel.csv'


test_data = pd.read_csv(test_file)
test_data = test_data.drop('day', axis=1)
# print test_data.head()

X_test = test_data
# X_test = X_test.drop('demand_act',axis=1)
dtest = xgb.DMatrix(X_test)

bst = pickle.load(open('xgb_model.p','r'))

Y_pred = bst.predict(dtest)   
Y_pred = np.asarray(Y_pred)


preds = pd.DataFrame(Y_pred, columns=['demand_model'])


no_days = (preds.shape[0]/24)
# print no_days
np_preds = np.asarray(preds)
demand_vals = [[0 for y in range(24)] for x in range(no_days)]

for i in range(no_days):
	for j in range(24):
		demand_vals[i][j] = float(np_preds[24*i+j])

columns = []
for i in range(1,25):
	fname = 'hour{}'.format(i)	
	columns.append(str(fname))

demand_vals_df = pd.DataFrame(demand_vals, columns = columns)
# demand_vals_df.to_csv('../../DataSets/Demand_postModel.csv', index=False)				#### Training

# demand_vals_df.to_csv('../../DataSets/Data_LeaderBoard/Demand_postModel.csv', index=False)	#### Leaderboard
demand_vals_df.to_csv('../../DataSets/Data_TestSet_PrivateEvaluation/Demand_postModel.csv', index=False)
