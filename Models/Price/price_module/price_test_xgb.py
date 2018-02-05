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



test_file = '../../DataSets/Price_beforeModel.csv'    #### For Training
# test_file = '../../DataSets/Data_LeaderBoard/Price_beforeModel.csv'			#### For Leaderboard


test_data = pd.read_csv(test_file)

test_data = test_data.drop('day', axis=1)
# test_data = test_data.drop('price_act',axis=1)			######## uncomment while running for training data
print test_data.head()

X_test = test_data
dtest = xgb.DMatrix(X_test.drop('price_act',axis=1))

bst = pickle.load(open('xgb_model.p','r'))

Y_pred = bst.predict(dtest)   
Y_pred = np.asarray(Y_pred)


preds = pd.DataFrame(Y_pred, columns=['price_model'])


no_days = (preds.shape[0]/24)
# print no_days
np_preds = np.asarray(preds)
price_vals = [[0 for y in range(24)] for x in range(no_days)]

for i in range(no_days):
	for j in range(24):
		price_vals[i][j] = float(np_preds[24*i+j])

columns = []
for i in range(1,25):
	fname = 'hour{}'.format(i)	
	columns.append(str(fname))

price_vals_df = pd.DataFrame(price_vals, columns = columns)
price_vals_df.to_csv('../../DataSets/Price_postModel.csv', index=False)				#### Training

# price_vals_df.to_csv('../../DataSets/Data_LeaderBoard/Price_postModel.csv', index=False)	#### Leaderboard


