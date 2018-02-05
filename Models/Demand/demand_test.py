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
import sys
import graphlab as gl 


# test_file = '../../DataSets/Demand_beforeModel.csv' 							###### From Training 
test_file = '../../DataSets/Data_LeaderBoard/Demand_beforeModel.csv' 			######## From Leaderboard Dataset

test_data = pd.read_csv(test_file)
test_data = test_data.drop('day',axis=1)
# test_data = test_data.drop('demand_act',axis=1)

print test_data.head()

test_data = gl.SFrame(test_data)
model = gl.load_model('demand_module')
preds = model.predict(test_data)
preds = pd.DataFrame(np.asarray(preds), columns=['demand_model'])
preds['demand_model'] = preds['demand_model'].apply(lambda x: 0 if x<=0 else x)

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
# demand_vals_df.to_csv('../../DataSets/Demand_postModel.csv', index=False)					### For Training
demand_vals_df.to_csv('../../DataSets/Data_LeaderBoard/Demand_postModel.csv', index=False)	### For Leaderboard



# preds.to_csv('demand_test.csv',index=False)
# print preds.head()

