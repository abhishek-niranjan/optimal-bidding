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


# test_file = '../../DataSets/Solar_beforeModel.csv' 		##### From Training Data
test_file = '../../DataSets/Data_LeaderBoard/Solar_beforeModel.csv'				#### From Leaderboard Data


test_data = pd.read_csv(test_file)
test_data = test_data.drop('actual',axis=1)
print test_data.head()

test_data = gl.SFrame(test_data)
model = gl.load_model('solar_module')
preds = model.predict(test_data)
preds = pd.DataFrame(np.asarray(preds), columns=['solar_model'])
preds['solar_model'] = preds['solar_model'].apply(lambda x: 0 if x<=0 else x)

no_days = (preds.shape[0]/24)
np_preds = np.asarray(preds)
solar_vals = [[0 for y in range(24)] for x in range(no_days)]

for i in range(no_days):
	for j in range(24):
		solar_vals[i][j] = float(np_preds[24*i+j])

columns = []
for i in range(1,25):
	fname = 'hour{}'.format(i)
	columns.append(str(fname))

solar_vals_df = pd.DataFrame(solar_vals, columns = columns)


#solar_vals_df.to_csv('../../DataSets/Solar_postModel.csv', index=False)   	### For Training Data
solar_vals_df.to_csv('../../DataSets/Data_LeaderBoard/Solar_postModel.csv', index=False)

# preds.to_csv('../../DataSets/Solar_Model.csv',index=False)
# print preds.head()



