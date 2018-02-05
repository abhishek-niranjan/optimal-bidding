import os
import pandas as pd
import pickle
import math
from math import sqrt
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


train_file = '../../DataSets/Demand_beforeModel.csv'
train_data = pd.read_csv(train_file)
train_data = train_data.drop('day',axis=1)

X_train, X_test = train_test_split(train_data, test_size=0.33, random_state=10)

print X_train.head()



# params = {"objective: reg:linear"
#           "booster": "gbtree",
#           "eta": 0.20,
#           "max_depth": 1,
#           "subsample": 0.80,
#           "colsample_bytree": 0.75,
#           "silent": 1,
#           "eval_metric": "rmse",
#           }

params = {"objective: reg:linear"
         "booster": "gbtree",
         "eta": 0.20,
         "max_depth": 3,
         "subsample": 0.80,
         "colsample_bytree": 1,
         "silent": 1,
         "eval_metric": "rmse",
         "learning_rate": 0.06,
         "n_estimators": 1000,
         "min_child_weight":5,
         "gamma":0,
         "nthread":4,
         "scale_pos_weight":1,
         "seed":27,
}

num_round = 100


dtrain = xgb.DMatrix(X_train.drop('demand_act', axis=1), label=X_train['demand_act'])
dtest = xgb.DMatrix(X_test.drop('demand_act',axis=1), label=X_test['demand_act'])

watchlist = [(dtrain,'train'), (dtest,'eval')]

bst = xgb.train(params, dtrain, num_round, watchlist)
pickle.dump(bst, open('xgb_model.p','wb'))




Y_oracle = X_test['demand_td0']
Y_pred = bst.predict(dtest)   
Y_pred = np.asarray(Y_pred)

out = pd.DataFrame(X_test['demand_act'], columns = ['demand_act'])
out['pred'] = Y_pred
out['oracle'] = Y_oracle


oracle_rmse = mean_squared_error(out['demand_act'],out['oracle'])
model_rmse = mean_squared_error(out['demand_act'],out['pred'])

print sqrt(oracle_rmse), sqrt(model_rmse)

out['diff'] = out['pred'] - out['demand_act']
out['ifpos'] = out['diff'].apply(lambda x: 1 if x>0 else 0) 
# print out['ifpos'].sum()
# print out['diff'].max(), out['diff'].min()
out.to_csv("check_xgb.csv", index=False)

