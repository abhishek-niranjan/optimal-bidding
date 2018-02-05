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


train_file = '../../DataSets/Price_beforeModel.csv'
train_data = pd.read_csv(train_file)
train_data = train_data.drop('day',axis=1)

X_train, X_test = train_test_split(train_data, test_size=0.33, random_state=10)

X_train['price_act'] += 0.4
print X_train.head()



params = {"objective: reg:linear"
          "booster": "gbtree",
          "eta": 0.20,
          "max_depth":3,
          "subsample": 0.75,
          "colsample_bytree": 0.65,
          "silent": 1,
          "eval_metric": "rmse",
          }

num_round = 400

"""
def custom_loss(preds, dtrain):
    labels = dtrain.get_label()
    df = preds - labels
    df = pd.DataFrame(df, columns=['val'])
    df['valg'] = df['val'].apply(lambda x: 5*abs(x) if x<0 else x)
    df['valh'] = df['val'].apply(lambda x: 25*abs(x) if x<0 else x)
    grad = df['valg'].as_matrix()
    hess = df['valh'].as_matrix()
    return preds-labels, grad
"""

dtrain = xgb.DMatrix(X_train.drop('price_act', axis=1), label=X_train['price_act'])
dtest = xgb.DMatrix(X_test.drop('price_act',axis=1), label=X_test['price_act'])

watchlist = [(dtrain,'train'), (dtest,'eval')]

bst = xgb.train(params, dtrain, num_round, watchlist)
pickle.dump(bst, open('xgb_model.p','wb'))




Y_oracle = X_test['price_td0']
Y_pred = bst.predict(dtest)   
Y_pred = np.asarray(Y_pred)


#################### FAALTU KA KAAM ####################
"""
def mod(x):
	if(x>0 and x<=2):
		x += 0.35
	elif(x >2 and x<=4):
		x += 0.40
	elif(x>4 and x<=5.5):
		x+= 0.40
	else:
		x += 0.30
	return x
"""
##########################################################


out = pd.DataFrame(X_test['price_act'], columns = ['price_act'])
out['pred'] = Y_pred
out['oracle'] = Y_oracle


oracle_rmse = mean_squared_error(out['price_act'],out['oracle'])
model_rmse = mean_squared_error(out['price_act'],out['pred'])



# out['pred'] = out['pred'].apply(lambda x: mod(x))
print out
out['diff'] = out['pred'] - out['price_act']
out['diff2'] = out['oracle'] - out['price_act']
print "___"*30
print oracle_rmse, model_rmse
print out['diff'].min(), out['diff2'].min()
out['ifpos'] = out['diff'].apply(lambda x: 1 if x>0 else 0)
# print out['ifpos'].sum()
out.to_csv("check_xgb.csv", index=False)

