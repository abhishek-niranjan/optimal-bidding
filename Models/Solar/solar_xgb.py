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


train_file = '../../DataSets/Solar_beforeModel.csv'
train_data = pd.read_csv(train_file)

X_train, X_test = train_test_split(train_data, test_size=0.33, random_state=10)
print X_train.head()



params = {"objective: reg:linear"
          "booster": "gbtree",
          "eta": 0.15  ,
          "max_depth":4 ,
          "subsample": 1,
          "colsample_bytree": 1,
          "silent": 1,
          "eval_metric": "rmse",
          }

num_round = 200


dtrain = xgb.DMatrix(X_train.drop('actual', axis=1), label=X_train['actual'])
dtest = xgb.DMatrix(X_test.drop('actual',axis=1), label=X_test['actual'])

watchlist = [(dtrain,'train'), (dtest,'eval')]

bst = xgb.train(params, dtrain, num_round, watchlist)
pickle.dump(bst, open('xgb_model.p','wb'))




Y_oracle = X_test['curr_pred']
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


out = pd.DataFrame(X_test['actual'], columns = ['actual'])
out['pred'] = Y_pred
out['oracle'] = Y_oracle
out.to_csv("check_xgb.csv", index=False)

oracle_rmse = mean_squared_error(out['actual'],out['oracle'])
model_rmse = mean_squared_error(out['actual'],out['pred'])

objects = ('Oracle', 'Model')
y_pos = np.arange(len(objects))
performance = [oracle_rmse, model_rmse]
 
fig, ax = plt.subplots()
ax.bar(y_pos, performance, align='center', alpha=0.5, color='brown')
plt.xticks(y_pos, objects, fontweight='bold')
for i, v in enumerate(performance):
    ax.text( i-.08 , v + .01, str(v)[:5], color='brown', fontweight='bold')
ax.set_axis_bgcolor('lightgrey')
plt.ylabel('rmse', fontweight='bold')
plt.title('Model Evaluation', fontweight='bold')
plt.tight_layout()


 

plt.show()