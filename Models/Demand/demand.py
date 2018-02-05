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


train_file = '../../DataSets/Demand_beforeModel.csv'
train_data = pd.read_csv(train_file)
train_data = train_data.drop('day',axis=1)
print train_data.head()

train_data = gl.SFrame(train_data)
folds = gl.cross_validation.KFold(train_data, 5)



###################### CROSS VALIDATION CODE ################################

out = []
model = [0]*5
for i, (train, valid) in enumerate(folds):
	model[i] = gl.boosted_trees_regression.create(train, max_iterations=250, max_depth=7, step_size=0.10,target='demand_act', row_subsample='0.6', column_subsample='0.7', metric='rmse',verbose=False)
	results = model[i].evaluate(valid)
	out.append(results)


best_model = sys.maxint
best_model_index = 0
for i, item in enumerate(out):
	print item
	if(item['rmse']<best_model):
		best_model = item['rmse']
		best_model_index = i

print best_model_index

##################### SAVE THE MODEL #################
model[best_model_index].save('demand_module')

######################## CHECK THE PREDICTED DEMAND ##################
test_data = pd.read_csv(train_file)
act = test_data['demand_act']
test_data = test_data.drop('demand_act',axis=1)
test_data = test_data.drop('day',axis=1)
test_data = gl.SFrame(test_data)

model = gl.load_model('demand_module')
preds = model.predict(test_data)
preds = pd.DataFrame(np.asarray(preds), columns=['demand_model'])
preds['demand_act'] = act
preds['diff'] = preds['demand_act'] - preds['demand_model']
preds['posdiff'] = preds['diff'].apply(lambda x: x if x>0 else 0)
preds['ifneg'] = preds['diff'].apply(lambda x: 1 if x<0 else 0)

preds.to_csv('demand_test.csv')
