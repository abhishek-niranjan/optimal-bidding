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


train_file = '../../DataSets/Solar_Train_Input.csv'
train_data = pd.read_csv(train_file)
check_data = train_data[(train_data['hour'] >= 6) & (train_data['hour']<=18)]
# print check_data.head()
print train_data.head()


train_data = gl.SFrame(train_data)
check_data = gl.SFrame(check_data)
folds = gl.cross_validation.KFold(train_data, 5)



###################### CROSS VALIDATION CODE ################################

out = []
model = [0]*5
for i, (train, valid) in enumerate(folds):
	model[i] = gl.boosted_trees_regression.create(train, max_iterations=250, max_depth=5, step_size=0.08,target='actual',metric='rmse',verbose=True)
	# results = model.predict(valid)
	# print [results, valid['actual']]
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
# model[best_model_index].save('solar_module')

# model = gl.boosted_trees_regression.create(check_data, max_iterations=250, max_depth=3, step_size=0.12,target='actual',metric='rmse')
