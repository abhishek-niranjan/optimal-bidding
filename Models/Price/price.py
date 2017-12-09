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


train_file = '../../DataSets/price_train_processed2.csv'
train_data = pd.read_csv(train_file)
print train_data.head()

train_data = gl.SFrame(train_data)
folds = gl.cross_validation.KFold(train_data, 5)



###################### CROSS VALIDATION CODE ################################

out = []
model = [0]*5
for i, (train, valid) in enumerate(folds):
	model[i] = gl.boosted_trees_regression.create(train, max_iterations=250, max_depth=7, step_size=0.10,target='price_act',metric='rmse',verbose=False)
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
model[best_model_index].save('price_module')