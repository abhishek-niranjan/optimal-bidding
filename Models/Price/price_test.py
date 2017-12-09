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


test_file = '../../DataSets/price_train_processed2.csv'
test_data = pd.read_csv(test_file)
test_data = test_data.drop('price_act',axis=1)
print test_data.head()

test_data = gl.SFrame(test_data)
model = gl.load_model('price_module')
preds = model.predict(test_data)
preds = pd.DataFrame(np.asarray(preds), columns=['price_model'])
# preds['price_model'] = preds['demand_model'].apply(lambda x: 0 if x<=0 else x)
preds.to_csv('price_test.csv',index=False)
print preds.head()
