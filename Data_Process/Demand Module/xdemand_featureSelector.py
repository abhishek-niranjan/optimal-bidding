#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 18:30:04 2018

@author: bromance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 4


# loading dataset

train_file = '../../DataSets/Demand_all_features_processed.csv'
train_data = pd.read_csv(train_file)

# adj r squared calculator


def adjr2(true, pred, n, k):
    r2 = r2_score(true, pred)
    r2_adj = 1 - (((1 - r2) * (n - 1))/(n - k - 1))
    return r2, r2_adj

# doing some tweaks

# removing day column from the dataset
train_data = train_data.drop('day', axis=1)
# split
X_train, X_test = train_test_split(train_data, test_size=0.33, random_state=10)
# adding constant to demand_act
#X_train['demand_act'] += 0.4


# AV model fitting function


def modelfit(alg, dtrain, predictors, dtest=X_test, useTrainCV=True, cv_folds=5,
             early_stopping_rounds=50):


    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['demand_act'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds, metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['demand_act'], eval_metric='rmse')

    #test set predictions
    dtest_predictions = alg.predict(dtest[predictors])
    dtrain_predictions = alg.predict(dtrain[predictors])

    #scores
    rmse = sqrt(mean_squared_error(dtest['demand_act'].values, dtest_predictions))
    mae = np.amax(np.fabs(dtest['demand_act'].values - dtest_predictions))
    bid_loss_percentage = (np.sum(((dtest['demand_act'].values - dtest_predictions) > 0)) / dtest.shape[0]) * 100
    r2, adj_r2 = adjr2(dtrain['demand_act'].values, dtrain_predictions,
                   dtrain.shape[0], (dtrain.shape[1] - 1))


    #Print model report:
    print("\n###### Model Report #####\n")
    print("RMSE = {:.5f}".format(rmse))
    print("MAE = {:.5f}".format(mae))
    print('Unsatisfied demands percentage = {:.5f}'.format(bid_loss_percentage))
    print('R^2  = {:.5f}'.format(r2))
    print('adjusted R^2 = {:.5f}'.format(adj_r2))
    
    #returning scores
    return rmse, mae, bid_loss_percentage, r2, adj_r2
'''
    #plotting feature importance graph
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.xlabel('Features')
    plt.savefig('feature_importance_plot(with all features, +0.4).png', dpi=600, bbox_inches='tight')
'''

# columns a.k.a predictors constructing function


def column_set_creator(n):
    column_set = []
    i = 0
    default = ['hour_of_day', 'demand_td0', 'demand_act']
    for j in range(n):
        column_set.append('demand_t-' + str(i + 1))
        column_set.append('demand_t+' + str(i + 1))
        column_set.append('demand_d-' + str(i + 1))
        column_set.append('demand_d+' + str(i + 1))
        i = i + 1
    column_set.extend(default)
    return column_set

'''
def column_set_creator(n):
    column_set = []
    i = 0
    for j in range(n):
        if(j % 2 == 0):
            column_set.append('price_t-' + str(i + 1))
            column_set.append('price_t+' + str(i + 1))
        else:
            column_set.append('price_d-' + str(i + 1))
            column_set.append('solar_d-' + str(i + 1))
            i = i + 1
    return column_set

count = 5
default = ['hour_of_day', 'price_td0', 'demand_act']
columns = column_set_creator(count)
columns.extend(default)
'''

# results recorder

recorder = {}
recorder['lag_range'] = []
recorder['rmse'] = []
recorder['unsatisfied_demands_percentage'] = []
recorder['mae'] = []
recorder['r2'] = []
recorder['adjusted_r2'] = []

################## xgboost regressor model feature selection ################

def fselection(lag):
    for i in range(1, lag + 1):
        predictors = column_set_creator(i)
        predictors.remove('demand_act')  # IMPORTANT!!

        #model creation
        xmodel = xgb.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, 
                    objective= 'reg:linear', scale_pos_weight=1, seed=7)

        #fitting model
        rmse, mae, bid_loss_percentage, r2, adj_r2 = modelfit(xmodel, X_train, predictors)

        #recording results
        recorder['lag_range'].append(i)
        recorder['rmse'].append(rmse)
        recorder['unsatisfied_demands_percentage'].append(bid_loss_percentage)
        recorder['mae'].append(mae)
        recorder['adjusted_r2'].append(adj_r2)
        recorder['r2'].append(r2)

# main function call

fselection(15)

# dumping results csv

df = pd.DataFrame(recorder,
                  columns=['lag_range', 'rmse', 'mae', 'adjusted_r2', 'r2', 'unsatisfied_demands_percentage'])
df.to_csv('demand_feature_results.csv', index=False, sep=',')
