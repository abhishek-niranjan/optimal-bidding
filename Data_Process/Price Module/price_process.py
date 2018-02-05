import pandas as pd
import numpy as np

# importing dataset


# for training data (900 days)
# file = '../../DataSets/All_Vector.csv'
# file2 = '../../DataSets/Price_Train_Pred.csv'
# file3 = '../../DataSets/Solar_Train_Pred.csv'


# for public leadeboard data (50 days)
# file = '../../DataSets/Data_LeaderBoard/All_Vector.csv'
# file2 = '../../DataSets/Data_LeaderBoard/Price_LB_pred.csv'
# file3 = '../../DataSets/Data_LeaderBoard/Solar_LB_pred.csv'

# For test Leaderboard Data (100 days)
file = '../../DataSets/Data_TestSet_PrivateEvaluation/All_Vector.csv'
file2 = '../../DataSets/Data_TestSet_PrivateEvaluation/Price_Test_pred.csv'
file3 = '../../DataSets/Data_TestSet_PrivateEvaluation/Solar_Test_pred.csv'


dataset = pd.read_csv(file)
dataset2 = pd.read_csv(file2)
dataset3 = pd.read_csv(file3)

lag = 5
columns = []
t_dict = {}
t_dict['price_td0'] = []
t_dict['solar_td0'] = []
t_dict['day'] = []
t_dict['hour_of_day'] = []
# t_dict['price_act'] = []

for i in range(1, lag + 1):
    t_dict['price_t-'+str(i)] = []
    t_dict['price_d-'+str(i)] = []
    t_dict['price_t+'+str(i)] = []
    t_dict['price_d+'+str(i)] = []
    t_dict['solar_t-'+str(i)] = []
    t_dict['solar_d-'+str(i)] = []
    t_dict['solar_t+'+str(i)] = []
    t_dict['solar_d+'+str(i)] = []

numberofrows=dataset.shape[0]
numberofdays=dataset2.shape[0]

for i in range(numberofrows):
    # hour columns
    t_dict['price_td0'].append(dataset['price_pred'][i])
    # t_dict['price_act'].append(dataset['price_act'][i])
    if(i>= lag) and (i < (numberofrows - lag)):
        for j in range(0, lag):
            t_dict['price_t-'+str(j + 1)].append(dataset['price_pred'][i - j - 1])
            t_dict['price_t+'+str(j + 1)].append(dataset['price_pred'][i + j + 1])
    elif (i < lag):
        for j in range(0, lag):
            if(j < i):
                t_dict['price_t-'+str(j + 1)].append(dataset['price_pred'][i - j - 1])
            else:
                t_dict['price_t-'+str(j + 1)].append(dataset['price_pred'][0])
        for j in range(0, lag):
             t_dict['price_t+'+str(j + 1)].append(dataset['price_pred'][i + j + 1])
    elif (i >= (numberofrows-lag)):
        for j in range(0, lag):
            t_dict['price_t-'+str(j + 1)].append(dataset['price_pred'][i - j - 1])
        for j in range(0, lag):
            if j < (numberofrows - i - 1):
              t_dict['price_t+'+str(j + 1)].append(dataset['price_pred'][i + j + 1])
            else:
              t_dict['price_t+'+str(j + 1)].append(dataset['price_pred'][numberofrows - 1])
    # day and day of hour columns
    t_dict['day'].append(dataset['day'][i])
    t_dict['hour_of_day'].append(dataset['hourofday'][i])

    # day columns
    name='hour'+str((i%24)+1)
    # t_dict['price_td0'].append(dataset2[name][i])
    day = int(i / 24)
    if(day>= lag) and (day < (numberofdays - lag)):
        for j in range(0, lag):
            t_dict['price_d-'+str(j + 1)].append(dataset2[name][day - j - 1])
            t_dict['price_d+'+str(j + 1)].append(dataset2[name][day + j + 1])
    elif (day < lag):
        for j in range(0, lag):
            if(j < day):
                t_dict['price_d-'+str(j + 1)].append(dataset2[name][day - j - 1])
            else:
                t_dict['price_d-'+str(j + 1)].append(dataset2[name][0])
        for j in range(0, lag):
             t_dict['price_d+'+str(j + 1)].append(dataset2[name][day + j + 1])
    elif (day >= (numberofdays-lag)):
        for j in range(0, lag):
            t_dict['price_d-'+str(j + 1)].append(dataset2[name][day - j - 1])
        for j in range(0, lag):
            if j < (numberofdays - day - 1):
              t_dict['price_d+'+str(j + 1)].append(dataset2[name][day + j + 1])
            else:
              t_dict['price_d+'+str(j + 1)].append(dataset2[name][numberofdays - 1])
###Solar train
    t_dict['solar_td0'].append(dataset['solar_pred'][i])
    if(i>= lag) and (i < (numberofrows - lag)):
        for j in range(0, lag):
            t_dict['solar_t-'+str(j + 1)].append(dataset['solar_pred'][i - j - 1])
            t_dict['solar_t+'+str(j + 1)].append(dataset['solar_pred'][i + j + 1])
    elif (i < lag):
        for j in range(0, lag):
            if(j < i):
                t_dict['solar_t-'+str(j + 1)].append(dataset['solar_pred'][i - j - 1])
            else:
                t_dict['solar_t-'+str(j + 1)].append(dataset['solar_pred'][0])
        for j in range(0, lag):
             t_dict['solar_t+'+str(j + 1)].append(dataset['solar_pred'][i + j + 1])
    elif (i >= (numberofrows-lag)):
        for j in range(0, lag):
            t_dict['solar_t-'+str(j + 1)].append(dataset['solar_pred'][i - j - 1])
        for j in range(0, lag):
            if j < (numberofrows - i - 1):
              t_dict['solar_t+'+str(j + 1)].append(dataset['solar_pred'][i + j + 1])
            else:
              t_dict['solar_t+'+str(j + 1)].append(dataset['solar_pred'][numberofrows - 1])

    name='hour'+str((i%24)+1)
    day = int(i / 24)
    if(day>= lag) and (day < (numberofdays - lag)):
        for j in range(0, lag):
            t_dict['solar_d-'+str(j + 1)].append(dataset2[name][day - j - 1])
            t_dict['solar_d+'+str(j + 1)].append(dataset2[name][day + j + 1])
    elif (day < lag):
        for j in range(0, lag):
            if(j < day):
                t_dict['solar_d-'+str(j + 1)].append(dataset2[name][day - j - 1])
            else:
                t_dict['solar_d-'+str(j + 1)].append(dataset2[name][0])
        for j in range(0, lag):
             t_dict['solar_d+'+str(j + 1)].append(dataset2[name][day + j + 1])
    elif (day >= (numberofdays-lag)):
        for j in range(0, lag):

            t_dict['solar_d-'+str(j + 1)].append(dataset2[name][day - j - 1])
        for j in range(0, lag):
            if j < (numberofdays - day - 1):
              t_dict['solar_d+'+str(j + 1)].append(dataset2[name][day + j + 1])
            else:
              t_dict['solar_d+'+str(j + 1)].append(dataset2[name][numberofdays - 1])
'''
print (len(t_dict['solar_td0']))
print(len(t_dict['price_td0']))
print (len(t_dict['day']))
print (len(t_dict['hour_of_day']))

for i in range(0,lag):
    print (len(t_dict['price_t-'+str(i+1)]))
    print (len(t_dict['solar_t-'+str(i+1)]))
    print (len(t_dict['price_t+'+str(i+1)]))
    print (len(t_dict['solar_t+'+str(i+1)]))
'''
# # constructing required modded dataset
mod_df = pd.DataFrame.from_dict(t_dict)
# mod_df.to_csv('../../DataSets/Price_beforeModel.csv', sep=',', index=False)
# mod_df.to_csv('../../DataSets/Data_LeaderBoard/Price_beforeModel.csv', sep=',', index=False)
mod_df.to_csv('../../DataSets/Data_TestSet_PrivateEvaluation/Price_beforeModel.csv', index=False)