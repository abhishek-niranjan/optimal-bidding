import os
import csv
import pandas as pd

# reponame = '../../DataSets/'
# reponame = '../../DataSets/Data_LeaderBoard'  ### For leaderboard
reponame = '../../DataSets/Data_TestSet_PrivateEvaluation'  ### For leaderboard

os.chdir(reponame)
days=30
def vectorise(filename):
    values=[]
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for i in range(24):
                name='hour'+str(i+1)
                values.append(float(row[name]))
    return values


demand_pred = vectorise('Demand_Test_pred.csv')
solar_pred = vectorise('Solar_Test_pred.csv')
price_pred = vectorise('Price_Test_pred.csv')

# demand_act, demand_pred = vectorise('Demand_Train.csv'), vectorise('Demand_Train_Pred.csv')
# solar_act, solar_pred = vectorise('Solar_Train.csv'), vectorise('Solar_Train_Pred.csv')
# price_act, price_pred = vectorise('Price_Train.csv'), vectorise('Price_Train_Pred.csv')

hourofday=[]
hour=[]
day=[]
no_days=int(len(demand_pred)/24)
for i in range(no_days):
    for j in range(24):
        hourofday.append(j+1)
        hour.append(24*i+j+1)
        day.append(i+1)
#print(no_days)
all_vectors = pd.DataFrame({'price_pred':price_pred, 'solar_pred':solar_pred, 'demand_pred':demand_pred,'hour':hour,'day':day,'hourofday':hourofday})
all_vectors.to_csv('All_Vector.csv', sep = ',', index = False) 