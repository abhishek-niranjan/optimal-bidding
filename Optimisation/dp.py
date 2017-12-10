import os
import sys
import numpy as np
import pandas as pd

no_days = 1

vec_data = pd.read_csv('../DataSets/All_Vector.csv')
print vec_data.head()
vec_data['price_act'] = vec_data['price_act'].apply(lambda x: 7 if x>7 else x)
price = list(vec_data['price_act'])
price = price[:24*no_days]
solar = list(vec_data['solar_act'])
solar = solar[:24*no_days]
demand = list(vec_data['demand_act'])
demand = demand[:24*no_days]




demand = list(np.asarray(demand) - np.asarray(solar))
# print demand

print len(price), len(solar), len(demand)
cost = [[sys.maxint for y in range(6)] for x in range(24*no_days)]
battery = [[sys.maxint for y in range(6)] for x in range(24*no_days)]
path = []


##### Fill for the first hour ######
for i in range(6):
	cost[0][i] = sys.maxint
for i in range(2):
	cost[0][i] = price[0]*(demand[0]+5*i)



for i in range(1,24*no_days):
	for j in range(6):
		if(j==0):
			cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-4))
			ind =np.argmin(np.asarray([cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-4)]))
			battery[i][j] = ind
		if(j==5):
			cost[i][j] = min(cost[i-1][j]+price[i]*demand[i], cost[i-1][j-1] + price[i]*(demand[i]+5))
			ind =np.argmin(np.asarray([cost[i-1][j]+(price[i]*demand[i]),cost[i-1][j-1] + price[i]*(demand[i]+5)]))
			battery[i][j] = 5-ind
		else:
			cost[i][j] = min(cost[i-1][j]+price[i]*demand[i], cost[i-1][j-1]+price[i]*(demand[i]+5), cost[i-1][j+1]+price[i]*(demand[i]-4))
			ind = np.argmin(np.asarray([cost[i-1][j]+price[i]*demand[i], cost[i-1][j-1]+price[i]*(demand[i]+5), cost[i-1][j+1]+price[i]*(demand[i]-4)]))
			if(ind==1):
				battery[i][j] = j-1
			elif(ind==2):
				battery[i][j] = j + 1
			else:
				battery[i][j] = j

print  cost[24*no_days-1]



##################  Checking the Path  #########################

"""
# for i,item in enumerate(cost):
# 	print demand[i]
# 	print item

# for item in battery:
# 	print item

# path.append(np.argmin(np.asarray(cost[23])))
# curr = battery[23][path[0]]

# i = 23
# while(i>0):
# 	path.append(curr)
# 	curr = battery[i-1][curr]
# 	i -=1

# print path[::-1]
"""


#################################################################