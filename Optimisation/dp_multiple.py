import os
import sys
import numpy as np
import pandas as pd

no_days = 50

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
cost = [[sys.maxint for y in range(26)] for x in range(24*no_days)]
battery = [[sys.maxint for y in range(26)] for x in range(24*no_days)]
path = []


##### Fill for the first hour ######
for i in range(26):
	cost[0][i] = sys.maxint
for i in range(6):
	cost[0][i] = price[0]*(demand[0]+i)



for i in range(1,24*no_days):
	if(demand[i]<0):
		excess = int(abs(demand[i]))
		for j in range(26):
			if(j==0):
				cost[i][j] = cost[i-1][j]
				continue
			iterate = min(excess, j)
			for k in range(iterate+1):
				cost[i][j] = min(cost[i][j],cost[i][j-k])
	else:
		for j in range(26):
			if(j==0):

				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), 
									cost[i-1][j+2]+price[i]*(demand[i]-(2*0.8)), cost[i-1][j+3] + price[i]*(demand[i] - (3*0.8)),
									cost[i-1][j+4] + price[i]*(demand[i]-(4*0.8)), cost[i-1][j+5] + price[i]*(demand[i]-(5*0.8)))

				# ind =np.argmin(np.asarray([cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-4)]))
				# battery[i][j] = ind

			elif(j==1):
				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), 
									cost[i-1][j+2]+price[i]*(demand[i]-(2*0.8)), cost[i-1][j+3] + price[i]*(demand[i] - (3*0.8)),
									cost[i-1][j+4] + price[i]*(demand[i]-(4*0.8)), cost[i-1][j+5] + price[i]*(demand[i]-(5*0.8)),
									cost[i-1][j-1] + price[i]*(demand[i]+1))

			elif(j==2):
				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), 
									cost[i-1][j+2]+price[i]*(demand[i]-(2*0.8)), cost[i-1][j+3] + price[i]*(demand[i] - (3*0.8)),
									cost[i-1][j+4] + price[i]*(demand[i]-(4*0.8)), cost[i-1][j+5] + price[i]*(demand[i]-(5*0.8)),
									cost[i-1][j-1] + price[i]*(demand[i]+1), cost[i-1][j-2] + price[i]*(demand[i]+2))

			elif(j==3):
				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), 
									cost[i-1][j+2]+price[i]*(demand[i]-(2*0.8)), cost[i-1][j+3] + price[i]*(demand[i] - (3*0.8)),
									cost[i-1][j+4] + price[i]*(demand[i]-(4*0.8)), cost[i-1][j+5] + price[i]*(demand[i]-(5*0.8)),
									cost[i-1][j-1] + price[i]*(demand[i]+1), cost[i-1][j-2] + price[i]*(demand[i]+2), 
									cost[i-1][j-3] + price[i]*(demand[i]+3))

			elif(j==4):
				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), 
									cost[i-1][j+2]+price[i]*(demand[i]-(2*0.8)), cost[i-1][j+3] + price[i]*(demand[i] - (3*0.8)),
									cost[i-1][j+4] + price[i]*(demand[i]-(4*0.8)), cost[i-1][j+5] + price[i]*(demand[i]-(5*0.8)),
									cost[i-1][j-1] + price[i]*(demand[i]+1), cost[i-1][j-2] + price[i]*(demand[i]+2), 
									cost[i-1][j-3] + price[i]*(demand[i]+3), cost[i-1][j-4] + price[i]*(demand[i]+4) )



			elif(j==25):
				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j-1]+price[i]*(demand[i]+1), 
									cost[i-1][j-2]+price[i]*(demand[i]+2), cost[i-1][j-3] + price[i]*(demand[i]+3),
									cost[i-1][j-4] + price[i]*(demand[i]+4), cost[i-1][j-5] + price[i]*(demand[i]+5))
				

			elif(j==24):
				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j-1]+price[i]*(demand[i]+1), 
									cost[i-1][j-2]+price[i]*(demand[i]+2), cost[i-1][j-3] + price[i]*(demand[i]+3),
									cost[i-1][j-4] + price[i]*(demand[i]+4), cost[i-1][j-5] + price[i]*(demand[i]+5),
									cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)))

			elif(j==23):
				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j-1]+price[i]*(demand[i]+1), 
									cost[i-1][j-2]+price[i]*(demand[i]+2), cost[i-1][j-3] + price[i]*(demand[i]+3),
									cost[i-1][j-4] + price[i]*(demand[i]+4), cost[i-1][j-5] + price[i]*(demand[i]+5),
									cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), cost[i-1][j+2]+price[i]*(demand[i]-(2*0.8)))

			elif(j==22):
				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j-1]+price[i]*(demand[i]+1), 
									cost[i-1][j-2]+price[i]*(demand[i]+2), cost[i-1][j-3] + price[i]*(demand[i]+3),
									cost[i-1][j-4] + price[i]*(demand[i]+4), cost[i-1][j-5] + price[i]*(demand[i]+5),
									cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), cost[i-1][j+2]+price[i]*(demand[i]-(2*0.8)), 
									cost[i-1][j+3] + price[i]*(demand[i] - (3*0.8)))
			

			elif(j==21):
				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j-1]+price[i]*(demand[i]+1), 
									cost[i-1][j-2]+price[i]*(demand[i]+2), cost[i-1][j-3] + price[i]*(demand[i]+3),
									cost[i-1][j-4] + price[i]*(demand[i]+4), cost[i-1][j-5] + price[i]*(demand[i]+5),
									cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), cost[i-1][j+2]+price[i]*(demand[i]-(2*0.8)), 
									cost[i-1][j+3] + price[i]*(demand[i] - (3*0.8)), cost[i-1][j+4] + price[i]*(demand[i]-(4*0.8)))




				# ind =np.argmin(np.asarray([cost[i-1][j]+(price[i]*demand[i]),cost[i-1][j-1] + price[i]*(demand[i]+5)]))
				# battery[i][j] = 5-ind
			else:
				cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), 
									cost[i-1][j+2]+price[i]*(demand[i]-(2*0.8)), cost[i-1][j+3] + price[i]*(demand[i] - (3*0.8)),
									cost[i-1][j+4] + price[i]*(demand[i]-(4*0.8)), cost[i-1][j+5] + price[i]*(demand[i]-(5*0.8)), 
									cost[i-1][j-1]+price[i]*(demand[i]+1), 
									cost[i-1][j-2]+price[i]*(demand[i]+2), cost[i-1][j-3] + price[i]*(demand[i]+3),
									cost[i-1][j-4] + price[i]*(demand[i]+4), cost[i-1][j-5] + price[i]*(demand[i]+5))

				# ind = np.argmin(np.asarray([cost[i-1][j]+price[i]*demand[i], cost[i-1][j-1]+price[i]*(demand[i]+5), cost[i-1][j+1]+price[i]*(demand[i]-4)]))
				# if(ind==1):
				# 	battery[i][j] = j-1
				# elif(ind==2):
				# 	battery[i][j] = j + 1
				# else:
				# 	battery[i][j] = j

curr_day = 1
# print min(cost[-1])
days_tot = []
while(curr_day<=no_days):
	days_tot.append(min(cost[24*curr_day-1]))
	# print curr_day, min(cost[24*curr_day-1])
	curr_day += 1

days_val = [0]*len(days_tot)
days_val[0] = days_tot[0]
for i in range(1, len(days_tot)):
	days_val[i] = days_tot[i]-days_tot[i-1]

for i, item in enumerate(days_val):
	print i, item