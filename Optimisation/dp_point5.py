import os
import sys
import math
import numpy as np
import pandas as pd

no_days = 15

vec_data = pd.read_csv('../DataSets/All_Vector.csv')
# print vec_data.head()
vec_data['price_act'] = vec_data['price_act'].apply(lambda x: 7 if x>7 else x)
price = list(vec_data['price_act'])
price = price[:24*no_days]
solar = list(vec_data['solar_act'])
solar = solar[:24*no_days]
demand = list(vec_data['demand_act'])
demand = demand[:24*no_days]




demand = list(np.asarray(demand) - np.asarray(solar))
# print demand

# print len(price), len(solar), len(demand)
cost = [[sys.maxint for y in range(26)] for x in range(24*no_days)]
battery = [[sys.maxint for y in range(26)] for x in range(24*no_days)]
path = []


##### Fill for the first hour ######
for i in range(26):
	cost[0][i] = sys.maxint
for i in range(6):
	cost[0][i] = price[0]*(demand[0]+i)


carry = 0

for i in range(1,24*no_days):

	############################  HANDLE THE CASE WHERE SOLAR >= DEMAND ###############################################
	if(demand[i]<0):
		
		excess = (abs(demand[i])) + carry
		carry = 0
		excess = min(excess, 5)
		frac_part = excess - int(excess)
		# print "Carry: ", carry
		carry += frac_part
		# print "Carry: ", carry
		excess = int(excess)
		# print "Net Demand less than 0: ", i, " with excess value: ", excess
		for j in range(26):
			if(j==0):
				cost[i][j] = cost[i-1][j]
				battery[i][j] = j
				continue
			iterate = min(excess, j)
			for k in range(iterate+1):
			##################### HANDLE THE CASE OF BATTERY STATE: WE ARE NOT BUYING ANYTHING ###################################
				if(cost[i][j] >= cost[i-1][j-k]):
					battery[i][j] = j - k
				cost[i][j] = min(cost[i][j],cost[i-1][j-k])
		# print cost[i-1],cost[i]
		# print battery[i]
		demand[i] = 0.0
		continue
	####################################################################################################################


	########################### HANDLE THE FRACTIONAL PART CASE ########################################################
	
	# floor_demand = math.floor(demand[i])
	# if(floor_demand is demand[i]):
	# 	is_int = True
	# else:
	# 	frac_part = demand[i] - floor_demand
	# 	is_int = False

		################################### NOT NEEDED AT ALL ########################################
		
		# 	##### CHECK IF FRACTIONAL PART OF CURRENT DEMAND CAN BE FULFILLED FROM CARRY #######
		# if(carry >= frac_part):
		# 	carry -= frac_part
		# 	demand[i] -= frac_part
		# else:
		# 	############### FRACTIONAL PART IS GREATER THAN CURRENT RESIDUAL CARRY ############
		# 	carry = 1 - (frac_part - carry)
		
		# 	demand[i] = math.ceil(demand[i])
		################################################################################################

	demand[i] = demand[i] - 0.8*carry
	can_use = 5 - carry
	carry = 0

	#####################################################################################################################


	############################### CONTINUE WITH OUR DP ALGORITHM NOW #################################################
	for j in range(26):
		exc_encounters = 0
		for k in range(int(can_use)+1):
			try:
				term = cost[i-1][j+k] + price[i]*(math.ceil(demand[i] - (k*0.8)))
				if(cost[i][j] >= term):
					carry = math.ceil(demand[i]-(k*0.8)) - (demand[i]-(k*0.8))
					battery[i][j] = j + k 
				cost[i][j] = min(cost[i][j], term)
			except:
				exc_encounters += 1

			try:
				term = cost[i-1][j-k] + price[i]*(math.ceil(demand[i] + k))
				if(cost[i][j] >= term):
					carry = math.ceil(demand[i]+k) - (demand[i]+k)
					battery[i][j] = j - k 
				cost[i][j] = min(cost[i][j], term)
			except:
				exc_encounters += 1


	#####################################################################################################################



	######################################### OBSOLETE CODE: NOT NEEDED, KEEP IT AS BACKUP ###################################
		"""
		if(j==0):

			cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), 
								cost[i-1][j+2]+price[i]*(demand[i]-(2*0.8)), cost[i-1][j+3] + price[i]*(demand[i] - (3*0.8)),
								cost[i-1][j+4] + price[i]*(demand[i]-(4*0.8)), cost[i-1][j+5] + price[i]*(demand[i]-(5*0.8)))

			# ind =np.argmin(np.asarray([cost[i-1][j]+(price[i]*demand[i]), cost[i-1][j+1]+price[i]*(demand[i]-4)]))
			# battery[i][j] = ind

		elif(j==1):
			cost[i][j] = min(cost[i-1][j]+(price[i]*demand[i])
				, cost[i-1][j+1]+price[i]*(demand[i]-(1*0.8)), 
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
		"""
	##################################  OBSOLETE CODE ENDS HERE ########################################################

curr_day = 1
print min(cost[-1])



########################### CODE TO CHECK PER DAY PRICE ################################

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
#######################################################################################


###################### BEST TRANSITIONS DECISIONS PATH ###############################

path = []
path.append(np.argmin(np.asarray(cost[24*no_days-1])))
curr = battery[24*no_days-1-1][path[0]]

i = 24*no_days-1
while(i>0):
	path.append(curr)
	# print curr
	curr = battery[i-1][curr]
	i -=1
# print path[::-1]

######################################################################################



################## CREATE BID PRICE, BID QUANTITY FILE ###############################

path = path[::-1]
# print path#,demand, price

bid_price = [[0 for y in range(24)] for x in range(no_days)]
bid_quantity = [[0 for y in range(24)] for x in range(no_days)]

prev_state = 0
for i in range(no_days):
	for j in range(24):
		bid_price[i][j] = price[24*i+j]

		#####	IF SOLAR > DEMAND FOR THAT HOUR  #######
		if(demand[24*i+j] is 0.0):
			bid_quantity[i][j] = 0
			prev_state = path[24*i+j]
			continue

		curr_state = path[24*i+j]

		########  IF STATE OF BATTERY DIDN'T CHANGE FOR THAT HOUR #############
		if(curr_state is prev_state):
			bid_quantity[i][j] = math.ceil(demand[24*i+j])
			prev_state = curr_state
			continue

		################## IF BATTERY GETTING CHARGED ################
		if(curr_state > prev_state):
			hour_demand = demand[24*i+j]
			add = curr_state - prev_state
			bid_quantity[i][j] = math.ceil(hour_demand + add)
			prev_state = curr_state
			continue

		################# IF BATTERY GETTING DISCHARGED #################
		if(curr_state < prev_state):
			hour_demand = demand[24*i+j]
			diff = 0.8*(prev_state - curr_state)
			if(hour_demand-diff<0):
				bid_quantity[i][j] = 0
			else:
				bid_quantity[i][j] = math.ceil(hour_demand-diff)
			prev_state = curr_state
			continue

bid_price_df = pd.DataFrame(bid_price)
bid_quantity_df = pd.DataFrame(bid_quantity)
output_dir = '../Output/'
bid_price_file = output_dir + 'bid_price.csv'
bid_quantity_file = output_dir + 'bid_quantity.csv'
bid_price_df.to_csv(bid_price_file, index=False)
bid_quantity_df.to_csv(bid_quantity_file, index=False)











