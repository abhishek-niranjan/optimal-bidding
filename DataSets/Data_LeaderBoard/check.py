import pandas as pd  
import numpy as np


df = pd.read_csv('InputDp.csv')

df['net_demand'] = df['demand_pred'] - df['solar_pred']
df['cost'] = np.multiply(df['net_demand'].as_matrix(), df['price_pred'].as_matrix())

def mod(x):
	if(x>0 and x<=2):
		x += 0.35
	elif(x >2 and x<=4):
		x += 0.40
	elif(x>4 and x<=5.5):
		x+= 0.40
	elif(x>5.5 and x<=7):
		x += 0.30
	else:
		x = 7
	return x


df['price_pred'] = df['price_pred'].apply(lambda x: mod(x))
print df.head()
df[['price_pred','net_demand']].to_csv('23.csv',index=False, header=False)