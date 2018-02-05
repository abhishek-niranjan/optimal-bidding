
import os
import pandas as pd
import pickle
import numpy as np
import scipy as sp


# solar_pred = '../../DataSets/Solar_ema.csv'                        ########### For Training Folder
# solar_pred = '../../DataSets/Data_LeaderBoard/Solar_ema.csv'         ########### For Leaderboard Folder
solar_pred = '../../DataSets/Data_TestSet_PrivateEvaluation/Solar_ema.csv'
solar_act = '../../DataSets/Solar_Train.csv'

pred_df = pd.read_csv(solar_pred)
act_df = pd.read_csv(solar_act)


train_df = pd.DataFrame(columns = ['t1','t2','t3','t4','t5','t6','t7','exp_ma','hour','slope1','slope2','curr_pred','actual'])
       

t1 = t2 = t3 = t4 = t5 = t6 = t7 = [0]*1
curr = hour = slope1 = slope2 = actual = [0]*1
exp_ma = [0]*1

for index, row  in pred_df.iterrows():
    # print index
    if(index<7):
        for i in range(0, 24):
            exp_ma = [pred_df.iloc[index,25]]
            actual = [act_df.iloc[index,i]]
            if(i==0):
                slope1 = [0]
                slope2 = [pred_df.iloc[index,i+1] - pred_df.iloc[index,i]]
            elif(i==23):
                slope2 = [0]
                slope1 = [pred_df.iloc[index,i] - pred_df.iloc[index,i-1]]
            else:
                slope1 = [pred_df.iloc[index,i] - pred_df.iloc[index,i-1]]
                slope2 = [pred_df.iloc[index,i+1] - pred_df.iloc[index,i]]
            curr = [pred_df.iloc[index,i]]
            hour = [i]
            t1, t2, t3, t4 = [pred_df.iloc[index,i]],[pred_df.iloc[index,i]], [pred_df.iloc[index,i]], [pred_df.iloc[index,i]]
            t5, t6, t7 = [pred_df.iloc[index,i]], [pred_df.iloc[index,i]], [pred_df.iloc[index,i]]
            temp = list(zip(t1,t2,t3,t4,t5,t6,t7,exp_ma,hour,slope1,slope2,curr, actual))
            # print(slope[0], slope2[0])
            temp = pd.DataFrame(temp, columns = ['t1','t2','t3','t4','t5','t6','t7','exp_ma','hour','slope1','slope2','curr_pred','actual'])
           # print(temp)
            train_df = train_df.append(temp, ignore_index=True)

    else:
        #exp_ma = get_exp_ma()
        for i in range(0, 24):
            t1, t2, t3, t4 = [pred_df.iloc[index-1,i]],[pred_df.iloc[index-2,i]], [pred_df.iloc[index-3,i]], [pred_df.iloc[index-4,i]]
            t5, t6, t7 = [pred_df.iloc[index-5,i]], [pred_df.iloc[index-6,i]], [pred_df.iloc[index-7,i]]
            curr = [pred_df.iloc[index,i]]
            hour = [i]
            if(i==0):
                slope1 = [0]
                slope2 = [pred_df.iloc[index,i+1] - pred_df.iloc[index,i]]
            elif(i==23):
                slope2 = [0]
                slope1 = [pred_df.iloc[index,i] - pred_df.iloc[index,i-1]]
            else:
                slope1 = [pred_df.iloc[index,i] - pred_df.iloc[index,i-1]]
                slope2 = [pred_df.iloc[index,i+1] - pred_df.iloc[index,i]]
            # slope2 = [act_df.iloc[index,i]-act_df.iloc[index,i-1]]
            actual = [act_df.iloc[index,i]]
            exp_ma = [pred_df.iloc[index,25]]
            temp = list(zip(t1,t2,t3,t4,t5,t6,t7,exp_ma,hour,slope1,slope2,curr, actual))
            # print(slope[0], slope2[0])
            temp = pd.DataFrame(temp, columns = ['t1','t2','t3','t4','t5','t6','t7','exp_ma','hour','slope1','slope2','curr_pred','actual'])
           # print(temp)
            train_df = train_df.append(temp, ignore_index=True)

# train_df.to_csv('../../DataSets/Solar_beforeModel.csv',index=False)
# train_df.to_csv('../../DataSets/Data_LeaderBoard/Solar_beforeModel.csv',index=False) 
train_df.to_csv('../../DataSets/Data_TestSet_PrivateEvaluation/Solar_beforeModel.csv',index=False)   
            

            
            
            
        



