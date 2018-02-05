# optimal-bidding
Repository for Inter-IIT tech 2017 meet event Optimal Bidding. 


Requirements: 
1. Bash Shell
2. xgboost library 0.62 version
3. Numpy, Pandas, Scipy, Matplotlib
4. Sklearn Library


########## THIS WILL PRODUCE THE OUTPUT FOR THE PRIVATE FILES #####################
To Run from the scratch: (Involve Data Processing, Solar, Price and Demand Estimation, and Then Optimization Algorithm)
Run job.sh file on shell: 
	Open Terminal with bash shell
	Execute bash job.sh


########## TO RUN ON LEADERBOARD FILES #####################################
cd ./Optimisation/
Open dp_point5.py

change LINE #7 
no_of_days = 100  -----> no_of_days = 100

Uncomment LINE #10
Comment LINE #11

vec_data = pd.read_csv('../DataSets/Data_LeaderBoard/InputDp.csv') 		### For Leaderboard
#vec_data = pd.read_csv('../DataSets/Data_TestSet_PrivateEvaluation/InputDp.csv')

And then execute 
python dp_point5.py









################ OUTPUT ################################

Outputs are stored in optimal-bidding/Output Folder with the name of 11.csv
after each run.



