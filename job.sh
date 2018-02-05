#!/bin/sh
cd './Data_Process/All Vector' && \
python allvector1.py && \
echo "AllVector1 ran"

cd '../Solar Module/' && \
python exp_avg_code.py && \
python solar_process.py && \
echo "Solar Data Process Done"

cd '../Demand Module/' && \
python demand_process.py && \
echo "Demand Data Process Done"

cd '../Price Module/' && \
python price_process.py && \
echo "Price Data Process Done"

cd '../../Models/Solar/' &&\
python solar_test_xgb.py && \
echo "Solar Model Estimation Done"

cd '../Demand/' && \
python demand_test_xgb.py
echo "Demand Model Estimation Done"

cd '../Price/' && \
python price_test_xgb.py 
echo "Price Model Estimation Done"


cd '../../Data_Process/All Vector/' && \
python allvector2.py && \
echo "allvector2 ran"

cd '../../Optimisation/' && \
python dp_point5.py && \
echo "Optimisation Done!! Check Output Directory"
