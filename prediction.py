#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:16:55 2022

@author: chen
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import _pickle as cPickle
import sys
from train import oneHot

if len(sys.argv)==2:
    path_model=str(sys.argv[0])
    path_csv=str(sys.argv[1])
else:
    path_model='RF_regressor_inf.pkl'
    path_csv='template_test_sample.csv'
    print("Missing arguments, refer to the default paths:\n",path_model, path_csv  )

# Try to load the trained model
with open(path_model, 'rb') as joblib:
    joblib_loaded = cPickle.load(joblib)
    
test_csv=pd.read_csv(path_csv)
featuresTest=np.array(test_csv.iloc[:, 1:])
assert featuresTest.shape[1]==12, "Check number of features"
assert featuresTest[:,2].max()<=2 and featuresTest[:,2].min()>=0, 'Index of Tobacco should be 0, 1 or 2'
assert featuresTest[:,9].max()<=4 and featuresTest[:,9].min()>=1, 'Killip class should be 1, 2, 3 or 4'
featuresTest=oneHot(featuresTest, 2, 2, 0)
featuresTest=oneHot(featuresTest, 11, 4, 1)

print(joblib_loaded.predict(featuresTest))