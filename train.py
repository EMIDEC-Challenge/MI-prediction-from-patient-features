#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 21:07:52 2022

@author: chen
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import _pickle as cPickle


def oneHot(array, column, maxValue: int, minValue: int ):
    # Convert to one hot encoding
    # maxValue and minValue are the max and the min index of the class
    case=int(maxValue+1)-int(minValue)
    arrayOnehot=np.zeros([array.shape[0], case])
    arrayReturn=np.zeros([array.shape[0], array.shape[1]+case-1])
    for i in range (case):
        arrayOneCase=array[:,column]==(np.ones_like(array[:,column])*(i+int(minValue)))
        arrayOnehot[:,i]=arrayOneCase.astype(int)
    arrayReturn[:,0:column]=array[:,0:column]
    arrayReturn[:,column:column+case]=arrayOnehot
    arrayReturn[:,column+case:]=array[:,column+1:]
    return arrayReturn

classification=True 

features = pd.read_csv('template_train_sample.csv')
featuresLabel=(features.loc[:,'inf'])/(features.loc[:,'myo']) # the prediction output is the PIM (percentage of infarcted myocardium)
# featuresLabel=(features.loc[:,'pmo'])/(features.loc[:,'myo']) # if the target tissue is the PMO

if classification:
    featuresLabel=np.array(featuresLabel, dtype=bool) # if the model is a classifier

featuresTrain=features.iloc[:, 4:]

rf = RandomForestRegressor(n_estimators = 5000, random_state = 31)
rfc = RandomForestClassifier(n_estimators = 5000, random_state = 31)

# convert the feature Tobacco and Killip to one hot encoding
featuresTrainList=np.array(featuresTrain)
featuresTrainList=oneHot(featuresTrainList, 2, 2, 0) # Index of Tobacco should be 0, 1 or 2
featuresTrainList=oneHot(featuresTrainList, 11, 4, 1) # Killip class should be 1, 2, 3 or 4

# RF_regressor=rf.fit(featuresTrainList, featuresLabel)
RF_classifier=rfc.fit(featuresTrainList, featuresLabel)
with open('RF_classifier_inf.pkl', 'wb') as joblib:
    cPickle.dump(RF_classifier, joblib) 

