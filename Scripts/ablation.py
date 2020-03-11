# -*- coding: utf-8 -*-
"""
Created on Fri March 11 2020

@author: Farzin Negahbani, Ehsan Asali
"""

import numpy as np
import pandas as pd 
from time import time
from lightgbm import LGBMClassifier
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score


#enc = OneHotEncoder()


#enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])



#drop_enc.categories_

#drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()





def train_evaluate(name, clf, train_data, labels, k_folds):
    skf = StratifiedKFold(n_splits=k_folds)
    acc, auc = [], []
    for train, test in skf.split(train_data,labels):
        xt, xv, yt, yv = train_data[train,:], train_data[test,:], labels[train], labels[test]
        clf.fit(xt, yt)
        yhat = clf.predict(xv)
        proba = clf.predict_proba(xv)
        acc.append(np.mean(yhat == yv))
        auc.append(roc_auc_score(yv, proba,multi_class='ovr'))

    acc_mean, acc_std = np.mean(acc), np.std(acc)
    auc_mean, auc_std = np.mean(auc), np.std(auc)
    print ("In Case : ",name)
    print ('accuracy: {0:.3f} +/- {1:.3f}'.format(acc_mean, acc_std))
    print ('auc: {0:.3f} +/- {1:.3f}'.format(auc_mean, auc_std))





#Writing run time into Log
print("Run started in  ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Load Data
data            = pd.read_csv("../Data/data_features.csv")
data_label      = pd.read_csv("../Data/data_labels.csv", dtype={'label': str})

data.reset_index(drop=True)
data_label.reset_index(drop=True)

#Global params
N_KFOLDS = 5
mapping = {'4321':1,  '4141':2, '442':3, '523':4,  '415':5, '352':5,  '4411':5,
           '532':5 ,  '4123':5, '361':5, '3412':5, '343':5, '4213':5, '4312':5,
           '3421':5, '4213a':5, '451':5, '5212':5, '433':5, '4231t':5}

#Mapping formations to number
mappped_labels = data_label.applymap(lambda s: mapping.get(s) if s in mapping else s)['label'].to_numpy()

# LightGBM Classifier  
clf = LGBMClassifier(max_depth=50, min_child_samples=40, n_estimators=200,num_leaves=80,learning_rate=0.31245 )

#Without all data
start = time()
new_data  = data.to_numpy()
train_evaluate('All features', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"All features\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))
print ('-'*80)

#Without y positions
start = time()
new_data  = data[[col for col in data.columns if col not in  ['y2','y3','y4','y5','y6','y7','y8','y9','y10','y11']]].to_numpy()
train_evaluate('Without y', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"Without y\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))
print ('-'*80)


#Without x positions
start = time()
new_data  = data[[col for col in data.columns if col not in ['x2','x3','x4','x5','x6','x7','x8','x9','x10','x11']]].to_numpy()
train_evaluate('Without x', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"Without x\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))
print ('-'*80)

#Without avg x positions
start = time()
new_data  = data.loc[:,data.columns != 'avgx'].to_numpy()
train_evaluate('Without avgx', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"Without avgx\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))
print ('-'*80)

#Without avg y positions
start = time()
new_data  = data.loc[:,data.columns != 'avgy'].to_numpy()
train_evaluate('Without avgy', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"Without avgy\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))
print ('-'*80)

#Without avg x positions and y positions
start = time()
new_data  = data[[col for col in data.columns if col not in ['avgx','avgy']]].to_numpy()
train_evaluate('Without avgx and avgy', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"Without avgx and avgy\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))
print ('-'*80)

#Without ball x positions
start = time()
new_data  = data.loc[:,data.columns != 'ballx'].to_numpy()
train_evaluate('Without ballx', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"Without ballx\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))
print ('-'*80)

#Without ball y positions
start = time()
new_data  = data.loc[:,data.columns != 'bally'].to_numpy()
train_evaluate('Without bally', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"Without bally\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))
print ('-'*80)

#Without ball positions
start = time()
new_data  = data[[col for col in data.columns if col not in ['ballx','bally']]].to_numpy()
train_evaluate('Without ballx and bally', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"Without ballx and bally\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))
print ('-'*80)

#Without layers measure
start = time()
new_data  = data.loc[:,data.columns != 'layers'].to_numpy()
train_evaluate('Without layers', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"Without layers\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))
print ('-'*80)

#Without x11-x2 measure
start = time()
new_data  = data.loc[:,data.columns != 'x11_x2'].to_numpy()
train_evaluate('Without x11_x2', clf, new_data, mappped_labels, N_KFOLDS)
print("Learning took %.2f seconds for \"Without x11_x2\" case with %d Stratified Cross Validation" % (time() - start, N_KFOLDS))


