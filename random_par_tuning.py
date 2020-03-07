# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 2020

@author: Farzin Negahbani, Ehsan Asali
"""

import numpy as np
import pandas as pd 
from time import time
import scipy.stats as stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from datetime import datetime


# This function copied from Sklearn, link provided below. 
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#Writing run time into Log
print("##############################################################")
print("New run started in  ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Load Data
train_data            = pd.read_csv("Data/data_features.csv")
train_data_label      = pd.read_csv("Data/data_labels.csv")

# Random Forest Classifier  
rf_param = {
            'max_features':['auto','log2'],
            'n_estimators': [80,90,100,110,120,140,150,200],
            'max_depth':[30,35,40,45,50,55,60],
            'random_state':[55,60,65,70,75,80,85]}
  
rf = RandomForestClassifier()
start = time()
random_search = RandomizedSearchCV(rf, param_distributions=rf_param,n_iter=20)
random_search.fit(train_data,train_data_label.label )
print("RandomizedSearchCV took %.2f seconds for Random Forest." % (time() - start))
report(random_search.cv_results_)

# XGboost Classifier 
xg_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'learning_rate': stats.uniform(0, 0.2),
        'max_depth': [3, 4, 5,6,7,8,9,10],
        'eval_metric':['auc','mlogloss'],
        'n_estimators':[100,200,250,300]
        }

xgb = XGBClassifier(silent=True, nthread=8)

#skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)  cv=skf.split(X,Y), 

start = time()
random_search = RandomizedSearchCV(xgb, param_distributions=xg_params, n_iter=20, scoring='roc_auc', n_jobs=4, random_state=1001 )
random_search.fit(train_data,train_data_label.label )
print("RandomizedSearchCV took %.2f seconds for XGboost." % (time() - start))
report(random_search.cv_results_)



# for fixing LightGBMError: Do not support special JSON characters in feature name.
train_data.columns       = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in train_data.columns]
train_data_label.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in train_data_label.columns]

# LightGBM Classifier  
lgbm_param = {
            'max_depth':[30,35,40,45,50,55,60],
            'min_child_samples':[10,15,20,30,40,45,50],
            'n_estimators':[60,70,80,90,100,110,120],
            'learning_rate':stats.uniform(0.2, 0.5),
            'num_leaves':[20,30,40,45,50,60,70,80],
            'tree_learner':['feature']}


lgbm = LGBMClassifier()
start = time()
random_search = RandomizedSearchCV(lgbm, param_distributions=lgbm_param,n_iter=20)
random_search.fit(train_data,train_data_label.label )
print("RandomizedSearchCV took %.2f seconds for LGBM." % (time() - start))
report(random_search.cv_results_)
