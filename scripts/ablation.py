# -*- coding: utf-8 -*-
"""
Created on Fri March 11 2020

@author: Farzin Negahbani, Ehsan Asali
"""

import numpy as np
import pandas as pd 
from lightgbm import LGBMClassifier
from datetime import datetime
from utils import train_evaluate, feature_ablation


#Writing run time into Log
print("Run started in  ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Load Data
data            = pd.read_csv("data/data_features.csv",index_col=0)
data_label      = pd.read_csv("data/data_labels.csv", dtype={'label': str},index_col=0)

# Reset the index
data.reset_index(drop=True)
data_label.reset_index(drop=True)

# Global params
mapping = {'4321':0,  '4141':1, '442':2, '523':3,  '415':4, '352':5,  '4411':6,
           '532':7 ,  '4123':8, '361':9, '3412':10, '343':11, '4213':12, '4312':13,
           '3421':14, '4213a':15, '451':16, '5212':17, '433':18, '4231t':19}

# Mapping formations to number
mapped_labels = data_label.applymap(lambda s: mapping.get(s) if s in mapping else s)['label'].to_numpy()

# LightGBM Classifier  
#clf = LGBMClassifier(learning_rate= 0.3478916588531426, max_depth=55,
#    min_child_samples=45, n_estimators=600, num_leaves=80)

# LightGBM Classifier  
clf = LGBMClassifier(learning_rate= 0.24, max_depth=55, min_child_samples=45, n_estimators=200, num_leaves=80)


# Performing the ablation for all formations proposed 
#feature_ablation(data, mapped_labels, clf,  mapping.keys())

# For 3 layers formationss
mapping      = {'442':0,  '523':1, '415':2, '352':3, 
                '532':4, '361':5, '343':6,'451':7, '433':8}

# Preparing 3 layer formations data
label_3_layer   = data_label.loc[data_label['label'].isin(mapping.keys()) ]
data_3_layer    = data.loc[data_label['label'].isin(mapping.keys()) ]
mapped_3_labels = label_3_layer.applymap(lambda s: mapping.get(s) if s in mapping else s)['label'].to_numpy()

# Performing the ablation for 3 layer formations
feature_ablation(data_3_layer, mapped_3_labels, clf, mapping.keys())
