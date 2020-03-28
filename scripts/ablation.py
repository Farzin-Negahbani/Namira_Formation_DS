# -*- coding: utf-8 -*-
"""
Created on Fri March 11 2020

@author: Farzin Negahbani, Ehsan Asali
"""

import numpy as np
import pandas as pd 
from lightgbm import LGBMClassifier
from datetime import datetime
from utils import train_evaluate, feature_ablation, data_loader,feature_ablation_seq


#Writing run time into Log
print("Run started in  ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Load Data
r_data, r_data_label, s_data, s_data_label = data_loader("data/rand_data_cons_10_pks_15.csv","data/seq_data_cons_10_pks_15.csv")


###### Random Data 
# Global params
mapping = {'4321':0,  '4141':1, '442':2, '523':3,  '415':4, '352':5,  '4411':6,
           '532':7 ,  '4123':8, '361':9, '3412':10, '343':11, '4213':12, '4312':13,
           '3421':14, '4213a':15, '451':16, '5212':17, '433':18, '4231t':19}

#mapping = {'4321\n':0,  '4141\n':1, '442\n':2, '523\n':3,  '415\n':4, '352\n':5,  '4411\n':6,
#           '532\n':7 ,  '4123\n':8, '361\n':9, '3412\n':10, '343\n':11, '4213\n':12, '4312\n':13,
#           '3421\n':14, '4213a\n':15, '451\n':16, '5212\n':17, '433\n':18, '4231t\n':19}

# Mapping formations to number
#r_mapped_labels     = r_data_label.applymap(lambda s: mapping.get(s) if s in mapping else print("MAPPING ERROR"))['label'].to_numpy()
#s_mapped_data_label = s_data_label.applymap(lambda s: mapping.get(s) if s in mapping else print("MAPPING ERROR"))['label'].to_numpy()

# LightGBM Classifier 1
clf = LGBMClassifier(learning_rate= 0.3478916588531426, max_depth=55,min_child_samples=45, n_estimators=600, num_leaves=80)

# LightGBM Classifier 2  
#clf = LGBMClassifier(learning_rate= 0.24, max_depth=55, min_child_samples=45, n_estimators=200, num_leaves=80)


# Performing the ablation for all formations proposed 
#feature_ablation(r_data, r_mapped_labels, clf,  mapping.keys())

# For 3 layers formationss
mapping_3L      = {'442':0,  '523':1, '415':2, '352':3, 
                '532':4, '361':5, '343':6,'451':7, '433':8}

# Preparing 3 layer formations data
#label_3_layer   = r_data_label.loc[r_data_label['label'].isin(mapping_3L.keys()) ]
#data_3_layer    = r_data.loc[r_data_label['label'].isin(mapping_3L.keys()) ]
#mapped_3_labels = label_3_layer.applymap(lambda s: mapping_3L.get(s) if s in mapping_3L else s)['label'].to_numpy()

# Performing the ablation for 3 layer formations
#feature_ablation(data_3_layer, mapped_3_labels, clf, mapping_3L.keys())


###### Seq Data 
feature_ablation_seq(r_data, r_data_label, clf, mapping.keys(), s_data, s_data_label,10)