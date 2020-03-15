# -*- coding: utf-8 -*-
"""
Created on Fri March 11 2020

@author: Farzin Negahbani, Ehsan Asali
"""

import numpy as np
import pandas as pd 
from lightgbm import LGBMClassifier
from datetime import datetime
from utils import train_evaluate


#Writing run time into Log
print("Run started in  ", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# Load Data
data            = pd.read_csv("data/data_features.csv",index_col=0)
data_label      = pd.read_csv("data/data_labels.csv", dtype={'label': str},index_col=0)

# Reset the index
data.reset_index(drop=True)
data_label.reset_index(drop=True)

#Global params
mapping = {'4321':0,  '4141':1, '442':2, '523':3,  '415':4, '352':5,  '4411':6,
           '532':7 ,  '4123':8, '361':9, '3412':10, '343':11, '4213':12, '4312':13,
           '3421':14, '4213a':15, '451':16, '5212':17, '433':18, '4231t':19}

#Mapping formations to number
mapped_labels = data_label.applymap(lambda s: mapping.get(s) if s in mapping else s)['label'].to_numpy()

# LightGBM Classifier  
clf = LGBMClassifier(learning_rate= 0.2521512265909897, max_depth=55, min_child_samples=45, n_estimators=100, num_leaves=80, tree_learner= 'feature')

#Without all data
new_data  = data.to_numpy()
train_evaluate('All features', clf, new_data, mapped_labels)

#Without y positions
new_data  = data[[col for col in data.columns if col not in  ['y2','y3','y4','y5','y6','y7','y8','y9','y10','y11']]].to_numpy()
train_evaluate('Without y', clf, new_data, mapped_labels)


#Without x positions
new_data  = data[[col for col in data.columns if col not in ['x2','x3','x4','x5','x6','x7','x8','x9','x10','x11']]].to_numpy()
train_evaluate('Without x', clf, new_data, mapped_labels)

#Without avg x positions
new_data  = data.loc[:,data.columns != 'avgx'].to_numpy()
train_evaluate('Without avgx', clf, new_data, mapped_labels)

#Without avg y positions
new_data  = data.loc[:,data.columns != 'avgy'].to_numpy()
train_evaluate('Without avgy', clf, new_data, mapped_labels)

#Without avg x positions and y positions
new_data  = data[[col for col in data.columns if col not in ['avgx','avgy']]].to_numpy()
train_evaluate('Without avgx and avgy', clf, new_data, mapped_labels)

#Without ball x positions
new_data  = data.loc[:,data.columns != 'ballx'].to_numpy()
train_evaluate('Without ballx', clf, new_data, mapped_labels)

#Without ball y positions
new_data  = data.loc[:,data.columns != 'bally'].to_numpy()
train_evaluate('Without bally', clf, new_data, mapped_labels)

#Without ball positions 
new_data  = data[[col for col in data.columns if col not in ['ballx','bally']]].to_numpy()
train_evaluate('Without ballx and bally', clf, new_data, mapped_labels)

#Without layers measure 
new_data  = data.loc[:,data.columns != 'layers'].to_numpy()
train_evaluate('Without layers', clf, new_data, mapped_labels)

#Without x11-x2 measure
new_data  = data.loc[:,data.columns != 'x11_x2'].to_numpy()
train_evaluate('Without x11_x2', clf, new_data, mapped_labels)


#Just with x and y
new_data  = data[['y2','y3','y4','y5','y6','y7','y8','y9','y10','y11','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11']].to_numpy()
train_evaluate('Just with x and y', clf, new_data, mapped_labels)

#Just with avgx and avgy
new_data  = data[['avgy','avgx']].to_numpy()
train_evaluate('Just with avgx and avgy', clf, new_data, mapped_labels)


#Just with x11-x2 measure
new_data  = data[['x11_x2']].to_numpy()
train_evaluate('Just with x11-x2 measure', clf, new_data, mapped_labels)

#Just with ball position
new_data  = data[['ballx','bally']].to_numpy()
train_evaluate('Just with ball position', clf, new_data, mapped_labels)

#Just with ball position and x,y
new_data  = data[['ballx','bally','y2','y3','y4','y5','y6','y7','y8','y9','y10','y11','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11']].to_numpy()
train_evaluate('Just with ball position and x,y', clf, new_data, mapped_labels)

#Just with x11-x2 and ball position
new_data  = data[['x11_x2','ballx','bally',]].to_numpy()
train_evaluate('Just with x11-x2 and ball position', clf, new_data, mapped_labels)


#Just with avgx, avgy, and ball position
new_data  = data[['avgy','avgx','ballx','bally',]].to_numpy()
train_evaluate('Just with avgx, avgy, and ball position', clf, new_data, mapped_labels)


