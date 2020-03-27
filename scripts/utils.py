# -*- coding: utf-8 -*-
"""
Created on Fri March 15 2020

@author: Farzin Negahbani, Ehsan Asali
"""

import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy import interp
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from time import time


def gen_seq_data(inputfile="../data/sorted_raw_data.csv",cons_cycles=10, pks=15):
    '''
        Generates non-overlapping consecutive and random datasets form 
        sorted raw dataset 
    '''
    i = 0
    dic      = {}
    pks_dic  = {} 
    cons_set = ["UPCI,PCI,game,cycle,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,avgx,avgy,x11-x2,ballx,bally,layers,label\n"]
    rand_set = ["UPCI,PCI,game,cycle,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,avgx,avgy,x11-x2,ballx,bally,layers,label\n"]
    temp_set = []

    fo  = open(inputfile, "r")

    for line in fo:
        if i>1:
            # same UPCI not registered in dic still need more data
            if not line.split(',')[0] in dic.keys() and len(temp_set) < cons_cycles and last_line.split(',')[0] == line.split(',')[0]:
                temp_set.append(line)

            # same UPCI not registered in dic found enough data
            elif not line.split(',')[0] in dic.keys() and len(temp_set) == cons_cycles and last_line.split(',')[0] == line.split(',')[0]:
                if line.split(',')[-1] in pks_dic.keys() and pks_dic[line.split(',')[-1]] < pks:
                    temp_set.append(line)
                    cons_set.extend(temp_set)
                    pks_dic[line.split(',')[-1]] +=1
                    dic[line.split(',')[0]]       = cons_cycles
                
                elif not line.split(',')[-1] in pks_dic.keys() :
                    pks_dic[line.split(',')[-1]] = 1
                    temp_set.append(line)
                    cons_set.extend(temp_set)
                    dic[line.split(',')[0]]       = cons_cycles
                
                else:
                    temp_set.append(line)
                    rand_set.extend(temp_set)
                
                temp_set=[]
            
            # UPCI changed but couldn't find sufficient amount of data
            elif not line.split(',')[0] in dic.keys() and last_line.split(',')[0] != line.split(',')[0]:
                temp_set.append(line)
                rand_set.extend(temp_set)
                temp_set=[]

            # same UPCI registered in dic
            elif line.split(',')[0] in dic.keys():
                temp_set.append(line)
                rand_set.extend(temp_set)
                temp_set=[]
    
        last_line = line
        i+=1

    fo.close()
    
    fo = open("cons_data_cons_"+str(cons_cycles)+"_pks_"+str(pks)+".csv", "w")
    fo.writelines(cons_set)
    fo.close()

    fo = open("rand_data_cons_"+str(cons_cycles)+"_pks_"+str(pks)+".csv", "w")
    fo.writelines(rand_set)
    fo.close()



def feature_ablation(data, mapped_labels, clf, mapping_list):
    '''
        Applies the ablation on the input data and corresponding labels
        
        For all formations: 
        mapping_list = ['4321',  '4141', '442', '523',  '415', '352',  '4411',
           '532' ,  '4123', '361', '3412', '343', '4213', '4312',
           '3421', '4213a', '451', '5212', '433', '4231t']
    
        For 3 Layer formations:
        mapping_list = ['442', '523', '415', '352', '532', '361', '343','451', '433']
    
    '''
    #Without all data
    new_data  = data.to_numpy()
    train_evaluate('All features', clf, new_data, mapped_labels, mapping_list)

    #Without y positions
    new_data  = data[[col for col in data.columns if col not in  ['y2','y3','y4','y5','y6','y7','y8','y9','y10','y11']]].to_numpy()
    train_evaluate('Without y', clf, new_data, mapped_labels, mapping_list)


    #Without x positions
    new_data  = data[[col for col in data.columns if col not in ['x2','x3','x4','x5','x6','x7','x8','x9','x10','x11']]].to_numpy()
    train_evaluate('Without x', clf, new_data, mapped_labels, mapping_list)

    #Without avg x positions
    new_data  = data.loc[:,data.columns != 'avgx'].to_numpy()
    train_evaluate('Without avgx', clf, new_data, mapped_labels, mapping_list)

    #Without avg y positions
    new_data  = data.loc[:,data.columns != 'avgy'].to_numpy()
    train_evaluate('Without avgy', clf, new_data, mapped_labels, mapping_list)

    #Without avg x positions and y positions
    new_data  = data[[col for col in data.columns if col not in ['avgx','avgy']]].to_numpy()
    train_evaluate('Without avgx and avgy', clf, new_data, mapped_labels, mapping_list)

    #Without ball x positions
    new_data  = data.loc[:,data.columns != 'ballx'].to_numpy()
    train_evaluate('Without ballx', clf, new_data, mapped_labels, mapping_list)

    #Without ball y positions
    new_data  = data.loc[:,data.columns != 'bally'].to_numpy()
    train_evaluate('Without bally', clf, new_data, mapped_labels, mapping_list)

    #Without ball positions 
    new_data  = data[[col for col in data.columns if col not in ['ballx','bally']]].to_numpy()
    train_evaluate('Without ballx and bally', clf, new_data, mapped_labels, mapping_list)

    #Without layers measure 
    new_data  = data.loc[:,data.columns != 'layers'].to_numpy()
    train_evaluate('Without layers', clf, new_data, mapped_labels, mapping_list)

    #Without x11-x2 measure
    new_data  = data.loc[:,data.columns != 'x11_x2'].to_numpy()
    train_evaluate('Without x11_x2', clf, new_data, mapped_labels, mapping_list)


    #Just with x and y
    new_data  = data[['y2','y3','y4','y5','y6','y7','y8','y9','y10','y11','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11']].to_numpy()
    train_evaluate('Just with x and y', clf, new_data, mapped_labels, mapping_list)

    #Just with avgx and avgy
    new_data  = data[['avgy','avgx']].to_numpy()
    train_evaluate('Just with avgx and avgy', clf, new_data, mapped_labels, mapping_list)


    #Just with x11-x2 measure
    new_data  = data[['x11_x2']].to_numpy()
    train_evaluate('Just with x11-x2 measure', clf, new_data, mapped_labels, mapping_list)

    #Just with ball position
    new_data  = data[['ballx','bally']].to_numpy()
    train_evaluate('Just with ball position', clf, new_data, mapped_labels, mapping_list)

    #Just with ball position and x,y
    new_data  = data[['ballx','bally','y2','y3','y4','y5','y6','y7','y8','y9','y10','y11','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11']].to_numpy()
    train_evaluate('Just with ball position and x,y', clf, new_data, mapped_labels, mapping_list)

    #Just with x11-x2 and ball position
    new_data  = data[['x11_x2','ballx','bally',]].to_numpy()
    train_evaluate('Just with x11-x2 and ball position', clf, new_data, mapped_labels, mapping_list)


    #Just with avgx, avgy, and ball position
    new_data  = data[['avgy','avgx','ballx','bally',]].to_numpy()
    train_evaluate('Just with avgx, avgy, and ball position', clf, new_data, mapped_labels, mapping_list)


def train_evaluate(name, clf, train_data, labels, mapping_list):
    '''
    This function trains, evaluates and then reports the classification metrics
    '''

    start = time()
    xt, xv, yt, yv = train_test_split(train_data, labels, test_size=0.33, random_state=42)
    clf.fit(xt, yt)
    yhat  = clf.predict(xv)

    print ("In Case : ",name)
    print("Traing and evaluation took %.2f seconds." % (time() - start))
    
    sk_report    = classification_report(digits=4,y_true=yv, y_pred=yhat,target_names=mapping_list)
    print("Classification Report: ")
    print(sk_report)

    #my_sk_report = class_report(y_true=yv, y_pred=yhat)
    #print(my_sk_report)

    print ('-'*80)


def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(y_pred,return_counts=True)
    n_classes   = len(labels)
    pred_cnt    = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df

