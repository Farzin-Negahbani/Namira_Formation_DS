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


def train_evaluate(name, clf, train_data, labels):
    '''
    This function trains, evaluates and then reports the classification metrics
    '''
    mapping_list = ['4321',  '4141', '442', '523',  '415', '352',  '4411',
           '532' ,  '4123', '361', '3412', '343', '4213', '4312',
           '3421', '4213a', '451', '5212', '433', '4231t']

    start = time()
    xt, xv, yt, yv = train_test_split(train_data, labels, test_size=0.33, random_state=42)
    clf.fit(xt, yt)
    yhat  = clf.predict(xv)
    proba = clf.predict_proba(xv)
    acc=np.mean(yhat == yv)

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

