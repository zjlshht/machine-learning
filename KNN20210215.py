# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:31:57 2021

@author: zjlsyhhht
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
def squares(x,y):
    s=0
    for i in range(len(x)-1):
       s=s+(x[i]-x[i+1])*(y[i]+y[i+1])/2
    return s
X=np.loadtxt('Xtrain.csv')

y=np.loadtxt('Ytrain.csv')
skf = StratifiedKFold(n_splits=5)
S=[]
AUC_ROC=[]
AUC_PR=[]
for train_index, validation_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", validation_index)
    X_train, X_test = X[train_index], X[validation_index]
    y_train, y_test = y[train_index], y[validation_index]
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train,y_train)
    y_pred=neigh.predict(X_test)
    a = accuracy_score(y_test, y_pred)
    S.append(a)
    AUC_roc=roc_auc_score(y_test, neigh.predict_proba(X_test)[:, 1])
    AUC_ROC.append(AUC_roc)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    AUC_pr=squares(recall,precision)
    AUC_PR.append(AUC_pr)
Smean=np.mean(S)
SROC=np.mean(AUC_ROC)
SPR=np.mean(AUC_PR)
Sstd=np.std(S)


'''
neigh = KNeighborsRegressor(n_neighbors=1)#k=10
neigh.fit(X_train,Y_train)
y_pred=neigh.predict(X_test)
#y_pred= [round(value) for value in y_pred]
#y_pred=neigh.predict(X_test)
   






skf = StratifiedKFold(n_splits=5)
'''


