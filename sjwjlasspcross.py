# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:21:17 2020

@author: zjlsyhhht
"""

from pandas.io.parsers import read_csv
import numpy as np
data=read_csv("data.csv")
df=data.values
a1=df[:,0]
a2=df[:,2]
a3=df[:,3]
a4=df[:,4]
a5=df[:,5]
a6=df[:,7]
a7=df[:,8]
a8=df[:,9]
a9=df[:,10]
a10=df[:,11]
a11=df[:,13]#pupolarity
a12=df[:,15]
a13=df[:,16]
a14=df[:,17]
a15=df[:,18]#year
X=np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a12,a13,a14,a15]).T
y=a11
seed=666
import  matplotlib.pyplot  as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
kf = KFold(n_splits=4,random_state=seed, shuffle=True)
Score=[]
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = linear_model.Lasso(alpha=0.2)
    clf.fit(X_train,y_train)
    print("Score=%.2f%%"  %(100*clf.score(X_test,y_test)))
    Score.append(clf.score(X_test,y_test))
    y_pred=clf.predict(X_test)
    plt.scatter(y_test,y_pred)
    plt.plot([0,0],[100,100],'yellow')
    plt.xlabel("y_test")
    plt.ylabel("y_pred")
    plt.title("prediction and true value")
    plt.show()
plt.bar(range(len(Score)),Score)
plt.title("4-Cross-validation")
plt.xlabel("time")
plt.xticks(range(len(Score)))
plt.ylabel("accuracy")