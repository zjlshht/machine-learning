# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:19:34 2020

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
from sklearn.cross_decomposition import PLSRegression
import seaborn as sns
clf = PLSRegression(n_components=2)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)
clf.fit(X_train,y_train)
print("Score=%.2f%%"  %(100*clf.score(X_test,y_test)))
import  matplotlib.pyplot  as plt
y_pred=clf.predict(X_test)
sns.set()
f,ax=plt.subplots()
plt.scatter(y_test,y_pred)
plt.xlabel("y_true")
plt.ylabel("y_predict")
plt.title("prediction and truth of y")
plt.plot([0,0],[100,100],'yellow')
plt.show()