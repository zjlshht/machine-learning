# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:34:27 2020

@author: zjlsyhhht
"""

from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt
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
X=np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a12,a13,a14]).T
b=np.arange(100)
b[:]=0
for j in range(1921,2021):
    for i in range(len(a15)):
        if a15[i]==j:
            b[(j-1928)]=b[(j-1928)]+1
plt.plot(b)
plt.xlabel("s")
plt.show()
def classs(X,Y,a,b): #X年 Y流行度 a年份   b对应年的数
    cc=np.arange(b[a-1928])
    k=np.arange(b[a-1928])
    j=0
    for i in range(len(X)): 
        if X[i]==(a):
            t=Y[i]
            cc[j]=t
            k[j]=i
            j=j+1
        if j==b[a-1928]:
            break
    return [cc,k]
popularity=np.arange(169908.9)
for i in range(100):
    [t,k]=classs(a15,a11,(i+1921),b)
    t_min=np.min(t)
    t_max=np.max(t)
    t=(t-t_min)/(t_max-t_min)
    s=0
    for j in k:
        popularity[j]=t[s]
        s=s+1
        
y=popularity