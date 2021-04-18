from pandas.io.parsers import read_csv
import numpy as np
#import pandas as pd
np.random.seed(0)
df1=read_csv("a.csv")
df2=read_csv("b.csv")
'''笨办法555
a1=df1["Sepal.Length"]
a2=df1["Sepal.Width"]
a3=df1["Petal.Length"]
a4=df1[""]'''
a=df1[:40].append(df2[:40])
a=a.values
A=np.mat(a)
A_1=A.T
A=A_1[1:]
B=A.T
beta=np.mat("0.0;0.0;0.0;0.0")
y1=[i-i+1 for i in range(100) if i <40]
y2=[i-i for i in range(100) if i <40]
y=y1+y2 #分类值
error=1
def logistic(x): #逻辑斯蒂函数
    s=np.exp(x)
    return s/(s+1)
def lo(x): #指数函数
    s=np.exp(x)
    return s
def fanshu(x): #求向量的二范数
    x=x.tolist()[0]
    a=0
    for i in range(len(x)):
        a=a+x[i]**2
    return np.sqrt(a)
while np.abs(error)>0.0005:
    W=np.arange(79.9)
    W1=W.copy()
    for i in range(80):
        W[i]=(lo(B[i]*beta)/(lo(B[i]*beta)+1)-y[i])
        W1[i]=lo(B[i]*beta)/(lo(B[i]*beta)+1)**2
    #W=np.mat(np.diag(W.tolist()))
    W=np.mat(W)
    #W1=np.mat(np.diag(W1.tolist()))
    W1=np.mat(np.diag(W1))
    df=A*W.T/80
    ddf=A*W1*B/80
    err=np.dot(np.linalg.inv(ddf),df)
    beta=beta-err
    error=fanshu(err)    
#测试
test=df1[40:50].append(df2[40:50])
test=np.mat(test.values)
test_1=test.T
test=test_1[1:].T
aa=test[:10]*beta
bb=test[10:20]*beta
aaresult=logistic(aa)
bbresult=logistic(bb) 