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
a=df1.values#从dataframe中提取值
A=np.mat(a)#转化为矩阵
A_1=A.T
A=A_1[1:].T #因为矩阵没法取列
b=df2.values
B=np.mat(b)
B_1=B.T
B=B_1[1:].T
beta=np.mat("0.0;0.0;0.0;0.0")
def maxtrnum(x): #将矩阵数转化成数值
    x=x.tolist()
    return x[0][0]
def fanshu(x): #求向量的二范数
    x=x.tolist()[0]
    a=0
    for i in range(len(x)):
        a=a+x[i]**2
    return np.sqrt(a)
def logistic(x): #逻辑斯蒂函数
    s=np.exp(x)
    return s/(s+1)
error=1
while np.abs(error>0.0005):
    W1=np.arange(39.9) #创建浮点数列
    W2=np.arange(39.9)
    #W2=W1.copy()
    tdf=0
    tddf=0
    for i in range(40):
        W1[i] = np.exp(np.dot(A[i],beta))
        W2[i] = np.exp(np.dot(B[i],beta))
        tdf=(W1[i]/(1+W1[i])-1)*A[i].T+(W2[i]/(1+W2[i]))*B[i].T+tdf
        tddf=W1[i]/(1+W1[i])**2*np.dot(A[i].T,A[i])+W2[i]/(1+W2[i])**2*np.dot(B[i].T,B[i])+tddf
    df=tdf/80
    ddf=tddf/80
    err=np.dot(np.linalg.inv(ddf),df)
    beta=beta-err
    error=fanshu(err)
#测试
aa=A[40:50]*beta
bb=B[40:50]*beta
aaresult=logistic(aa)
bbresult=logistic(bb)     
#datatime:2020.11.5 15:24       
    
    
    