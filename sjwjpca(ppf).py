# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:00:59 2020

@author: zjlsyhhht
"""

from sklearn.preprocessing import StandardScaler
#PCR matlab结果是62.5%
from sklearn.preprocessing import  MinMaxScaler
from sklearn.linear_model import Ridge 
mm = MinMaxScaler()
from sklearn.decomposition import PCA
from pandas.io.parsers import read_csv
import numpy as np
data=read_csv("data.csv")#读取数据
df=data.values#转换数据类型
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
X=np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a12,a13,a14]).T#生成参变量
b=np.arange(100)
b[:]=0
'''统计每年音乐数'''
for j in range(1921,2021):
    for i in range(len(a15)):
        if a15[i]==j:
            b[(j-1928)]=b[(j-1928)]+1            
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
'''对流行度变量关于年份归一化'''
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
seed=666#随机种子
test_size=0.25#测试集比例
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)
'''自变量X标准化'''        
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
pca = PCA(n_components=7)
pca.fit(X_train,y_train)
y_pred=pca.predict(X_test)
    #ridge-ols
#RidgePcrModel = Ridge(alpha=0.001)
#RidgePcrModel.fit(mm.fit_transform(X_train_tran), mm.fit_transform(y_train))
#pcr_ans = RidgePcrModel.predict(mm.fit_transform(X_test_tran))

#print('pcr:',mse(pcrcv(X_train, y_train, X_test, y_test),mm.fit_transform(y_test)))
from sklearn.metrics import r2_score
print(r2_score(y_pred,y_test))