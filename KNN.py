from pandas.io.parsers import read_csv
import numpy as np
#from sklearn import preprocessing
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
#放入的KNN
def zscore(X):
    for i in range(len(X.T[0].tolist()[0])):
        u=np.mean(X[i])
        sigma=np.sqrt(np.var((X[i])))
        X[i]=(X[i]-u)/sigma
    return X    
def distance(x,y):
    x=x.tolist()[0]
    y=y.tolist()[0]
    k=len(x)
    l=0
    for i in range(k):
        l=(x[i]-y[i])**2+l
    return np.sqrt(l)
def distancemat(X):#距离矩阵
    a=len(X[0].tolist()[0])
    B1=np.eye(a)
    A1=X.T
    for i in range(a):
        for j in range(a):
            B1[i][j]=distance(A1[i], A1[j])
    return B1
T=A.copy()
X=zscore(T)#标准化 会改变T所以取copy
print(X.mean(axis=1),X.var(axis=1)) #等于一按行取均值or标准差 等于0按列
#这里等价于
#T=A.copy()
#Y=preprocessing.scale(T.T)
#Y=Y.T
#assert X==Y  这个是对的 但是不能这么用
D=distancemat(X)#距离矩阵
y1=[1 for i in range(40)]
y2=[0 for i in range(40)]
y=y1+y2#指标值
def KNNmodel(K,D,Y,number):
    dd=D.tolist()[number]#识别目标距离矩阵的列表
    dd.sort()#排序
    d=D.tolist()#距离矩阵化成列表
    s=[]#空列表 存数，防止距离重复值
    for i in range(1,K+1):
        for j in range(len(dd)):
            if d[number][j]==dd[i]:
                t=[True for x in s if x!=j]
                if len(t)==len(s):
                    s.append(j)
                    print(j,Y[j])
                    break
KNNmodel(10,D,y,6)    
KNNmodel(6,D,y,53)

    
        
        
        
        
    