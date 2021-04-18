from pandas.io.parsers import read_csv
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
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
'''划分流行与不流行'''
for i in range(len(y)):
    if y[i]>=0.6: #0.6为分界线
        y[i]=1
    else:
        y[i]=0

ros = RandomOverSampler(random_state=0)
scaler = StandardScaler()  
Score=[]
kf = KFold(n_splits=10,random_state=seed, shuffle=True)
for train_index, test_index in kf.split(X):
    '''先随机分为10份'''
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    '''随机过采样'''
    X_train, y_train = ros.fit_sample(X_train, y_train)
    scaler.fit(X_train) 
    '''标准化'''
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  # doctest: +SKIP
    neigh = KNeighborsRegressor(n_neighbors=6)#k=10
    neigh.fit(X_train,y_train)
    y_pred=neigh.predict(X_test)
    y_pred= [round(value) for value in y_pred]
    '''计算准确率'''
    a = accuracy_score(y_test, y_pred)
    Score.append(a)
    '''画混淆矩阵图'''
    sns.set()
    f,ax=plt.subplots()
    C2= confusion_matrix(y_test, y_pred, labels=[0 , 1])
    print(C2) #打印出来看看
    sns.heatmap(C2,annot=True,ax=ax) #画热力图
    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    plt.show()
'''画准确率图'''
plt.plot(range(10),Score)
plt.title("accuracy of diffierent K-value")
plt.xlabel("K value")
plt.xticks(range(10))
plt.ylabel("accuracy")