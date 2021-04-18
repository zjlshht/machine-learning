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
X=np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a12,a13,a14]).T
b=np.arange(100)
b[:]=0
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
for i in range(100):
    [t,k]=classs(a15,a11,(i+1921),b)
    u=np.mean(t)
    sigma=np.sqrt(np.var(t))
    t=(t-u)/sigma
    s=0
    for j in k:
        popularity[j]=t[s]
        s=s+1
        
y=popularity
seed=666
test_size=0.25

for i in range(len(y)):
    if y[i]>=0.2816: #0.9分位数
        y[i]=1
    else:
        y[i]=0
        
from sklearn.model_selection import KFold

  

import numpy as np
kf = KFold(n_splits=4,random_state=seed, shuffle=True)
Score=[]
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    scaler = StandardScaler()  # doctest: +SKIP
# Don't cheat - fit only on training data
    scaler.fit(X_train)  # doctest: +SKIP
    X_train = scaler.transform(X_train)  # doctest: +SKIP
# apply same transformation to test data
    X_test = scaler.transform(X_test)  # doctest: +SKIP

    model = Sequential()


    model.add(Dense(units=20, activation='relu', input_dim=13))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    y_pr=model.predict(X_test)
    y_pr[y_pr>0.5]=1
    y_pr[y_pr<=0.5]=0
    sns.set()
    f,ax=plt.subplots()
    C2= confusion_matrix(y_test, y_pr, labels=[0, 1])
    print(C2) #打印出来看看
    sns.heatmap(C2,annot=True,ax=ax) #画热力图
    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    plt.show()
    score = model.evaluate(X_test, y_test, batch_size=32)
    Score.append(score[1])


plt.bar(range(len(Score)),Score)
plt.title("4-Cross-validation")
plt.xlabel("time")
plt.xticks(range(len(Score)))
plt.ylabel("accuracy")



