# First XGBoost model for Pima Indians dataset
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn import metrics
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
    if y[i]>=1.2816:
        y[i]=1
    elif y[i]>=0.2355:
        y[i]=0
    else:
        y[i]=-1
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size, random_state=seed)
'''for i in range(len(y_test)):
    if y_test[i]>=1.2816: #0.9分位数
        y_test[i]=1
    elif y_test[i]>0.2355:#0.6分位数
        y_test[i]=0
    else:
        y_test[i]=-1
for j in range(len(y_train)):
    if y_train[j]>=1.2816: #0.9分位数
        y_train[j]=1
    elif y_train[j]>0.2355:
        y_train[j]=0
    else:
        y_train[j]=-1'''
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sns.set()
f,ax=plt.subplots()
C2= confusion_matrix(y_test, predictions, labels=[-1 ,0 , 1])
print(C2) #打印出来看看
sns.heatmap(C2,annot=True,ax=ax) #画热力图
ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴
plt.show()
'''
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,                                          estimator_name='example estimator')
display.plot()  # doctest: +SKIP
plt.show()  '''
'''
metrics.plot_roc_curve(model, X_test, y_test)  # doctest: +SKIP
plt.title("GRAPH OF ROC")
plt.show()'''
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#查准率不错！