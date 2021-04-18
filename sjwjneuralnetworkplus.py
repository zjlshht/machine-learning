from pandas.io.parsers import read_csv
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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
'''划分流行与不流行'''
for i in range(len(y)):
    if y[i]>=0.6: #0.6为分界线
        y[i]=1
    else:
        y[i]=0
ros = RandomOverSampler(random_state=0)#设定过采样函数
'''划分训练集和测试集并对训练集随机过采样'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size, random_state=seed)
X_train, y_train = ros.fit_sample(X_train, y_train)
'''自变量X标准化'''        
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
'''神经网络建模'''
model = Sequential()
model.add(Dense(units=20, activation='relu', input_dim=13))#输入层-隐藏层
model.add(Dense(units=1,activation='sigmoid'))#隐藏层-输出层
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])#设定损失判别标准
model.fit(X_train, y_train, epochs=10, batch_size=32)#对训练集训练
predictions=model.predict(X_test) #对测试集作预测
predictions[predictions>=0.6]=1 #根据分类阈值划分
predictions[predictions<0.6]=0 
accuracy = accuracy_score(y_test, predictions)#计算准确率
print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''画混淆矩阵图'''
sns.set()
f,ax=plt.subplots()
C2= confusion_matrix(y_test, predictions, labels=[0 , 1])
print(C2) #打印出来看看
sns.heatmap(C2,annot=True,ax=ax) #画热力图
ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴
plt.show()
'''画roc曲线''' 
fpr,tpr,threshold = roc_curve(y_test, predictions) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
