import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pandas.io.parsers import read_csv
from imblearn.over_sampling import RandomOverSampler
A=read_csv("Data_on_year.csv",index_col=0).values
X=A[:,:13]
y=A[:,13]
seed=666#随机种子
test_size=0.25#测试集比例
'''划分流行与不流行'''
for i in range(len(y)):
    if y[i]>=0.6: #0.6为分界线
        y[i]=1
    else:
        y[i]=0
#划分数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=666)
clf = RandomForestClassifier(max_depth=4, random_state=0) 
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
clf.fit(X_resampled, y_resampled)

#clf.fit(X_train,y_train)
y_pr=clf.predict(X_test)
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

print(clf.score(X_test,y_test))