from pandas.io.parsers import read_csv
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

X_train=read_csv("aliX_train.csv",index_col=0).values
X_test=read_csv("aliX_test.csv",index_col=0).values

y_train=read_csv("aliy_train.csv",index_col=0).values
y_test=read_csv("aliy_test.csv",index_col=0).values

model = Sequential()
model.add(Dense(units=100, activation='sigmoid', input_dim=36))#输入层-隐藏层
model.add(Dense(units=1,activation='sigmoid'))#隐藏层-输出层
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])#设定损失判别标准
model.fit(X_train, y_train, epochs=10, batch_size=32)#对训练集训练
predictions=model.predict(X_test) #对测试集作预测
threshold=sum(y_train)/len(y_train)
predictions[predictions>=threshold]=1 #根据分类阈值划分
predictions[predictions<threshold]=0 
accuracy = accuracy_score(y_test, predictions)#计算准确率

C2= confusion_matrix(y_test, predictions, labels=[0 , 1])
print(C2) #打印出来看看
