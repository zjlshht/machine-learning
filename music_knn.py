from pandas.io.parsers import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
A = read_csv("Data_on_year.csv", index_col=0).values
X = A[:, :13]
y = A[:, 13]
seed = 666  # 随机种子
test_size = 0.25  # 测试集比例
'''划分流行与不流行'''
for i in range(len(y)):
    if y[i] >= 0.6:  # 0.6为分界线
        y[i] = 1
    else:
        y[i] = 0
threshold = y.mean()
'''划分训练集和测试集并对训练集随机过采样'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed)
'''自变量X标准化'''
scaler = StandardScaler()
scaler.fit(X_train[:, :11])
X_train[:, :11] = scaler.transform(X_train[:, :11])
X_test[:, :11] = scaler.transform(X_test[:, :11])
'''神经网络建模'''
model = Sequential()
model.add(Dense(units=20, activation='sigmoid', input_dim=13))  # 输入层-隐藏层
model.add(Dense(units=1, activation='sigmoid'))  # 隐藏层-输出层
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])  # 设定损失判别标准
model.fit(X_train, y_train, epochs=10, batch_size=32)  # 对训练集训练
predictions = model.predict(X_test)  # 对测试集作预测
predictions[predictions >= threshold] = 1  # 根据分类阈值划分
predictions[predictions < threshold] = 0
accuracy = accuracy_score(y_test, predictions)  # 计算准确率
print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''画混淆矩阵图'''
C2 = confusion_matrix(y_test, predictions, labels=[0, 1])
print(C2)  # 打印出来看看
