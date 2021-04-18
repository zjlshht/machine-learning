from pandas.io.parsers import read_csv
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
X_train=read_csv("aliX_train.csv",index_col=0).values
X_test=read_csv("aliX_test.csv",index_col=0).values
y_train=read_csv("aliy_train.csv",index_col=0).values
y_test=read_csv("aliy_test.csv",index_col=0).values
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
C2= confusion_matrix(y_test, y_pred, labels=[0 , 1])
print(C2) 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
