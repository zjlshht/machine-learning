from pandas.io.parsers import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
X_train=read_csv("aliX_train.csv",index_col=0).values
X_test=read_csv("aliX_test.csv",index_col=0).values
y_train=read_csv("aliy_train.csv",index_col=0).values
y_test=read_csv("aliy_test.csv",index_col=0).values
'''跑得k=5时最优
Score=[]
for i in range(1,10):
    neigh = KNeighborsClassifier(n_neighbors=i)#k=10
    neigh.fit(X_train,y_train)
    y_pred=neigh.predict(X_test)
    a = accuracy_score(y_test, y_pred)
    Score.append(a)
    sns.set()
    f,ax=plt.subplots()
    C2= confusion_matrix(y_test, y_pred, labels=[0 , 1])
    print(C2) 
'''
neigh = KNeighborsClassifier(n_neighbors=5)#k=5
neigh.fit(X_train,y_train)
y_pred=neigh.predict(X_test)
C2= confusion_matrix(y_test, y_pred, labels=[0 , 1])
print(C2) 
