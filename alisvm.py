from pandas.io.parsers import read_csv
from sklearn.metrics import confusion_matrix
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
X_train=read_csv("aliX_train.csv",index_col=0).values
X_test=read_csv("aliX_test.csv",index_col=0).values
y_train=read_csv("aliy_train.csv",index_col=0).values
y_test=read_csv("aliy_test.csv",index_col=0).values
clf = make_pipeline(SVC(kernel='poly',random_state=0, tol=1e-5))#sigmoid最差
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
C2= confusion_matrix(y_test, y_pred, labels=[0, 1])
print(C2) 

