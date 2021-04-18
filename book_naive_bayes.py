#朴素贝叶斯回归分类器
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
def plot_classifier(classifier,X,Y):#这个函数似乎很好用啊！
    x_min,x_max=min(X[:,0])-1.0,max(X[:,0])+1.0
    y_min,y_max=min(X[:,1])-1.0,max(X[:,1])+1.0
    step_size=0.01
    x_values,y_values=np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))
    mesh_output=classifier.predict(np.c_[x_values.ravel(),y_values.ravel()])
    mesh_output=mesh_output.reshape(x_values.shape)
    plt.figure()
    plt.pcolormesh(x_values,y_values,mesh_output,cmap=plt.cm.gray)
    plt.scatter(X[:,0],X[:,1],c=y,s=80,edgecolors='black',linewidth=1,cmap=plt.cm.Paired)
    #specify the boundaries of the figure
    plt.xlim(x_values.min(),x_values.max())
    plt.ylim(y_values.min(),y_values.max())
    plt.xticks((np.arange(int(min(X[:,0])-1),int(max(X[:,0])+1),1.0)))#这两行加不加似乎没影响
    plt.yticks((np.arange(int(min(X[:,1])-1),int(max(X[:,1])+1),1.0)))
    plt.show()
'''
我从book_logistic_regression调用函数时，会把里面该函数绘的图一起画出来，因此直接把函数复制过来
'''
#you were provided with a data_multivar.txt file. This contains data that we will
#use here.This contains comma_separated numerical data in each line. Let's load data
#from this file:
'''
input_file = 'data_multivar.txt'
x=[]
y=[]
with open(input_file,'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        x.append(data[:-1])
        y.append(data[-1])

x=np.array(x)
y=np.array(y)
'''
#we now loaded the input data into x and the labels into y
import numpy as np
x=np.array([[4,7],[3.5,8],[3.1,6.2],[0.5,1],[1,2],[1.2,1.9],[6,2],[5.7,1.5],[5.4,2.2]])
y=np.array([0,0,0,1,1,1,2,2,2])
#let's build the naive bayes classifier:
classifier_gaussiannb=GaussianNB()
classifier_gaussiannb.fit(x,y)
y_pred=classifier_gaussiannb.predict(x)
#the GaussianNB function specifies Gaussian Naive Bayes model
#Let's compute the accuracy of the classifier:
accuracy=100.0*(y==y_pred).sum()/x.shape[0]
print("accuracy",round(accuracy,2),"%")
#let's plot the data and the boundaries 
plot_classifier(classifier_gaussiannb,x,y)
