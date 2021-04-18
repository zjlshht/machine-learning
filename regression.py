#import sys
import numpy as np
'''filename=sys.argv[1]
x=[]
y=[]
with open(filename,'r') as f:
    for line in f.readlines():
        xt,yt=[float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)
需要提供一个名为data_singlevar.txt的文本，第一列元素是x，第二列是y        
''' 

#这个思路可以适当参考

'''
num_tr=int(0.8*len(x))
num_te=len(x)-num_tr
x_tr=np.array(x[:num_tr]).reshape((num_tr,1))
y_tr=np.array(y[:num_tr]).reshape((num_tr,1))
x_te=np.array(x[num_tr:]).reshape((nunm_te,1))
y_te=np.array(y[num_tr:])
'''
x_tr=np.array([1,2.5,3,4,5,6]).reshape(6,1)
y_tr=np.array([1,3.5,7.4,10.7,12.8,15.4]).reshape(6,1)
x_te=np.array([7,8,9]).reshape(3,1)
y_te=np.array([19.9,22,25.4])
from sklearn import linear_model
linear_regressor=linear_model.LinearRegression()
linear_regressor.fit(x_tr,y_tr)
import matplotlib.pyplot as plt
y_tr_pred=linear_regressor.predict(x_tr)
plt.figure()
plt.scatter(x_tr,y_tr,color='green')
plt.plot(x_tr,y_tr_pred,color='black',linewidth=4)
plt.title('Training data')
plt.show()
y_te_pred=linear_regressor.predict(x_te)

#compute accuray
import sklearn.metrics as sm
print("mean absolute error=",round(sm.mean_absolute_error(y_te,y_te_pred),2))
print("mean squared error=",round(sm.mean_squared_error(y_te, y_te_pred),2))
print("median absolute error=",round(sm.median_absolute_error(y_te, y_te_pred),2))
print("explained variance score=",round(sm.explained_variance_score(y_te, y_te_pred),2))
print("R2 score=",round(sm.r2_score(y_te,y_te_pred),2))


#下面是岭回归
ridge_regressor=linear_model.Ridge(alpha=0.01,fit_intercept=True,max_iter=10000)
'''
the alpha controls the complexity,越接近0越接近线性回归
'''
ridge_regressor.fit(x_tr,y_tr)
y_te_pred_rid=ridge_regressor.predict(x_te)
print("R2 score=",round(sm.r2_score(y_te, y_te_pred_rid),2))

#下面是多项式回归
'''from sklearn.preprocessing import PolynomialFeatures
polynomial=PolynomialFeatures(degree=3)
data=[0.39,2.78,7.11]
poly=polynomial.fit_transform(data)
poly_linear_model=linear_model.LinearRegression()
x_tr2=np.array([1,2,3,4]).reshape(4,1)
y_tr2=np.array([-1,0,9,32]).reshape(4,1)
poly_linear_model.fit(x_tr2,y_tr2)
print(linear_regressor.predict(data)[0])
print(poly_linear_model.predict(poly[0]))
'''
#改绿是因为跑不了，但是可以考虑构造x，x^2,x^3,...,的形式做线性拟合
#如下：
import matplotlib.pyplot as plt 
#import numpy as np 
x=[1,2,3,4,5,6,7,8]
y=[1,4,9,13,30,25,49,70]
a=np.polyfit(x,y,2)#用2次多项式拟合x，y数组
b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
c=b(x)#生成多项式对象之后，就是获取x在这个多项式处的值
plt.scatter(x,y,marker='o',label='original datas')#对原始数据画散点图
plt.plot(x,c,ls='--',c='red',label='fitting with second-degree polynomial')#对拟合之后的数据，也就是x，c数组画图
plt.legend()
plt.show()
 
                                   