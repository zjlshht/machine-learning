from pandas.io.parsers import read_csv
from sklearn import linear_model
#from matplotlib import pyplot as plt
#import numpy as np
#import pandas as pd
#np.random.seed(0)
data=read_csv("data.csv")
#data.describe()
c=data['year']
d=data['popularity']
import simple_linear_regression
(alpha,beta)=simple_linear_regression.least_squares_fit(c,d)
simple_linear_regression.r_squared(alpha,beta,c,d)
c=c.values.reshape(169909,1)
d=d.values.reshape(169909,1)
linear_regressor=linear_model.LinearRegression()
linear_regressor.fit(c,d)
import matplotlib.pyplot as plt
import sklearn.metrics as sm
d_pred=linear_regressor.predict(c)
plt.figure()
plt.scatter(c,d,color='green')
plt.plot(c,d_pred,color='black',linewidth=4)
plt.title('Training data')
plt.show()
print("R2 score=",round(sm.r2_score(c,d_pred),2))
ridge_regressor=linear_model.Ridge(alpha=0.05,fit_intercept=True,max_iter=10000)
'''
the alpha controls the complexity,越接近0越接近线性回归
'''
ridge_regressor.fit(c,d)
d_pred_rid=ridge_regressor.predict(c)
print("R2 score=",round(sm.r2_score(c, d_pred_rid),2))
                                   
'''
from pandas.io.parsers import read_csv
#from simple_linear_regression import r_squared
#from matplotlib import pyplot as plt
#import numpy as np
#import pandas as pd
#np.random.seed(0)
data=read_csv("data.csv")
#data.describe()
c=data['year']
d=data['popularity']
import simple_linear_regression
(alpha,beta)=simple_linear_regression.least_squares_fit(c,d)
#simple_linear_regression.r_squared(alpha,beta,c,d)
r=simple_linear_regression.r_squared(alpha,beta,c,d)
'''