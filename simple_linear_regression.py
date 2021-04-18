#y=beta*x
def predict(alpha:float,beta:float,x_i:float)->float:
    return beta * x_i+alpha

def error(alpha:float,beta:float,x_i:float,y_i:float)->float:
    '''
    The error from predicting beta*x_i + alpha
    when the actual value is y_i
    Parameters
    ----------
    alpha : float
        DESCRIPTION.
    beta : float
        DESCRIPTION.
    x_i : float
        DESCRIPTION.
    y_i : float
        DESCRIPTION.
    Returns
    -------
    float
        DESCRIPTION.
    '''
    return predict(alpha, beta, x_i)-y_i

from function import Vector,correlation,standard_deviation,mean,de_mean
def sum_of_sqerrors(alpha:float,beta:float,x:Vector,y:Vector)->float:
    return sum(error(alpha,beta,x_i,y_i)**2 for x_i,y_i in zip(x,y))

from typing import Tuple 

def least_squares_fit(x:Vector,y:Vector)->Tuple[float,float]:
    '''
    Given two vectors x and y
    find the least-squares values of alpha and bera
    '''
    beta=correlation(x,y)*standard_deviation(y)/standard_deviation(x)
    alpha=mean(y)-beta*mean(x)
    return alpha,beta

#测试
x=[i for i in range(-100,110,10)]
y=[3*i-5 for i in x]
#should find that y=3x-5
assert least_squares_fit(x,y)==(-5,3)

def total_sum_of_squares(y:Vector)->float:
    return sum(v**2 for v in de_mean(y))

def r_squared(alpha:float,beta:float,x:Vector,y:Vector)->float:
    ''' 
    the fraction of variation in y captured by the model,which equals
    1 - the fraction of variation in y not captured by the model
    越大效果越好
    '''
    return 1.0-(sum_of_sqerrors(alpha,beta,x,y)/total_sum_of_squares(y))

    