#yi=a+b1*xi1+...+bk*xik

#beta=[alpha,beta1,beta2,...,betak]
#x_i=[1,x_i1,...,x_ik]
from function import dot,Vector
def predict(x:Vector,beta:Vector)->float:
    '''assumes that the first element of x is 1'''
    return dot(x,beta)
def error(x:Vector,y:float,beta:Vector)->float:
    return predict(x,beta)-y

def squared_error(x:Vector,y:float,beta:Vector)->float:
    return error(x,y,beta)**2

def sqrror_gradient(x:Vector,y:float,beta:Vector)->Vector:
    err=error(x,y,beta)
    return [2*err*x_i for x_i in x]#残差平方的向量数据

x=[1,2,3]
y=30
beta=[4,4,4]#so prediction = 4+8+12=24
assert error(x,y,beta)==-6
assert squared_error(x, y, beta)==36
assert sqrror_gradient(x, y, beta)==[-12,-24,-36]
'''
这里用的梯度下降 后面就不写了
'''
