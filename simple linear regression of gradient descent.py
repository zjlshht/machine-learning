import random 
import tqdm
from gradient_descent_function import gradient_step
from function import Vector
num_epochs=10000
random.seed(0)

guess=[random.random(),random.random()] #choose random value to start

learning_rate=0.00001
def predict(alpha:float,beta:float,x_i:float)->float:
    return beta * x_i+alpha
def error(alpha:float,beta:float,x_i:float,y_i:float)->float:
    return predict(alpha, beta, x_i)-y_i
def sum_of_sqerrors(alpha:float,beta:float,x:Vector,y:Vector)->float:
    return sum(error(alpha,beta,x_i,y_i)**2 for x_i,y_i in zip(x,y))
a=[i*i for i in range(10)]

b=[i^2 for i in range(10)]

with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha,beta=guess
        grad_a=sum(2*error(alpha,beta,x_i,y_i) for x_i,y_i in zip(a,b))
        grad_b=sum(2*error(alpha,beta,x_i,y_i)*x_i for x_i,y_i in zip(a,b))
        loss=sum_of_sqerrors(alpha,beta,a,b)
        t.st_description(f"loss:{loss:.3f}")
        guess=gradient_step(guess,[grad_a,grad_b],-learning_rate)
        
#运行不了 醉了