from function import Vector,dot,distance,add,scalar_multiply

def sum_of_squares(v:Vector)->float:
    return dot(v,v)

from typing import Callable

def difference_quotient(f:Callable[[float],float],
                        x:float,
                        h:float)->float:
    return(f(x+h)-f(x))/h

def square(x:float)->float:
    return x*x

def derivative(x:float)->float:
    return 2*x

def partial_difference_quotient(f:Callable[[Vector],float],
                                v:Vector,
                                i:int,
                                h:float)->float:
    '''returns the i-th partial diference quotient of f at v'''
    w = [v_j +(h if j==i else 0) for j,v_j in enumerate(v)]
    return (f(w)-f(v))/h

def estimate_gradient(f:Callable[[Vector].float],
                      v:Vector,
                      h:float=0.0001):
    return [partial_difference_quotient(f,v,i,h) for i in range(len(v))]

#import random
def gradient_step(v:Vector,gradient:Vector,step_size:float)->Vector:
    '''Moves step_size in the gradient direction from v'''
    assert len(v)==len(gradient)
    step=scalar_multiply(step_size,gradient)
    return add(v,step)

def sum_of_squares_gradient(v:Vector)->Vector:
    return [2*v_i for v_i in v]

    