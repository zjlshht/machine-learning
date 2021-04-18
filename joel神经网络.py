from function import Vector
from numpy import dot
import numpy as np
from typing import List
def step_function(x:float)->float: 
    return 1.0 if x>=0 else 0.0

def perception_output(weights:Vector,bias:float,x:Vector)->float:
    '''Return 1 if the preception 'fires',0 if not'''
    calculation=dot(weights,x)+bias
    return step_function(calculation)

w1=[2,2]
bias1=-3
w2=[2,2]
bias2=-1
assert perception_output(w1, bias1, [1,1])==1
assert perception_output(w1, bias1, [0,1])==0
assert perception_output(w2, bias2, [1,1])==1
assert perception_output(w2, bias2, [0,1])==1
w3=[-2]
bias3=1
assert perception_output(w3, bias3, [0])==1
#of course,you don't need to approximate a neuron in order to build a logic gate

#Feed-Forward Neural Networks
def sigmoid(t:float)->float:
    return 1/(1+np.exp(-t))

def neuron_output(weights:Vector,inputs:Vector)->float:
    #weights includes the bias term,imputs includes a 1
    return sigmoid(dot(weights,input))

def feed_forward(neural_network:List[List[Vector]],
                 input_vector:Vector)->List[Vector]:
    outputs:List[Vector]=[]
    for layer in neural_network:
        input_with_bias=input_vector+[1]
        output=[neuron_output(neuron, input_with_bias)
                for neuron in layer]
        outputs.append(output)
        input_vector=output
    return outputs

'''
demo
'''
xor=[#hidden layer
     [[20.,20,-30],
      [20.,20,-10]],
     #output layer
     [[-60.,60,-30]]] #2ndd imput but not 1st input neuron
#feed_forward returns the outputs of all layers,so the [-1] gets the 
#final output,and the [0] gets the value out of the resulting vector
assert 0.000<feed_forward(xor, [0,0])[-1][0]<0.001
assert 0.999<feed_forward(xor, [1,0])[-1][0]<1.000
assert 0.999<feed_forward(xor, [0,1])[-1][0]<1.000
assert 0.000<feed_forward(xor, [1,1])[-1][0]<0.001
#不好用  算了 去用