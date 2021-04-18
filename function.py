#from collections import Counter
#from matplotlib import pyplot as plt
import math,numpy,random
from typing import List,Callable,TypeVar,Tuple
X= TypeVar('X') 
Y= TypeVar('Y')
Vector=List[float]
Matrix=List[List[float]]
#import matplotlib.pyplot as plt
def make_matrix(num_rows:int,num_cols:int,
                entry_fn:Callable[[int,int],float])->Matrix:
    '''Returns a num_rows x num_cols matrix
    whose(i,j)-th entry is entry_fn(i,j)'''
    return [[entry_fn(i,j)
             for j in range(num_cols)]
            for i in range(num_rows)]

def ReLU(x):
    z=max(x)
    y=max(z,0)
    return y

def Sigmoid(x):
    y=numpy.zeros(len(x))
    for i in range(len(x)):
        y[i]=1/(1+math.exp(-x[i]))
    return y

def sigmoid(x):
    y=1/(1+math.exp(-x))
    return y

def Softmax(x):
    y=numpy.zeros(len(x))
    c=0
    for i in range(len(x)):
        y[i]=math.exp(x[i])
        c=c+y[i]
    for j in range(len(x)):
        y[j]=y[j]/c
    return y 

SQRT_TWO_PI=math.sqrt(2*math.pi)
def normal_pdf(x:float,mu:float=0,sigma:float=1)->float:
    return (math.exp(-(x-mu)**2/2/sigma**2)/(SQRT_TWO_PI*sigma))
'''
xs=[x/10.0 for x in range(-50,50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()
'''
def normal_cdf(x:float,mu:float=0,sigma:float=1)->float:
    return (1+math.erf((x-mu)/math.sqrt(2)/sigma))/2
'''
xs=[x/10.0 for x in range(-50,50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4)#bottom right
plt.title("Various Normal cdfs")
plt.show()
'''
def inverse_normal_cdf(p:float,
                       mu:float=0,
                       sigma:float=1,
                       tolerance:float=0.00001)->float:
    '''find approximate inverse using binary search'''
    if mu !=0 or sigma !=1:
        return mu+sigma*inverse_normal_cdf(p,tolerance=tolerance)
    low_z=-10.0
    hi_z=10.0
    while hi_z - low_z >tolerance:
        mid_z=(low_z+hi_z)/2
        mid_p=normal_cdf(mid_z)
        if mid_p<p:
            low_z=mid_z
        else:
            hi_z=mid_z
    return mid_z

def bernoulli_trial(p:float)->int:
    '''Returns 1 with probability p and 0 with probability 1-p'''
    return 1 if random.random()<p else 0

def binomial(n:int,p:float) ->int:
    '''Return the sum of n bernoulli(p) trials'''
    return sum(bernoulli_trial(p) for _ in range(n))

'''def diff(f:callable[[Array],float],
         v:Array,
         i:int,
         h:float) ->float:
    """returns the i-th partial difference quotient of f at v"""
    w=[v_j+(h if j==i else 0) #add h to just ith element of v
       for j,v_j in enumerate(v)]
    
    return (f(w)-f(v))/h'''
def sum_of_squares_gradient(v):
    return [2*v_i for v_i in v]

def random_normal() ->float:
    """Returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random())

def mean(xs:List[float])->float:
    return sum(xs)/len(xs)

def de_mean(xs:List[float]) ->List[float]:
    """translate xs by subtracting its muan (so the result has mean 0)"""
    x_bar=mean(xs)
    return [x-x_bar for x in xs]

def sum_of_squares(v:Vector) ->float:
    return numpy.dot(v,v)

def variance(xs:List[float])->float:
    assert len(xs)>2
    n=len(xs)
    deviations=de_mean(xs)
    return sum_of_squares(deviations)/(n-1)
    
def standard_deviation(xs:List[float])->float:
    return math.sqrt(variance(xs))

def covariance(xs:List[float],ys:List[float])->float:
    assert len(xs)==len(ys),"xs and ys must have same number of element"
    return numpy.dot(de_mean(xs),de_mean(ys))/(len(xs)-1) #E(X-E(X))(Y-E(Y))

def correlation(xs:List[float],ys:List[float])->float:
    '''measures how much xs and ys vary in tandem about their means'''
    stdev_x=standard_deviation(xs)
    stdev_y=standard_deviation(ys)
    if stdev_x > 0 and stdev_y >0:
        return covariance(xs,ys)/stdev_x/stdev_y
    else:
        return 0
    
def correlation_matrix(data:List[Vector])->Matrix:
    '''
    Returns the len(data) x len(data) matrix whose (i,j)-th entry
    is the correlation between data[i] and data[j]
    '''
    def correlation_ij(i:int,j:int)->float:
        return make_matrix(len(data),len(data),correlation_ij)
    
    return make_matrix(len(data), len(data), correlation_ij)
    
def split_data(data:List[X],prob:float)->Tuple[List[X],List[X]]:
    """Split data into fractions [prob,1 - prob]"""
    data=data[:] #make a shallow copy
    random.shuffle(data) #bacause shuffle modifies the list
    cut=int(len(data)*prob) #use prob to find a cutoff
    return data[:cut],data[cut:]#and split the shuffled list there
'''demo
data=[n for n in range(1000)]
train,test=split_data(data,0.75) #按0.75比例选取训练集
'''

'''选取带标签的数据集'''
def train_test_split(xs:List[X],
                     ys:List[Y],
                     test_pct:float)->Tuple[List[X],List[X],List[Y],List[Y]]:
    idxs=[i for i in range(len(xs))]
    train_idxs,test_idxs=split_data(idxs,1-test_pct)
    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs])

'''tp:true positive 真正
   fp:false positive 假正（假的预测为真）
   fn:false negative 假反（真的预测为假）
   tn:true negative 真反'''
def accuracy(tp:int,fp:int,fn:int,tn:int)->float:#准确率
    correct=tp+tn    
    total=tp+tn+fp+fn
    return correct/total

def precision(tp:int,fp:int,fn:int,tn:int)->float:
    return tp/(tp+fp) #查准率

def recall(tp:int,fp:int,fn:int,tn:int)->float:
    return tp/(tp+fn) #查全率

'''F1 score/harmonic mean'''
def f1_score(tp:int,fp:int,fn:int,tn:int)->float:
    p=precision(tp,fp,fn,tn)
    r=recall(tp,fp,fn,tn)
    return 2*p*r/(p+r)
def subtract(v:Vector,w:Vector)->Vector:
    '''Subtracts corresponding elements'''
    assert len(v)==len(w) 
    return [v_i-w_i for v_i,w_i in zip(v,w)]

def distance(v:Vector,w:Vector)->float:
    return math.sqrt(sum_of_squares(subtract(v, w)))

def dot(v:Vector,w:Vector) ->float:
    assert len(v) == len(w)
    return sum(v_i * w_i for v_i,w_i  in zip(v,w))
#和numpy里面的dot作用差不多
assert dot([1,2,3],[4,5,6])==32 #1*4+2*5+3*6

def add(v:Vector,w:Vector)->Vector:
    ''''adds corresponding elements'''
    assert len(v)==len(w)
    return [v_i + w_i for v_i,w_i in zip(v,w)]

def scalar_multiply(c:float,v:float)->Vector:
    return [c*v_i for v_i in v]



    

    
    