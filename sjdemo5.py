import numpy as np
from matplotlib.pyplot import plot

cash=np.zeros(10000)#生成一个10000元的列向量
cash[0]=1000#初值为1000R
outcome=np.random.binomial(9,0.5,size=len(cash))
'''
生成二项分布 其中实验9次 成功概率是0.5 生成len(cash)个结果
'''
for i in range(len(cash)):
    if outcome[i]<5:
        cash[i]=cash[i-1]-1
    elif outcome[i]<10:
        cash[i]=cash[i-1]+1
    else:
        raise AssertionError("Unexpected outcome"+outcome)
    
print(outcome.min(),outcome.max())

plot(np.arange(len(cash)),cash)#画出现金流的图

           
    