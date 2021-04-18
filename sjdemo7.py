import numpy as np
a=np.array([1,2,-2,1,45,-1,2,4,2,1,-3])

triples=np.arange(0,len(a),3)
print("Triples",triples[:10],"...")
signs=np.ones(len(a))
print("Signs",signs[:10],"...")

signs[triples]=-1
print("Signs",signs[:10])

ma_log=np.ma.log(a*signs)
print("Masked logs",ma_log[:10],"...")

'''
上面说是能找到负数 但是并没有发现相应的作用
'''

dev=a.std()
avg=a.mean()
inside=np.ma.masked_outside(a,avg-dev,avg+dev)
print("Inside",inside[:10],"...")

'''
这个能去极值 结果在inside中
'''
