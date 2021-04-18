import numpy as np
a=np.arange(9).reshape(3,3)
b=2*a

c=np.hstack((a,b))#矩阵水平叠加

d=np.vstack((a,b))#矩阵垂直叠加

print("np.hsplit(a,3)")
print(np.hsplit(a,3)) #拆成三个列

print("np.xsplit(a,3)")
print(np.vsplit(a,3)) #拆成三个行

'''
深度拆分
'''
g=np.arange(27).reshape(3,3,3)

print("np.dsplit(g,3)")
print(np.dsplit(g, 3))  #暂时不知道哪可以用
'''
深度叠加
'''
f=np.hsplit(a,3)
print("np.dstack(f)")
print(np.dstack(f))  #似乎更没用

a.flat=0#把a中元素全赋值为0



