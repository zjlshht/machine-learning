import numpy as np
a=np.array([[1,2,3],[0,1,0],[0,1,0]])  ##这是数组
b=np.mat("2 4 6;4 2 6;10 -4 18")   ##这是矩阵

print(np.linalg.pinv(a))#a不可逆
print(np.linalg.inv(b))

"""
如果是用数组array表示的矩阵*的时候是对应元素相乘
如果是用矩阵mat表示的矩阵*的时候是正常的矩阵乘法
如果是数组*矩阵 则按矩阵乘法算，得到的结果也是矩阵
"""

c=np.eye(3)
d=np.linalg.norm(c)#求二范数
'''
eye创建的也是数组
'''


A=np.mat("1 -2 1;0 2 -8;-4 5 9")
c=np.array([0,8,-9])
x=np.linalg.solve(A, c)#求解线性方程组Ax=c
print("sulution\n",x)

print("check\n",np.dot(A,x))#求矩阵A乘以矩阵x

