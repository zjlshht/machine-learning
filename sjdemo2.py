import numpy as np
a=np.arange(9).reshape(3,3)
b=a.tolist()# 转化成列表

'''
astype()函数可以把数据元素转换成指定类型
'''
d=np.array([1.+1.j,3.+2.j])#生成一个复数数组
c=d.astype(int)#化成整数会将虚数部分抛弃

a=np.arange(81).reshape(9,9)

b=a.copy()

print("a>a.max()/4")
print(a>a.max()/4)
print("a<a.max()*3/4")
print(a<a.max()*3/4)
print("(a>a.max()/4)&(a<3*a.max()/4)")
b[(a>a.max()/4)&(a<3*a.max()/4)]=0#利用布尔值