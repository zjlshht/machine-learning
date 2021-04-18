import numpy as np
print("In:b=np.arange(24).reshape(2,3,4)")
b=np.arange(24).reshape(2,3,4)
print(b)


print("In:b.ravel()")#平摊 多维变成一维
print(b.ravel())


print("In:b.flatten()")#拉直
print(b.flatten())

print("In:b.shape=(6,4)")#指定形状
b.shape=(6,4)

print("In:b")
print(b)

print("In:b.transpose()")#转置
print(b.transpose())#等价于b.T

print("In:b.resize((2,12))")#重组 与shape类似
b.resize((2,12))

print("In:b")
print(b)
