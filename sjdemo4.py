import numpy as np
A=np.mat("3 -2;1 0")
print("特征值",np.linalg.eigvals(A))
'''
利用eig函数取得特征值和特征向量 特征向量是列向量 按顺序横向叠加
'''
eigenvalues,eigenvectors=np.linalg.eig(A)
print("特征值\n",eigenvalues)
print("特征向量\n",eigenvectors)

'''
验算
'''
for i in range(len(eigenvalues)):
    print("AX\n",np.dot(A,eigenvectors[:,i]))
    print("lamdaX\n",eigenvalues[i]*eigenvectors[:,i])


'''
pandas可以读csv 可以试试
'''

