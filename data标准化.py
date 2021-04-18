from sklearn import preprocessing
import numpy as np
data=np.array([[3,-1.5,2,-5.4],[0,4,-0.3,2.1],[1,3.3,-1.9,-4.3]])
datazscore=preprocessing.scale(data) #data是正常的数据表 列标准化
datazscore2=preprocessing.scale(data, axis=1)#这次按行标准化了
datanormal=preprocessing.normalize(data, norm='l1')#按L1范数行标准化（每个元除以该行的范数）
datanormal2=preprocessing.normalize(data,norm='l1',axis=0)#按L1范数列标准化