import pandas as pd
filename='lmodel.xlsx'
data=pd.read_excel(filename)
x=data.iloc[:4,:8].values
y=data.iloc[:4,8].values

from sklearn.linear_model import LogisticRegression as LR
#from sklearn.linear_model import RandomizeLogisticRegression as RLR
#rlr=RLR()#建立随机逻辑回归模型，筛选变量
#rlr.fit(x,y) #训练模型
#rlr.get_support()#获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数
#print(u'通过随机逻辑回归模型筛选特征结束。')
#print(u'有效特征为：%s' % ','.join(data.columns[rlr.get_support()]))
#x=data[data.columns[rlr.get_support()]].as_matrix()#筛选特征

lr=LR()#建立逻辑回归模型
lr.fit(x,y)#用筛选后的特征数据来训练模型
print(u'逻辑回归模型训练结束。')
print(u'模型的正确率为：%s' % lr.score(x, y))


