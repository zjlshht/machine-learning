from pandas.io.parsers import read_csv
import pandas as pd
#import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
data=read_csv("data_train.csv")#读取数据
#对金额做特征工程
def amount(x):
    if x<0:
        return 0
    elif x<10:
        return 1
    elif x<100:
        return 2
    elif x<1000:
        return 3
    elif x<10000:
        return 4
    else:return 5
data['amount']=data['amount'].apply(amount)
seed=666
test_size=0.25
'''划分训练集和测试集并对训练集随机过采样'''
data_train,data_test= train_test_split(data,test_size=test_size, random_state=seed)
ss=data_train[data_train['fraud']==1].groupby('customer_id').count()
yy=data_train[data_train['fraud']==1].groupby('merchant_id').count()
#对顾客id做特征工程
train=data_train.copy()
test=data_test.copy()
train['customer_id'][~data_train['customer_id'].isin(data_train[data_train['fraud']==1]['customer_id'])]=0
train['customer_id'][data_train['customer_id'].isin(ss[ss['fraud']>=1].index)]=1
train['customer_id'][data_train['customer_id'].isin(ss[ss['fraud']>=2].index)]=2
train['customer_id'][data_train['customer_id'].isin(ss[ss['fraud']>=4].index)]=3
train['customer_id'][data_train['customer_id'].isin(ss[ss['fraud']>=8].index)]=4
train['customer_id'][data_train['customer_id'].isin(ss[ss['fraud']>=16].index)]=5
train['customer_id'][data_train['customer_id'].isin(ss[ss['fraud']>=32].index)]=6
train['customer_id'][data_train['customer_id'].isin(ss[ss['fraud']>=64].index)]=7
test['customer_id'][~data_test['customer_id'].isin(data_train[data_train['fraud']==1]['customer_id'])]=0
test['customer_id'][data_test['customer_id'].isin(ss[ss['fraud']>=1].index)]=1
test['customer_id'][data_test['customer_id'].isin(ss[ss['fraud']>=2].index)]=2
test['customer_id'][data_test['customer_id'].isin(ss[ss['fraud']>=4].index)]=3
test['customer_id'][data_test['customer_id'].isin(ss[ss['fraud']>=8].index)]=4
test['customer_id'][data_test['customer_id'].isin(ss[ss['fraud']>=16].index)]=5
test['customer_id'][data_test['customer_id'].isin(ss[ss['fraud']>=32].index)]=6
test['customer_id'][data_test['customer_id'].isin(ss[ss['fraud']>=64].index)]=7
#对商家id做特征工程
train['merchant_id'][~data_train['merchant_id'].isin(data_train[data_train['fraud']==1]['merchant_id'])]=0
train['merchant_id'][data_train['merchant_id'].isin(yy[yy['fraud']>=1].index)]=1
train['merchant_id'][data_train['merchant_id'].isin(yy[yy['fraud']>=2].index)]=2
train['merchant_id'][data_train['merchant_id'].isin(yy[yy['fraud']>=4].index)]=3
train['merchant_id'][data_train['merchant_id'].isin(yy[yy['fraud']>=8].index)]=4
train['merchant_id'][data_train['merchant_id'].isin(yy[yy['fraud']>=16].index)]=5
train['merchant_id'][data_train['merchant_id'].isin(yy[yy['fraud']>=32].index)]=6
train['merchant_id'][data_train['merchant_id'].isin(yy[yy['fraud']>=64].index)]=7
train['merchant_id'][data_train['merchant_id'].isin(yy[yy['fraud']>=108].index)]=8
train['merchant_id'][data_train['merchant_id'].isin(yy[yy['fraud']>=256].index)]=9
train['merchant_id'][data_train['merchant_id'].isin(yy[yy['fraud']>=512].index)]=10
test['merchant_id'][~data_test['merchant_id'].isin(data_train[data_train['fraud']==1]['merchant_id'])]=0
test['merchant_id'][data_test['merchant_id'].isin(yy[yy['fraud']>=1].index)]=1
test['merchant_id'][data_test['merchant_id'].isin(yy[yy['fraud']>=2].index)]=2
test['merchant_id'][data_test['merchant_id'].isin(yy[yy['fraud']>=4].index)]=3
test['merchant_id'][data_test['merchant_id'].isin(yy[yy['fraud']>=8].index)]=4
test['merchant_id'][data_test['merchant_id'].isin(yy[yy['fraud']>=16].index)]=5
test['merchant_id'][data_test['merchant_id'].isin(yy[yy['fraud']>=32].index)]=6
test['merchant_id'][data_test['merchant_id'].isin(yy[yy['fraud']>=64].index)]=7
test['merchant_id'][data_test['merchant_id'].isin(yy[yy['fraud']>=108].index)]=8
test['merchant_id'][data_test['merchant_id'].isin(yy[yy['fraud']>=256].index)]=9
test['merchant_id'][data_test['merchant_id'].isin(yy[yy['fraud']>=512].index)]=10
#对type做特征工程
train['type'][~data_train['type'].isin(data_train[data_train['fraud']==1]['type'])]=0
train['type'][data_train['type'].isin(data_train[data_train['fraud']==1]['type'])]=1
test['type'][~data_test['type'].isin(data_train[data_train['fraud']==1]['type'])]=0
test['type'][data_test['type'].isin(data_train[data_train['fraud']==1]['type'])]=1
#制作one-hot编码
onehot=train[['customer_id','age_group','gender','merchant_id','type','amount']].values
train2onehot=train[['customer_id','age_group','gender','merchant_id','type','amount']].values
test2onehot=test[['customer_id','age_group','gender','merchant_id','type','amount']].values
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(onehot)
X_train_time=data_train['time']/180
X_train_time.index=range(len(X_train_time))
X_test_time=data_test['time']/180
X_test_time.index=range(len(X_test_time))
X_train=pd.DataFrame(enc.transform(train2onehot).toarray())
X_test=pd.DataFrame(enc.transform(test2onehot).toarray())
X_train=pd.concat([X_train,X_train_time],axis=1)
X_test=pd.concat([X_test,X_test_time],axis=1)
y_train=data_train['fraud']
y_test=data_test['fraud']

X_train.to_csv('aliX_train.csv')#只写出部分列
X_test.to_csv('aliX_test.csv')
y_train.to_csv('aliy_train.csv')
y_test.to_csv('aliy_test.csv')

#处理未给指标的测试集
Ddata=read_csv("data_test.csv")
#先转换amount
Ddata['amount']=Ddata['amount'].apply(amount)
#再转换customer_id
Ddata['customer_id'][~Ddata['customer_id'].isin(data_train[data_train['fraud']==1]['customer_id'])]=0
Ddata['customer_id'][Ddata['customer_id'].isin(ss[ss['fraud']>=1].index)]=1
Ddata['customer_id'][Ddata['customer_id'].isin(ss[ss['fraud']>=2].index)]=2
Ddata['customer_id'][Ddata['customer_id'].isin(ss[ss['fraud']>=4].index)]=3
Ddata['customer_id'][Ddata['customer_id'].isin(ss[ss['fraud']>=8].index)]=4
Ddata['customer_id'][Ddata['customer_id'].isin(ss[ss['fraud']>=16].index)]=5
Ddata['customer_id'][Ddata['customer_id'].isin(ss[ss['fraud']>=32].index)]=6
Ddata['customer_id'][Ddata['customer_id'].isin(ss[ss['fraud']>=64].index)]=7
#再转换merchant_id
Ddata['merchant_id'][~Ddata['merchant_id'].isin(data_train[data_train['fraud']==1]['merchant_id'])]=0
Ddata['merchant_id'][Ddata['merchant_id'].isin(yy[yy['fraud']>=1].index)]=1
Ddata['merchant_id'][Ddata['merchant_id'].isin(yy[yy['fraud']>=2].index)]=2
Ddata['merchant_id'][Ddata['merchant_id'].isin(yy[yy['fraud']>=4].index)]=3
Ddata['merchant_id'][Ddata['merchant_id'].isin(yy[yy['fraud']>=8].index)]=4
Ddata['merchant_id'][Ddata['merchant_id'].isin(yy[yy['fraud']>=16].index)]=5
Ddata['merchant_id'][Ddata['merchant_id'].isin(yy[yy['fraud']>=32].index)]=6
Ddata['merchant_id'][Ddata['merchant_id'].isin(yy[yy['fraud']>=64].index)]=7
Ddata['merchant_id'][Ddata['merchant_id'].isin(yy[yy['fraud']>=108].index)]=8
Ddata['merchant_id'][Ddata['merchant_id'].isin(yy[yy['fraud']>=256].index)]=9
Ddata['merchant_id'][Ddata['merchant_id'].isin(yy[yy['fraud']>=512].index)]=10
#再转换type
Ddata['type'][~Ddata['type'].isin(data_train[data_train['fraud']==1]['type'])]=0
Ddata['type'][Ddata['type'].isin(data_train[data_train['fraud']==1]['type'])]=1
#取出特定值
Ddata2onehot=Ddata[['customer_id','age_group','gender','merchant_id','type','amount']].values
#时间指标做归一化
Ddata_time=Ddata['time']/180
#转码
X_final_test=pd.DataFrame(enc.transform(Ddata2onehot).toarray())
X_final_test=pd.concat([X_final_test,Ddata_time],axis=1)
#从Dataframe中取值
X_final=X_final_test.values

