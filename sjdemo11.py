import pandas as pd
from numpy.random import seed 
from numpy.random import rand
from numpy.random import random_integers
import numpy as np

seed(42)

df=pd.DataFrame({'Weather':['cold','hot','cold','hot','cold','hot','cold'],
                 'Food':['soup','soup','icecream','chocolate','icecream','icecream','soup'],
                 'Price':10*rand(7),
                 'Number':random_integers(1,9,size=(7,))})

print(df)
weather_group=df.groupby('Weather')#通过Weather列为数据分组，然后遍历各组数据
#weather_group是一种特殊的pandas对象
i=0
#根据天气情况 把数据分为两组
for name,group in weather_group:
    i=i+1
    print("Group",i,name)
    print(group)
    
print("Weather group first\n",weather_group.first())
print("Weather group last\n",weather_group.last())
print("Weather group mean\n",weather_group.mean())
#可以利用groups属性来了解所生成的数据组，以及每一组包含的行数：
'''
针对天气数据和食物数据的每一种可能的组合，都会为其生成一个新的数据组。每一行的各个
数据项都可以通过索引值引用，具体如下所示：
'''
wf_group=df.groupby(['Weather','Food'])
#print(wf_group.groups()) 这个运行不了了
'''
通过agg()方法，可以对数据组施加一系列的NumPy函数：
'''
print("WF Aggregated\n",wf_group.agg([np.mean,np.median]))

'''
数据库的数据表有内部连接和外部连接两种连接操作类型。实际上，pandas的DataFrame也有
类似的操作，因此我们也可以对数据行进行串联和附加。
'''
print("df前三行\n",df[:3])
'''
函数concat()的作用是串联DataFrame，如可以把一个由3行数据组成的DataFrame与其他数据
行串接，以便重建原DataFrame
'''
print("Concat Back togrther\n",pd.concat([df[:3],df[3:]]))
'''
将前三个的内容和前三个后面的内容相拼接
'''

print("追加几个行\n",df[:3].append(df[5:]))

'''
连接DataFrame：
读取dest.csv和tips.csv两个文件
pandas提供的merge()函数或DataFrame的join()实例方法都能实现类似数据库的连接操作功能
默认情况下，join()实例方法会按照索引进行连接。使用关系型数据库查询语言（SQL）时，
可以进行内部连接、左外连接、右外连接与完全外部连接等操作。
    用merge()函数按照员工编号进行连接：
    pd.merge(dests,tips,on='EmpNr') 

    用join()方法执行连接操作时，需要使用后缀来指示左操作对象和右操作对象：
    dests.join(tips,lsuffix='Dest',rsuffix='Tips') 这个方法会连接索引值，因此得
    到的结果与SQl内部连接会有所不同
    
    用merge()执行内部连接时，更显式的方法如下所示：
        pd.merge(dests,tips,how='inner')
    只要稍作修改，就可以变成完全外部连接：
        pd.merge(dests,tips,how='outer')
        
    如果使用关系型数据库的查询操作，这些数据都会被设为NULL。
    
处理缺失数据问题：
    对于pandas来说，它会把缺失的数值标为NaN，还有一个类似的符号是NaT，不过，它代表
    的是datetime64型对象。对NaN这个数值进行算术运算时，得到的结果还是NaN。有时NaN
    会被当作零值来进行处理，有时会被忽略。
    
    pandas的isnull()函数可以帮助我们检查缺失值，使用方法如下：
        pf.isnull(df) #返回布尔值
    类似的，可以用DataFrame的notnull()方法来考察非缺失值：
        df.notnull() #返回布尔值
    通过fillna()方法，可以用一个标量值来替换缺失数据，如：
        df.fillna(2) #用2代替缺失值
    

'''


























