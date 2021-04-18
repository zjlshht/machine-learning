'''
利用pandas的DataFrame进行统计计算
首先了解下函数
describe：返回描述性统计信息
count：返回非NaN数据项的数量
mad：计算平均绝对偏差，一个类似于标准差的有力统计工具
median：返回中位数
min，max：顾名思义
mode：返回众数
std，var：顾名思义
skew：偏态系数，偏度
kurt：峰态系数，峰度
'''

'''
演示代码
'''
import quandl
ss=quandl.get("SIDC/SUNSPOTS_A")
print("describde\n",ss.describe())
print("count\n",ss.count())
print("mad\n",ss.mad())
print("median\n",ss.median())
print("min\n",ss.min,"max\n",ss.max())
print("mode\n",ss.mode())
print("std\n",ss.std(),"var\n",ss.var())
print("skew\n",ss.skew(),"kurt\n",ss.kurt())