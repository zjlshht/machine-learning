'''
处理日期数据
'''
import pandas as pd
'''
下面，我们设定一个自1900.1.1开始的为期42天的时间范围，具体如下：(这个方法只能创建
                                      1677-2262年之间的日期)
'''
print("Data range",pd.date_range('1/1/1900',periods=42,freq='D'))
'''
D是日历日频率 W是周频率 M是月末频率 MS是月初频率 Q季末频率 A年终频率
'''
'''
下面用pandas的DateOffset函数来计算允许的日期范围
'''
#offset=pd.DateOffset(seconds=2 ** 63/10 ** 9)
#mid=pd.to_datetime('1/1/1900')
#print("Start valid range\n",mid-offset)
#print("End valid range\n",mid+offset.T)
#运行不了 算了算了
'''
to_datetime()函数可以把字符型日期转换成日期数据
'''
a=pd.to_datetime(['19021112','19031230'])
b=pd.to_datetime(['19021112','19031230'],format='%Y%m%d')
print(a)
print(b) #其实a和b是一样的
#'''
#当然可以强制转换 不能啦 代码坏了
#'''
#c=pd.to_datetime(['1902-11-12','not a date'],coerce=True)
#print(c)
#'''
#由于第二个数据不是日期且强制输出，所以会以NaT的形式输出
#'''