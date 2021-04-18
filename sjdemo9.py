import quandl #下载数据
sunspots=quandl.get("SIDC/SUNSPOTS_A") 
'''
head(n)和tail(n)两个函数分别是选取前n个和后n个数据记录，其中n是一个整数参数：
'''
print("头两个\n",sunspots.head(2))
print("末两个\n",sunspots.tail(2))

last_data=sunspots.index[-1]#最近一个值
print("Last value\n",sunspots.loc[last_data])#输出最近的一个值
'''
输出指定时间段的值
'''
print("Values slice by data\n",sunspots["20020101":"20131231"])
'''
索引列表也可也用于查询
'''
print("Slice from a list of indices\n",sunspots.iloc[[2,4,-4,-2]])
'''
同时可以查询指定行和列 iloc和iat
'''
print("第一行第一列\n",sunspots.iloc[0,0])
print("第二行第一列\n",sunspots.iat[1,0])

print("查询大于算术平均值的各个数组\n",sunspots[sunspots>sunspots.mean()])
#print("查询大于算术平均值的数组的列标签",sunspots[sunspots.Number>sunspots.Number.mean()])
'''
第一个会返回所有行，其中与条件不符的行会被赋予NaN值，第二个查询操作返回的值是其值大
于平均值的那些行，不过这里Number不能用
'''
