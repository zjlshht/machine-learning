'''
Pandas的DataFrame数据结构是一种带标签的二维对象，与Excel的电子表格或者关系型数据库的数据表
非常神似。可以用以下方式来创建DataFrame
1. 从另一个DataFrame创建DataFrame
2. 从具有二维形状的Numpy数组或者数组的复合结构来生成DataFrame
3. 可以用pandas的另外一钟数据结构series来创建DataFrame
4. 可以从类似CSV之类的文件来生成
'''
from pandas.io.parsers import read_csv

df=read_csv("555.csv")
print("DataFrame\n",df)

print("df的大小\n",df.shape)

print("df的行数\n",len(df))

print("列的标题\n",df.columns)

print("列标题的数据类型\n",df.dtypes)

print("Index\n",df.index)

print("遍历数组\n",df.values)

'''
Series
pandas的Series数据结构是由不同类型的元素组成的一维数组，该数据结构也具有标签。可以通过
以下方式来创建pandas的Series数据结构
1.由Python的字典来创建Series
2.由NumPy数组来创建Series
3.由单个标量值来创建Series
创建Series数据结构时，可以向构造函数递交一组轴标签，这些标签通常被称为索引，是一个
可选参数。默认情况下，如果使用NumPy数组作为输入数据，那么pandas会将索引值从0开始自
动递增。如果传递给构造函数的数据是一个Python字典，那么这个字典的键会经排序后变成相
应的索引：如果输入数据是一个标量值，那么就需要由我们来提供相应的索引。索引中的每一
个新值都要输入一个标量值。pandas的Series和DataFrame数据类型接口的特征和行为是从
NumPy数组和Python字典那么借用来的，如切片、通过键查找以及向量化运算等。
对一个DataFrame列执行查询操作时，会返回一个Series数据结构。
'''

cc=df["日期"] #用”日期“列生成一个Series型数据
print("Series的shape\n",cc.shape)
print("Series的index\n",cc.index)
print("Series的values\n",cc.values)
print("Series的name\n",cc.name)

'''
切片展示
'''
print("Last 2 data\n",cc[-2:])
print("except last 2 data\n",cc[:-2])

second_col=df.columns[1]
print("第二列名称",second_col)
print("第二列内容",df[second_col])









