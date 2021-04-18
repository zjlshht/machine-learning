'''
利用NumPy和pandas对CSV文件进行写操作
'''
import numpy as np
import pandas as pd
np.random.seed(42)

a=np.random.randn(3,4)
a[2][2]=np.nan
print(a)
'''
NumPy得savetxt()函数是与loadtxt()相对应的一个函数，它能以诸如CSV之类的区隔型文件
格式保存数组。下面的代码可以用来保存刚创建的那个数组
'''
np.savetxt('np.csv',a,fmt='%.2f',delimiter=',',header="#1,#2,#3,#4")
'''
上面的函数调用中，我们规定了用以保存数组的文件的名称、数组、可选格式、间隔符和一个
可选标题

下面利用随机数组来创建 pandas DataFrame
'''
df=pd.DataFrame(a)
print(df) #注意到pandas会自动替我们给数据取好列名：
'''
利用pandas的to_csv()方法可以为CSV文件生成一个DataFrame，代码如下
'''
df.to_csv('pd.csv',float_format='%.2f',na_rep="NAN!")#这个方式也创建了一个csv





