'''
通过pandas访问数据库
    我们可以向pandas提供一个数据库连接，如前面例子中的连接或SQLAlchemy连接。
    下面先创建元组列表，以构建pandas DataFrame
'''
#import quandl #下载数据
#sunspots=quandl.get("SIDC/SUNSPOTS_A") 
import statsmodels.api as sm
from pandas.io.sql import read_sql
import sqlite3

with sqlite3.connect(":memory:") as con:#数据库留在内存
    c=con.cursor()#创建游标
    
    data_loader=sm.datasets.sunspots.load_pandas()
    df=data_loader.data
    rows=[tuple(x) for x in df.values]#创建元组列表，以构建pandas DataFrame
    
    con.execute("CREATE TABLE sunspots(year,sunactivity)")#创建数据表
    con.executemany("INSERT INTO sunspots(year,sunactivity) VALUES (?,?)", rows)
    #executemany()执行多条语句，就本例而言，是插入一些记录
    c.execute("SELECT COUNT(*) FROM sunspots")
    print(c.fetchone())
    
    print("Deleted",con.execute("DELETE FROM sunspots where sunactivity>20").rowcount,"row")
    #删除事件数大于20的记录
    
'''
如果把数据库连接至pandas，就可以利用read_sql()函数来执行查询并返回pandas DataFrame了
'''
print(read_sql("SELECT * FROM SUNSPOTS where year <1732",con))
con.execute("DROP TABLE sunspots")

c.close()
    