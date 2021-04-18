'''
基于sqlite3的轻量级访问：
    Sqlite是一款非常流行的关系型数据可。sqlite3是python标准发行版自带的一个模块，
    可以用于处理sqlite数据库。
'''
import sqlite3
'''
首先连接数据库。如果希望把数据库存到文件中，那么必须提供一个文件名。否则可以通过以
下方式将数据库留在内存中：
'''
with sqlite3.connect(":memory:") as con:
    '''
上面使用了python的with语句，需要注意的是，这个语句依赖于特定上下文管理器类的__exit__()
方法的存在。如果我们使用了这个语句，就无需显式关闭数据库连接了。（上下文管理器会自动
关闭连接）连接到数据库后，还需要一个游标。游标在数据库中的作用，至少从概念上来讲，
类似于文本编辑器中的光标。注意，这个游标也需要由我们来关闭
    '''
c=con.cursor()#创建游标
'''
现在可以直接创建数据表了。为了创建数据库表，我们需要向游标传递一个SQL字符串，具体
如下所示：
'''
c.execute('''CREATE TABLE sensors
          (data text, city text, code text, sensor_id real, temperature real)''')
'''
上面的代码会创建一个包含很多列的数据表，具体名称为sensors。下面列出SQLite数据表：
'''
for table in c.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print("Table",table[0])

'''
现在我们要插入并查询一些随机数据
'''
c.execute("INSERT INTO sensors VALUES ('2016-11-05','Utrecht','Red',42,15.14)")
c.execute("SELECT * FROM sensors")
print(c.fetchone())
'''
当不需要某个数据表时，就可以将其删除了。需要注意的是，删除是一项非常危险的操作，因此
必须绝对肯定再也用不到这个数据表了。数据一旦被删除，就无法恢复。
下面的代码将删除数据表，并显示删除操作执行后所剩数据表的数量：
'''
con.executemany("DROP TABLE sensors")
print("of tables",c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type ='table'").fetchone()[0])

c.close()
