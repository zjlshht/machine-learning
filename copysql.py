import pymysql
import xlwt
def get_sel_excel(file_excel):
    #建立连接
    conn = pymysql.connect('localhost','root','zhaojing1314','stu')
 
    #建立游标
    cursor = conn.cursor()
    sel_sql = "SELECT * FROM PYTHON;"
    print("开始查询表！")
    #执行sql语句
    cursor.execute(sel_sql)
    #获取查询到结果
    res = cursor.fetchall()
    print(res)
    w_excel(res)
 
 
#操作excel
def w_excel(res):
    book = xlwt.Workbook() #新建一个excel
    sheet = book.add_sheet('STUDENTS6') #新建一个sheet页
    title = ['NUM','STU_NUM','NAME','CLASS']
    #写表头
    i = 0
    for header in title:
        sheet.write(0,i,header)
        i+=1
 
 
    #写入数据
    for row in range(1,len(res)):
        for col in range(0,len(res[row])):
            sheet.write(row,col,res[row - 1][col])#这里不写row-1则会把第一个数据丢失
        row+=1
    col+=1
    book.save('STUDENTS6')
    print("导出成功！")
 
if __name__ == "__main__":
    file_excel = r"D:\python\python名单.xls"#这条语句可以不写
    get_sel_excel(file_excel)