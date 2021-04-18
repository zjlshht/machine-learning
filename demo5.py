def check_index(key):
    ##键必须是非负整数，如果不是整数，将引发TypeError异常；如果是负数，将引发IndexError异常
    if not isinstance(key, int):raise TypeError
    if key<0:raise IndexError

class ArithmeticSequence:
    def __init__(self,start=0,step=1):
        #初始化这个算术序列
        #start#序列中的第一个值
        #step#两个相邻值得差
        #changed#一个字典，包括用户修改后得值
        self.start=start#存储起始值
        self.step=step#存储步长值
        self.changed={}#没有任何元素被修改
    def __getitem__(self,key):
            #从算术序列中获取一个元素
            #上述是返回与指定键相关联得值
            check_index(key)
            try:return self.changed[key]#修改过？
            except KeyError:
                return self.start+key*self.step
    def __setitem__(self,key,value):
        #修改算术序列中的元素
        #以键相关联的方式去存储值
         check_index(key)
         self.changed[key]=value
        
        ##代码实现一个算术序列 P151
            
        