class Student(object):

    def __init__(self, name, score):
        self.name = name
        self.score = score
        
    def print_score(self):
        print('%s: %s' % (self.name, self.score))
        
    def get_grade(self):
        if self.score >= 90:
            return 'A'
        elif self.score >= 60:
            return 'B'
        else:
            return 'C'
''' 注意：特殊方法“__init__”前后分别有两个下划线！！！
注意到__init__方法的第一个参数永远是self，表示创建的实例本身，因此，在__init__方法
内部，就可以把各种属性绑定到self，因为self就指向创建的实例本身。
有了__init__方法，在创建实例的时候，就不能传入空的参数了，必须传入与__init__方法匹
配的参数，但self不需要传，Python解释器自己会把实例变量传进去：
>>> bart = Student('Bart Simpson', 59)
>>> bart.name
'Bart Simpson'
>>> bart.score
59

>>> bart.print_score()
Bart Simpson: 59

>>>lisa = Student('Lisa', 99)
>>>bart = Student('Bart', 59)
>>>print(lisa.name, lisa.get_grade())
>>>print(bart.name, bart.get_grade())
Lisa A
Bart C

'''

#和普通的函数相比，在类中定义的函数只有一点不同，就是第一个参数永远是实例变量self，
#并且，调用时，不用传递该参数。除此之外，类的方法和普通函数没有什么区别，所以，你仍
#然可以用默认参数、可变参数、关键字参数和命名关键字参数。