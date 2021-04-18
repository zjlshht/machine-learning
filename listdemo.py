L = []
for x in range(1, 11):
    L.append(x * x)

#   >> 除以2后向下取整
 
#L=[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

L2=[x * x for x in range(1, 11)]
#L2=[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

L3=[x * x for x in range(1, 11) if x % 2 == 0]
#L3=[4, 16, 36, 64, 100]

L4=[m + n for m in 'ABC' for n in 'XYZ']
#L4=['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']


d = {'x': 'A', 'y': 'B', 'z': 'C' }
for k, v in d.items():
    print(k, '=', v)
'''
y = B
x = A
z = C
'''
L5 = ['Hello', 'World', 'IBM', 'Apple']
L5plus=[s.lower() for s in L5]
#L5plus=['hello', 'world', 'ibm', 'apple'

L6=[x if x % 2 == 0 else -x for x in range(1, 11)]
#L6=[-1, 2, -3, 4, -5, 6, -7, 8, -9, 10]

L1 = ['Hello', 'World', 18, 'Apple', None]
L2 = [x.lower() for x in L1 if isinstance(x,str)]
#L2 = ['hello', 'world', 'apple']

def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        a, b = b, a + b
        n = n + 1
    return 'done'

'''改写为generator
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'

for n in fib(6):
    print(n)
'''
'''小迭代器'''
def odd():
    print('step 1')
    yield 1
    print('step 2')
    yield(3)
    print('step 3')
    yield(5)

''' map类似于R中的sapply
def f(x):
...     return x * x
...
>>> r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]

>>> list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
['1', '2', '3', '4', '5', '6', '7', '8', '9']
'''
'''
reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)

>>> from functools import reduce
>>> def add(x, y):
...     return x + y
...
>>> reduce(add, [1, 3, 5, 7, 9])
25
'''
from functools import reduce
DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

def str2int(s):
    def fn(x, y):
        return x * 10 + y
    def char2num(s):
        return DIGITS[s]
    return reduce(fn, map(char2num, s))
'''进一步简化：
from functools import reduce

DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

def char2num(s):
    return DIGITS[s]

def str2int(s):
    return reduce(lambda x, y: x * 10 + y, map(char2num, s))
'''

'''和map()不同的是，filter()把传入的函数依次作用于每个元素，然后根据返回值是True
还是False决定保留还是丢弃该元素。'''
def is_odd(n):
    return n % 2 == 1

list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
# 结果: [1, 5, 9, 15]
list(map(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
#Out: [True, False, False, True, False, True, False, True] 
#filter 可以让真值返回

def not_empty(s):
    return s and s.strip()

list(filter(not_empty, ['A ', '', 'B', None, 'C', '  ']))
# 结果: ['A ', 'B', 'C']
list(map(not_empty, ['A ', '', 'B', None, 'C', '  ']))
#Out: ['A', '', 'B', None, 'C', '']
#真值返回

a='sad23d'
b=a[::-1]#反向输出
'''
#练习：判断是否回文数

def is_palindrome(n):

    x=n 

    y=0

    while x>0:

        y=y*10+x%10

        x=x//10

    return y==n
'''

#由66-85
''' map类似于R中的sapply
def f(x):
...     return x * x
...
>>> r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> list(r)
[1, 4, 9, 16, 25, 36, 49, 64, 81]

当我们在传入函数时，有些时候，不需要显式地定义函数，直接传入匿名函数更方便。
在Python中，对匿名函数提供了有限支持。还是以map()函数为例，计算f(x)=x2时，
除了定义一个f(x)的函数外，还可以直接传入匿名函数：

>>> list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
[1, 4, 9, 16, 25, 36, 49, 64, 81]
###用匿名函数lambda x:x *x 代替f
关键字lambda表示匿名函数，冒号前面的x表示函数参数。

匿名函数有个限制，就是只能有一个表达式，不用写return，返回值就是该表达式的结果。
'''

#用匿名函数有个好处，因为函数没有名字，不必担心函数名冲突。此外，匿名函数也是一个函
#数对象，也可以把匿名函数赋值给一个变量，再利用变量来调用该函数：
f = lambda x: x * x
print(f)
#<function <lambda> at 0x101c6ef28>
print(f(5))
#25

def build(x, y):
    return lambda: x * x + y * y
f=build(3,5)
f()==34

def build2(x, y):
    def g(x,y):
        return x*x+y*y
    return g
t=build2(3, 2)
t(3,5)==34

def build3():
    def g(x,y):
        return x*x+y*y
    return g
t=build3()
t(3,5)==34

L = list(filter(lambda x:x%2==1, range(1, 20)))

#如果要获得一个对象的所有属性和方法，可以使用dir()函数，它返回一个包含字符串的list，
#比如，获得一个str对象的所有属性和方法：


class MyObject(object):
     def __init__(self):
         self.x = 9
     def power(self):
         return self.x * self.x
     
'''hasattr(obj, 'x') # 有属性'x'吗？
True
>>> obj.x
9
>>> hasattr(obj, 'y') # 有属性'y'吗？
False
>>> setattr(obj, 'y', 19) # 设置一个属性'y'
>>> hasattr(obj, 'y') # 有属性'y'吗？
True
>>> getattr(obj, 'y') # 获取属性'y'
19
>>> obj.y # 获取属性'y'
19
可以传入一个default参数，如果属性不存在，就返回默认值：

>>> getattr(obj, 'z', 404) # 获取属性'z'，如果不存在，返回默认值404
404
也可以获得对象的方法：

>>> hasattr(obj, 'power') # 有属性'power'吗？
True
>>> getattr(obj, 'power') # 获取属性'power'
<bound method MyObject.power of <__main__.MyObject object at 0x10077a6a0>>
>>> fn = getattr(obj, 'power') # 获取属性'power'并赋值到变量fn
>>> fn # fn指向obj.power
<bound method MyObject.power of <__main__.MyObject object at 0x10077a6a0>>
>>> fn() # 调用fn()与调用obj.power()是一样的
81'''

class Student(object):
    def __init__(self, name):
        self.name = name

s = Student('Bob')
s.score = 90
'''s.name
Out[17]: 'Bob'  '''

'''>>> class Student(object):
...     name = 'Student'
...
>>> s = Student() # 创建实例s
>>> print(s.name) # 打印name属性，因为实例并没有name属性，所以会继续查找class的name属性
Student
>>> print(Student.name) # 打印类的name属性
Student
>>> s.name = 'Michael' # 给实例绑定name属性
>>> print(s.name) # 由于实例属性优先级比类属性高，因此，它会屏蔽掉类的name属性
Michael
>>> print(Student.name) # 但是类属性并未消失，用Student.name仍然可以访问
Student
>>> del s.name # 如果删除实例的name属性
>>> print(s.name) # 再次调用s.name，由于实例的name属性没有找到，类的name属性就显示出来了
Student
'''
#从上面的例子可以看出，在编写程序的时候，千万不要对实例属性和类属性使用相同的名字，
#因为相同名称的实例属性将屏蔽掉类属性，但是当你删除实例属性后，再使用相同的名称，访
#问到的将是类属性。

