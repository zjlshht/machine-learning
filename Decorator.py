#由于函数也是一个对象，而且函数对象可以被赋值给变量，所以，通过变量也能调用该函数。

def now():
    print('2015-3-25')

'''
>>> f = now
>>> f()
2015-3-25
函数对象有一个__name__属性，可以拿到函数的名字：

>>> now.__name__
'now'
>>> f.__name__
'now'
'''

#现在，假设我们要增强now()函数的功能，比如，在函数调用前后自动打印日志，但又不希望
#修改now()函数的定义，这种在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator）。

#本质上，decorator就是一个返回函数的高阶函数。所以，我们要定义一个能打印日志的decorator，
#可以定义如下：
def log(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__) #打印名字
        return func(*args, **kw)            #返回函数内容
    return wrapper
#观察上面的log，因为它是一个decorator，所以接受一个函数作为参数，并返回一个函数
now=log(now)
print(now())
#由于log()是一个decorator，返回一个函数，所以，原来的now()函数仍然存在，只是现在同
#名的now变量指向了新的函数，于是调用now()将执行新函数，即在log()函数中返回的wrapper()函数。
@log #相当于执行了now=log(now)
def now2():
    print('2015-3-25')
print(now2())

#如果decorator本身需要传入参数，那就需要编写一个返回decorator的高阶函数，写出来会
#更复杂。比如，要自定义log1的文本：
def log1(text):
    def decorator(func):
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator
#这个3层嵌套的decorator用法如下：
@log1('execute')
def now3():
    print('2015-3-25')
    
#>>now3()
'''
execute now():
2015-3-25

和两层嵌套的decorator相比，3层嵌套的效果是这样的：
>>> now3 = log1('execute')(now3)

>>>now3.__name__
Out:'wrapper'
'''
#因为返回的那个wrapper()函数名字就是'wrapper'，所以，需要把原始函数的__name__等属
#性复制到wrapper()函数中，否则，有些依赖函数签名的代码执行就会出错。
#不需要编写wrapper.__name__ = func.__name__这样的代码，Python内置的functools.wraps
#就是干这个事的，所以，一个完整的decorator的写法如下：
import functools

def log2(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper
'''
#>>now=log2(now)
>>now()
Out:
call now():
2015-3-25

now.__name__
Out: 'now'
'''

def log3(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator
@log3('execute')
def now4():
    print('2015-3-25')
'''
now4()
execute now4():
2015-3-25

now4.__name__
Out[65]: 'now4'
'''


