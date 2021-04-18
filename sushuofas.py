def jishu():
    n=1
    while True:
        n=n+2
        yield n
#奇数列       
def mod(n):
    return lambda x : x%n>0

def sushu():
    yield 2
    it=jishu()
    while True:
        n=next(it)
        yield n
        it=filter(mod(n),it) #将it中素数 n的倍数剔除

    
for n in sushu():
    if n < 100:
        print(n)
    else:
        break
'''
我们已经知道，可以直接作用于for循环的数据类型有以下几种：

一类是集合数据类型，如list、tuple、dict、set、str等；

一类是generator，包括生成器和带yield的generator function。

这些可以直接作用于for循环的对象统称为可迭代对象：Iterable。

可以使用isinstance()判断一个对象是否是Iterable对象

这是因为Python的Iterator对象表示的是一个数据流，Iterator对象可以被next()函数调用
并不断返回下一个数据，直到没有数据时抛出StopIteration错误。可以把这个数据流看做是
一个有序序列，但我们却不能提前知道序列的长度，只能不断通过next()函数实现按需计算下
一个数据，所以Iterator的计算是惰性的，只有在需要返回下一个数据时它才会计算。
'''