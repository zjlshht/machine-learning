'''
>>> sorted([36, 5, -12, 9, -21], key=abs)
[5, 9, -12, -21, 36]

key指定的函数将作用于list的每一个元素上，并根据key函数返回的结果进行排序。对比原始
的list和经过key=abs处理过的list：

list = [36, 5, -12, 9, -21]

keys = [36, 5,  12, 9,  21]
'''


'''
>>> sorted(['bob', 'about', 'Zoo', 'Credit'])
['Credit', 'Zoo', 'about', 'bob']

默认情况下，对字符串排序，是按照ASCII的大小比较的，由于'Z' < 'a'，结果，大写字母Z
会排在小写字母a的前面。现在，我们提出排序应该忽略大小写，按照字母序排序。要实现这个
算法，不必对现有代码大加改动，只要我们能用一个key函数把字符串映射为忽略大小写排序即
可。忽略大小写来比较两个字符串，实际上就是先把字符串都变成大写（或者都变成小写），
再比较。

>>> sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower)
['about', 'bob', 'Credit', 'Zoo']

如果要进行反向排序
>>> sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True)
['Zoo', 'Credit', 'bob', 'about']
'''
'''bad demo
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
def by_name(t):
    L=[]
    for i in range(len(t)):
#        k=t[i][0]
        L.append(t[i][0])
    L=sorted(L)
    
    return L
'''
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
def by_name(t):
    return t[0]
L2 = sorted(L, key=by_name)
print(L2)

def by_score(t):
    return t[1]
L1 = sorted(L, key=by_score)
print(L1)

def what(t):
    return t[-1] #2和3不可以
L3 = sorted(L, key=what)
print(L3)
#！！！这里的key是对里面每个元作用的  而不是整个数组 是对每个元作用完后再形成的数组
#by_name(L)=('Bob',75)   sorted是把函数对L的元素进行作用！