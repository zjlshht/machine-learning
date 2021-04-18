def normalize(name):
    return name.upper()[:][0] + name.lower()[:][1:]

# 测试:
L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
print(L2)

from functools import reduce
def prod(L):
    def f(x,y):
        return x*y
    return reduce(f,L)
print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))
if prod([3, 5, 7, 9]) == 945:
    print('测试成功!')
else:
    print('测试失败!')