from functools import reduce
DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
def str2float(x):
    b=x.split('.')
    b1=b[0]
    b2=b[1]
    n=len(b2)
    def char2num(s):
        return DIGITS[s]
    b11=map(char2num,b1)
    b21=map(char2num,b2)
    def f(x,y):
        return 10*x+y
    return reduce(f,b11)+reduce(f,b21)/10**n

print('str2float(\'123.456\') =', str2float('123.456'))
if abs(str2float('123.456') - 123.456) < 0.00001:
    print('测试成功!')
else:
    print('测试失败!')