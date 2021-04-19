import numpy


def junyun(p):
    """
    Description
    -------
    用一个伯努利分布来模拟一个均匀分布
    """
    left, right = 0, 1
    while right-left > 1e-5:
        a = numpy.random.binomial(1, p, 1)
        if a:
            left += (right-left) * (1-p)
        else:
            right -= (right-left)*p
    return left


k = 0
for i in range(1000):
    k += junyun(0.3)
print(k/1000)
