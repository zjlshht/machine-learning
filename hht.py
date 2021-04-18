import numpy
def arange(x):
    b=numpy.arange(x)
    return b

def array(x):
    c=numpy.array(x)
    return c

'''
a=arange(3)
b=arange(4)
c=array([a,b])
'''

'''
bool 布尔值
int8 字节类型
int16 整型
int32 整型-2^31~2^31-1
int64 整型
uint8 无符号整型0~255
uint16 uint32 uint64 
float16 半精度浮点型
float64 or float 双精度浮点型
complex64 复数类型，由两个32位浮点数表示
complex128 复数类型 由两个64位浮点数表示
‘’‘

’‘’
字符码
整型 i
无符号整型 u
单精度浮点型 f
双精度浮点型 d
布尔型 b
复数型 D
字符串 S
万国码 U
空类型 V
example: numpy.arange(7,dtype='f')
'''
