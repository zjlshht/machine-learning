def triangles():
    a=[]
    while True:
        a=a+[1]
        yield a
        #a=a+[1]
        if len(a)>1:
            for i in range(len(a)-1):
                a[-i-1]=a[-i-2]+a[-i-1]
    return 'done'

n = 0
results = []
for t in triangles():
    b=t.copy()
    results.append(b)
    n = n + 1
    if n == 10:
        break
m=0
for t in triangles():
    print(t)
    m=m+1
    if m==10:
        break

#for t in results:
#    print(t)

if results == [
    [1],
    [1, 1],
    [1, 2, 1],
    [1, 3, 3, 1],
    [1, 4, 6, 4, 1],
    [1, 5, 10, 10, 5, 1],
    [1, 6, 15, 20, 15, 6, 1],
    [1, 7, 21, 35, 35, 21, 7, 1],
    [1, 8, 28, 56, 70, 56, 28, 8, 1],
    [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
]:
    print('测试通过!')
else:
    print('测试失败!')