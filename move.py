def move(n,a,b,c):#把a的n个盘子从a移到c
    if n == 1:
        print(a, '-->', c)
    else:
        move(n-1,a,c,b) #把n-1个从a移到b
        move(1,a,b,c) #把一个从a移到c
        move(n-1,b,a,c) #把n-1个从b移到c
