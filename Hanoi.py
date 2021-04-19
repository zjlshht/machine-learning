def move(n, a, b, c):  # 把a的n个盘子从a移到c
    """
    汉诺塔问题
    将从小到大放的n个盘子从a移到c，每次只能移动一个盘子
    并且不允许上面的盘子比下面的大
    """
    if n == 1:
        print(a, '-->', c)
    else:
        move(n-1, a, c, b)  # 把n-1个从a移到b
        move(1, a, b, c)  # 把一个从a移到c
        move(n-1, b, a, c)  # 把n-1个从b移到c
