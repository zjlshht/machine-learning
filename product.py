def product(*x):
    a=len(x)
    b=1
    for i in range(a):
        b=b*x[i]
    return(b)
