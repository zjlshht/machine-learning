def binary(x):
    res=[]
    while x:
        res.append(x&1)
        x>>=1
    return res[::-1]