res=[2]
def shusu(x):
    i=2
    while(i<x):
        n=len(res)
        t=0
        while(i%res[t]):
            t+=1
            if t==n:
                res.append(i)
                i+=1
                break
        i+=1
    return