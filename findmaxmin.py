def findminmax(L):
    if L==[]:
        return (None, None)
    else:
        min=max=L[0]
        for i in L:
            if  min>i:
                min=i
            elif  max<i:
                max=i
        return(min,max)