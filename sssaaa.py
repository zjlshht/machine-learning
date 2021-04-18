import numpy as np
def solon(x):
    a=np.zeros(len(x))
    for n in range(1,len(x)):
        a[n]=x[n]*x[n]
    for i in range(1,len(x)):
        for j in range(1,len(x)):
            if a[j]>a[i]:
                a[j],a[i]=a[i],a[j]
    return(a)
