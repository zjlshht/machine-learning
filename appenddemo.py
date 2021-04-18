def f(x):
    return x * x
    
L = []
M=[]
for n in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    L.append(f(n))
    D=L.copy()
    M.append(D)
print(M)