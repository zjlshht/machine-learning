num=98368
num=list(map(int,list(str(num))))    
n=len(num)
def find(i):
    a=num[i]
    tmp=a
    k=i
    for j in range(i,n):
        if num[j]>=tmp:
            tmp=num[j]
            k=j
    if tmp==a:return i
    return k
for i in range(n):
    if find(i)!=i:
        t=find(i)
        num[i],num[t]=num[t],num[i]
        break
nums=list(map(str,num))
res="".join(nums)
