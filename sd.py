s="pwwkew"
import itertools
k="".join(list(set(s)))
def permutation(s: str):
    result={}
    for i in itertools.permutations(s):
        string=''.join(i)
        if string not in result:
            result[string]=None
    return list(result.keys())#把字典x打印成列表
res=permutation(k)
def judge(x):
    if x not in s:return 0
    else:
        return len(x)
t=[judge(x) for x in res]
a        