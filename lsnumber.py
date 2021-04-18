# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:13:13 2020

@author: zjlsyhhht
"""

class Solution:
    def romanToInt(self, s: str) -> int:
        i=s.count('I')
        iv=s.count('IV')
        ix=s.count('IX')
        v=s.count('V')
        x=s.count('X')
        xl=s.count('XL')
        xc=s.count('XC')
        l=s.count('L')
        c=s.count('C')
        cd=s.count('CD')
        cm=s.count('CM')
        d=s.count('D')
        m=s.count('M')
        number=1000*(m-cm)+500*(d-cd)+100*(c-cd-cm-xc)+50*(l-xl)+10*(x-xl-xc-ix)+5*(v-iv)+(i-iv-ix)+4*iv+9*ix+40*xl+90*xc+400*cd+900*cm
        return number

t=Solution
print(t.romanToInt(t,'CMXIV'))        

class Solution2:
    def romanToInt(self, s):
        a = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}        
        ans=0        
        for i in range(len(s)):            
            if i<len(s)-1 and a[s[i]]<a[s[i+1]]:                
                ans-=a[s[i]]
            else:
                ans+=a[s[i]]
        return ans
    
tt=Solution2
print(tt.romanToInt(tt,'CMXIV'))  

class Solution:
    def longestCommonPrefix(self, strs) -> str:
        if not strs: return ""
        s1 = min(strs)
        s2 = max(strs)
        for i,x in enumerate(s1): #用好enumerate！！！！
            if x != s2[i]:
                return s2[:i]
        return s1

class Solution:
    def isValid(self, s):
        while '{}' in s or '()' in s or '[]' in s:
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''
    
class Solution:
    def isValid(self, s: str) -> bool:
        dic = {'{': '}',  '[': ']', '(': ')', '?': '?'}
        stack = ['?']
        for c in s:
            if c in dic: stack.append(c)
            elif dic[stack.pop()] != c: return False  #pop删除右端一个元素
        return len(stack) == 1
#栈的思想
class Solution:
    def finalPrices(self, prices):
        n = len(prices)#数据长度
        stack, res = [], prices.copy()#存index的空栈和拷贝
        for i in range(n):
            while stack and prices[stack[-1]] >= prices[i]:#栈非空且栈对应元满足出栈条件
                res[stack[-1]] -= prices[i] #出栈数算折扣
                stack.pop() #出栈
            stack.append(i) #进栈
        return res
class Solution:
    def isPathCrossing(self, path: str) -> bool:
        dirs = {
            "N": (-1, 0),
            "S": (1, 0),
            "W": (0, -1),
            "E": (0, 1),
        }
        #其实里面是中括号也可以 
        x, y = 0, 0
        vis = set([(x, y)])
        for ch in path:
            dx, dy = dirs[ch]
            x, y = x + dx, y + dy
            if (x, y) in vis:
                return True
            vis.add((x, y))

        return False

class Solution:
    def interpret(self, command: str) -> str:
        return command.replace('()','o').replace('(al)','al')

class Solution:
    def maxRepeating(self, sequence: str, word: str) -> int:
        def qutou(x):
            k=list(x)
            k.pop(0)
            d=""
            for i in k:
                d=d+i
            return d
        def maxmin(x):
            trans=x.replace(word,'0')
            n=trans.count('0')
            for m in reversed(range(n+1)):
                k='0'*m
                if trans.count(k):
                    break
            return [n,m] 
        [n,m]=maxmin(sequence)
        for i in range(len(sequence)):
            sequence=qutou(sequence)
            [a,b]=maxmin(sequence)
            if b>m:
                m=b
            if a<m:
                break
        return m


        return t
            
class Solution:
    def maxRepeating(self, sequence: str, word: str) -> int:
        L, R = 0, len(sequence) // len(word)
        while L < R:
            mid = L + (R - L + 1) // 2
            if word * mid in sequence:
                L = mid
            else:
                R = mid - 1
        return L
'''
def sums(x,y,t):
    t=y
    y=x*y+t
    return [y,t]
t=1
for i in range(len(cont)-1):
    [cont[-(i+2)],t]=sums(cont[-(i+2)],cont[-(i+1)],t)
    a=cont[0]
    b=t
    k=a%b
while(k!=0):
    a=b
    b=k
    k=a%b
n=cont[0]/b
m=t/b
'''
class Solution:
    def countSubTrees(self, n: int, edges, labels: str):
        def countt(x,y):
            t=0
            for j in edges:
                if x==j[0]:
                    t=t+countt(j[1],y)
            if labels[x]==y:
                t=t+1
            return t
        k=[]
        for i in range(n):
            k.append(countt(i,labels[i]))
        return k

class Solution:
    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
        a=[0]
        while(len(a)<n):
            for i in range(n-1):
                if edges[i][0] in a and edges[i][1] not in a:
                    a.append(edges[i][1])
                if edges[i][0] not in a and edges[i][1] in a:
                    t=edges[i][0]
                    edges[i][0]=edges[i][1]
                    edges[i][1]=t
                    a.append(edges[i][1])    
        def countt(x,y):
            t=0
            for j in edges:
                if x==j[0]:
                    t=t+countt(j[1],y)
            if labels[x]==y:
                t=t+1
            return t
        k=[]
        for i in range(n):
            k.append(countt(i,labels[i]))
        return k

                
class Solution:
    def restoreString(self, s: str, indices: List[int]) -> str:
        length = len(s)
        result = [""] * length
        for i, ch in enumerate(s):
            result[indices[i]] = ch
        return "".join(result)

class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        word2ch = dict()
        ch2word = dict()
        words = s.split()
        if len(pattern) != len(words):
            return False
        
        for ch, word in zip(pattern, words):
            if (word in word2ch and word2ch[word] != ch) or (ch in ch2word and ch2word[ch] != word):
                return False
            word2ch[word] = ch
            ch2word[ch] = word
    
        return True
加字典的好办法



              
