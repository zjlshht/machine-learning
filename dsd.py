class Solution:
    def add(self, a: int, b: int) -> int:
        res=0
        i=1
        tmp=0
        while i<=a or i <=b:
            if tmp:
                if a&i and b&i:
                    res^=i
                    i<<=1         
                elif a&i or b&i:
                    i<<=1
                else:
                    tmp=0
                    res^=i
                    i<<=1
            else:
                if a&i and b&i:
                    tmp=1
                    i<<=1
                elif a&i or b&i:
                    res^=i
                    i<<=1
                else:
                    i<<=1
            print(res)
        if tmp:
            res^=i
        return res