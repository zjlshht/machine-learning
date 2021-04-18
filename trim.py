def trim(s) :
    if s=='':
        return s
    a=s
    if s[0]==' ':
        a=s[1:]
    if a[-1]==' ':
        a=a[:-1]
    if a==s:
        return(a)
    else:
        s=trim(a)
        return(s)

