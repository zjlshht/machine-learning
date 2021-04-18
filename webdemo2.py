from urllib.request import urlopen
from bs4 import BeautifulSoup

text1=urlopen('https://mxd.web.sdo.com/web6/home/index.asp').read()
soup=BeautifulSoup(text1,'heml.parser')#运行不了 好像缺个helm.parser

jobs=set()
for job in soup.body.section('h2'):
    jobs.add('{}({})'.format(job.a.string, job.a['href']))
    
print('\n'.join(sorted(jobs,key=str.lower)))