import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile

np.random.seed(42)
a=np.random.randn(365,4)

tmpf=NamedTemporaryFile(suffix='.xlsx')
df=pd.DataFrame(a)
print(tmpf.name)
df.to_excel(tmpf.name,sheet_name='Random Data')#不能运行
print("Mean\n",pd.read_excel(tmpf.name,'Random Data').mean())#这个是重建一个DataFrame的
#df.to_excel(tmpf.name,sheet_name='Random Data')  这个是创建一个excel文件得


