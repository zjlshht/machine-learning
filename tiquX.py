# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:21:00 2020

@author: zjlsyhhht
"""

from pandas.io.parsers import read_csv
import numpy as np
data=read_csv("data.csv")
df=data.values
a1=df[:,0]
a2=df[:,2]
a3=df[:,3]
a4=df[:,4]
a5=df[:,5]
a6=df[:,7]
a7=df[:,8]
a8=df[:,9]
a9=df[:,10]
a10=df[:,11]
a11=df[:,13]#pupolarity
a12=df[:,15]
a13=df[:,16]
a14=df[:,17]
a15=df[:,18]#year
X=np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a12,a13,a14,a15,a11]).T
import pandas as pd
df=pd.DataFrame(X)
df.to_csv('Xs.csv',float_format='%.2f',na_rep="NAN!")
