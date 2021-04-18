# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:55:43 2020

@author: zjlsyhhht
"""

import torch 
torch.is_tensor()#判断
torch.tensor#生成
torch.as_tensor()#转化
torch.from_numpy()#生成由数组转化的张量
torch.zeros()#生成元素为0的张量
torch.zeros_like()#生成与输入张量大小一致的元素为0的张量
torch.ones()
torch.ones_like()
torch.arange(3,7)
torch.linspace(-1, 1,10)
d=torch.logspace(-1, 1,5)#log(d)是等差数列
torch.eye(3)
torch.empty(4,5)#类似zeros
torch.empty_like()
torch.full(size=[3,4],fill_value=5)
torch.rand(4,5)#0-1均匀分布
torch.rand_like()
torch.randn(4,5)#标准正态分布
torch.randn_like()

