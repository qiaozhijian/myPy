import numpy as np
import os
import sys
import torch
import copy
import math
import torch.nn as nn

a=np.random.rand(3,4)
b=np.random.rand(4,4)
c=np.append(a,b,0)
print(c.shape)
def ch(a):
    a=np.eye(4,4).dot(a).dot(a)
    return a
print(b)
c=ch(b)
print(c)

# torch.bmm batch对应相乘
# print(y.transpose(1,2).shape) 表示把第几维和第几维进行交换
# softmax2=nn.Softmax(dim=0) 表示在第几维度进行运算
# print(y.sum(0)) 表示在第几维度进行求和,那一维度变成1
# tensor.masked_fill_(attn_mask, -np.inf) atten_mask与tensor同维，bool型，填充数字为-np.inf
# nn.Dropout 每个神经元有百分之50的几率被置为0
# torch.nn.Linear(20, 30) 线性器，包含w和bias，分别是（20,30）和30维
# nn.parameter() 将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
# print(x.eq(0))输出是否等于0的bool变量
# unsqueeze(n) squeeze(n)在第n维增加或减少一个维度
# np.squeeze(a)去除冗余维度
# key = key.view(1,2) 改变维度
# tensor.expand 扩展某个size为1的维度。如(5,1,6)扩展为(5,2,6) x(5,1,6) x.expand(-1,2,-1)
# torch.triu(input, diagonal=0, out=None) → Tensor 返回矩阵上三角部分，其余部分定义为0。
# nn.Embedding(2, 10)pytorch 自己的编码器
