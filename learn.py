import numpy as np
import os
import sys
import torch
import copy
import math
import torch.nn as nn
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

x=torch.arange(30,dtype=torch.float32).view(2,3,5)
x_mean=torch.mean(x,dim=0,keepdim=True)
x_mean0=torch.mean(x,dim=-1,keepdim=True)
def knn(x, k):
    print(x)
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    print(inner)
    print(x**2)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    print(xx)
    print("xx: ",xx.transpose(1,2),"\n:inner",inner)
    print("minus:\n",xx.transpose(1,2)-inner)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    print(pairwise_distance)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    print(idx)
    print(pairwise_distance.topk(k=k,dim=-1))
    return idx
knn(x,2)
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
#