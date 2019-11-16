import numpy as np
import itertools
import os
from plotpcl import lala2
def velodyne2camera(gt_v):
    t = np.asarray( [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
                    -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
                    9.999738645903e-01,4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
                    0,                  0,                  0,                   1                  ]).reshape(4,4)
    gt_c=t.dot(gt_v)
    gt_c = gt_c.dot(np.linalg.inv(t))
    return gt_c
def lala():
    print(np.random.rand(1,4))
np.random.seed(1234)
if 1:
    a=[1,2,3]
    a=np.random.rand(3,1)
    print(a.mean())
if 0:
    f=open('a.txt','a')
    for i in range(1000000):
        f = open('a.txt', 'a')
        f.write(str(i)+'\n')
        f.flush()

# print(y.transpose(1,2).shape) 表示把第几维和第几维进行交换
# print(y.sum(0)) 表示在第几维度进行求和,那一维度变成1
# print(x.eq(0))输出是否等于0的bool变量
# unsqueeze(n) squeeze(n)在第n维增加或减少一个维度
# np.squeeze(a)去除冗余维度
# key = key.view(1,2) 改变维度
