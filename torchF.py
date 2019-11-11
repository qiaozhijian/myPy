#
import numpy as np
import os
import sys
import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
import time
import torch.multiprocessing as mp

def sum1Norm(data):
    dims=len(data.shape)-1
    bs=torch.sum(data,dim=dims).unsqueeze(dims)
    c=bs.expand(data.shape)
    d=torch.div(data,c)
    return d
class GetCommons(nn.Module):
    def __init__(self,inlierRate,most):
        super(GetCommons, self).__init__()
        # 默认0.8
        self.inlierRate=inlierRate
        # 默认0.95
        self.most=most

    def sum1Norm(self,data):
        dims = len(data.shape) - 1
        # 对每一行标准化，使其和为1
        return torch.div(data, torch.sum(data, dim=dims).unsqueeze(dims).expand(data.shape))

    def getInx(self,scores):
        batch_size=scores.shape[0]
        num_points=scores.shape[1]
        idx1 = np.arange(batch_size).repeat(num_points * num_points)
        idx2 = np.tile(np.arange(num_points).repeat(num_points), batch_size)
        return torch.tensor(idx1).cuda(), torch.tensor(idx2).cuda()

    def getNewScore(self,scores):
        # 增加显存
        scores_new = scores.clone()
        # 对每一行的权重进行排序，从大到小 占用100
        valueSort, indexSort = torch.sort(scores_new, dim=2, descending=True)
        # 对前n-1个数进行累加求和，结果放在第n个位置，以most（default=0.95）为界，most后面的都是权重较小的值
        mask = torch.gt(torch.cumsum(valueSort, dim=2)-valueSort,self.most)
        # 得到两个索引
        idx1,idx2=self.getInx(scores_new)
        # 把权重较小的位置直接赋值为0
        scores_new[idx1[mask.flatten()], idx2[mask.flatten()], indexSort[mask].flatten()] = 0
        # 再次归一化，然后输出
        return self.sum1Norm(scores_new),mask
    # 输入源点云，目标点云，权重矩阵
    # 输出新点源点云和对应的匹配点云，维度缩减为原来的百分之八十
    def forward(self, src,tgt,scores):
        # 0
        batch_size=scores.shape[0]
        num_points=scores.shape[1]
        # 公共点个数
        inlierNum = int(self.inlierRate * num_points)
        # 初始化得到的新点云及其匹配点
        # 22％ 且会增大显存
        srcNew = torch.zeros(batch_size, 3, inlierNum).cuda()
        srCorrNew = torch.zeros(batch_size, 3, inlierNum).cuda()

        # 得到新的权重矩阵
        scores_new,mask = self.getNewScore(scores)
        # 生成匹配点
        scorr = torch.matmul(tgt, scores_new.transpose(2, 1).contiguous())
        # 计算每一行的相关性得分，越小越好，说明可以用更少的点得到0.95的权重
        srcInlierMap = mask.sum(dim=2)
        # 排序
        v, id = torch.topk(srcInlierMap, dim=1, largest=False, k=inlierNum)
        # 取前inlierNum个源点云和对应匹配点
        for batch in range(batch_size):
            srcNew[batch, :, :] = src[batch, :, id[batch]]
            srCorrNew[batch, :, :] = scorr[batch, :, id[batch]]
        return srcNew, srCorrNew
if 1:
    batch=2
    points=4
    a=torch.rand(batch,3,points).cuda()
    b=torch.rand(batch,points,points).cuda()
    fisrtk=GetCommons(0.8,0.6)
    fisrtk(a,a,sum1Norm(b))

if 0:
    a=torch.from_numpy(np.arange(1,25)).reshape(2,3,4)
    print(a)
    # b=np.zeros((3,2,2),dtype=float)

    b=np.zeros_like(a)
    b[0,1,0]=1
    b[1,2,1]=2
    b[1,1,2]=3
    b[0,2,3]=0
    idx1,idx2=getInx(b)
    # print('b shape: ',b.shape)
    x=a.size(0)
    y=a.size(1)
    z=a.size(2)
    idx1=np.arange(x).repeat(y*z)
    idx2=np.tile(np.arange(y).repeat(z),x)
    a3=b.flatten()
    print(a)
    # print(a)
    # print(a.flatten())
def getInx(data):
    x = data.size(0)
    y = data.size(1)
    z = data.size(2)
    idx1 = np.arange(x).repeat(y * z)
    idx2 = np.tile(np.arange(y).repeat(z), x)
    return idx1,idx2
def throw1(data,cumsum,idx):
    data_cp=data.clone()
    mask = torch.gt(cumsum, 0.9)
    idx1, idx2 = getInx(cumsum)
    idx1 = torch.tensor(idx1)
    idx2 = torch.tensor(idx2)
    idx1_mask = idx1[mask.flatten()]
    idx2_mask = idx2[mask.flatten()]
    newId = idx[mask].flatten()
    data_cp[idx1_mask, idx2_mask, newId] = 0
    return data_cp

def throw2(data,cumsum,idx):
    data_cp=data.clone()
    newId = torch.where(cumsum < 0.6, torch.full_like(idx, 0), idx)
    for batch in range(data_cp.shape[0]):
        for point in range(data_cp.shape[1]):
            temp = data_cp[batch, point, 0]
            data_cp[batch, point, newId[batch][point]] = 0
            data_cp[batch, point, 0] = temp
    return data_cp
if 0:
    a=torch.rand(2,3,6)
    b=torch.rand(2,6,6)
    d=sum1Norm(b)
    valueSort, indexSort = torch.sort(d, dim=2, descending=True)
    valueSortCumsum=torch.cumsum(valueSort, dim=2)-valueSort #计算前n-1个数的相加
    mask = torch.gt(valueSortCumsum, 0.9)
    d1=throw1(d,valueSortCumsum,indexSort)
    d2=throw2(d,valueSortCumsum,indexSort)
    d=sum1Norm(d1)
    scorr=torch.matmul(a,d.transpose(2,1).contiguous())
    srcInlierMap=mask.sum(dim=2)
    v,id=torch.topk(srcInlierMap,dim=1,largest=False,k=3)
    new_scr=torch.zeros(2,3,3)
    new_scorr=torch.zeros(2,3,3)
    for batch in range(2):
        new_scr[batch,:,:]=a[batch,:,id[batch].squeeze()]
        new_scorr[batch,:,:]=scorr[batch,:,id[batch].squeeze()]

def train(data):
    data=torch.rand(1000,1000).cuda()
    for i in range(10):
        torch.matmul(data,data)
if 0:
    num_processes = 4
    # NOTE: this is required for the ``fork`` method to work
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(0,))
        p.start()
        processes.append(p)
    t1=time.time()
    for p in processes:
      p.join()
    print(time.time()-t1)

    t1=time.time()
    for i in range(num_processes):
        train(0)
    print(time.time()-t1)
def topMost(value, index, most=0.95):
    value=torch.div(value,value.sum())
    value,index2=torch.sort(value,dim=0,descending=True)
    index=index[index2].squeeze(-1)

    num_0=0
    num_1=value.shape[0]
    num=num_1
    while num_1-num_0>1:
        if value[0:num].sum() > most:
            if value[0:num-1].sum() < most :
                break
            else:
                num_1=num
                num=(num_0+num_1)//2
        else:
            num_0=num
            num=num+(num_1-num_0)//2
    print(num,num_0,num_1)
    value=value[0:num]
    index=index[0:num]

    return value,index
if 0:
    value=torch.tensor([0.98,0.01,0.01,0.001,0.001,0.001,0.001,0.001])
    index=torch.tensor([0.98,0.01,0.01,0.001,0.001,0.001,0.001,0.001])
    data=topMost(value,index)
    print(55%3)
if 0:
    output = torch.tensor([[-5.4783, 0.2298],
                           [-4.2573, -0.4794],
                           [-0.1070, -5.1511],
                           [-0.1785, -4.3339]]).reshape(2,2,2)
    print(output.size())
    index,pred = torch.topk(output,k=1,dim=0)
    print(pred.shape)
    print(index)
    print(pred)
# 求tensor距离
if 0:
    a=torch.tensor([2.,2.,3.]).reshape(1,3)
    b=torch.tensor([1.,4.,3.]).reshape(1,3)
    rpe = F.pairwise_distance(a, b, p=2)
if 0:
    # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
    X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                      cluster_std=[0.2, 0.1, 0.2, 0.2],
                      random_state=9)
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
    pca = PCA(n_components=3)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    X_new = pca.transform(X)
    print("x_new:\n", X_new.shape)
    # plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
    # plt.show()
    pca = PCA(n_components=0.95)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    print(pca.n_components_)
    pca = PCA(n_components=0.99)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    print(pca.n_components_)
    pca = PCA(n_components='mle')
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    print(pca.n_components_)

# tensor.masked_fill_(attn_mask, -np.inf) atten_mask与tensor同维，bool型，填充数字为-np.inf
# torch.bmm batch对应相乘
# torch.nn.Linear(20, 30) 线性器，包含w和bias，分别是（20,30）和30维
# softmax2=nn.Softmax(dim=0) 表示在第几维度进行运算
# nn.Dropout 每个神经元有百分之50的几率被置为0
# nn.parameter() 将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
# tensor.expand 扩展某个size为1的维度。如(5,1,6)扩展为(5,2,6) x(5,1,6) x.expand(-1,2,-1)
# torch.triu(input, diagonal=0, out=None) → Tensor 返回矩阵上三角部分，其余部分定义为0。
# nn.Embedding(2, 10)pytorch 自己的编码器