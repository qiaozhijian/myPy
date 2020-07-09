import torch
import torch.nn as nn
import torch.nn.functional as F

cat_or_stack = True  # true表示cat
class LPDNet(nn.Module):
    def __init__(self,emb_dims=512, use_mFea=False,t3d=True,tfea=False,use_relu = False):
        super(LPDNet, self).__init__()
        self.negative_slope = 1e-2
        if use_relu:
            self.act_f = nn.ReLU(inplace=True)
        else:
            self.act_f = nn.LeakyReLU(negative_slope=self.negative_slope,inplace=True)
        self.use_mFea= use_mFea
        self.k = 20
        self.t3d=t3d
        self.tfea=tfea
        self.emb_dims=emb_dims
        if self.t3d:
            self.t_net3d = TranformNet(3)
        if self.tfea:
            self.t_net_fea = TranformNet(64)
        self.useBN = True
        if self.useBN:
            # [b,6,num,20] 输入 # 激活函数换成Leaky ReLU? 因为加了BN，所以bias可以舍弃
            if cat_or_stack:
                self.convDG1 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),nn.BatchNorm2d(128),self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),nn.BatchNorm2d(128),self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),nn.BatchNorm2d(256),self.act_f)
            else:
                self.convDG1 = nn.Sequential(nn.Conv2d(64*1, 128, kernel_size=1, bias=False),nn.BatchNorm2d(128),self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),nn.BatchNorm2d(128),self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128*1, 256, kernel_size=1, bias=False),nn.BatchNorm2d(256),self.act_f)
            # 在一维上进行卷积，临近也是左右概念，类似的，二维卷积，临近有上下左右的概念
            if self.use_mFea:
                self.conv1_lpd = nn.Conv1d(8, 64, kernel_size=1, bias=False)
            else:
                self.conv1_lpd = nn.Conv1d(3, 64, kernel_size=1, bias=False)
            self.conv2_lpd = nn.Conv1d(64, 64, kernel_size=1, bias=False)
            self.conv3_lpd = nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False)
            # 在relu之前进行batchNorm避免梯度消失，同时使分布不一直在变化
            self.bn1_lpd = nn.BatchNorm1d(64)
            self.bn2_lpd = nn.BatchNorm1d(64)
            self.bn3_lpd = nn.BatchNorm1d(self.emb_dims)
        else :
            # [b,6,num,20] 输入 # 激活函数换成Leaky ReLU? 因为加了BN，所以bias可以舍弃
            if cat_or_stack:
                self.convDG1 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=True),self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=True),self.act_f)
            else:
                self.convDG1 = nn.Sequential(nn.Conv2d(64*1, 128, kernel_size=1, bias=True),self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128*1, 256, kernel_size=1, bias=True),self.act_f)
            if self.use_mFea:
                self.conv1_lpd = nn.Conv1d(8, 64, kernel_size=1, bias=True)
            else:
                self.conv1_lpd = nn.Conv1d(3, 64, kernel_size=1, bias=True)
            self.conv2_lpd = nn.Conv1d(64, 64, kernel_size=1, bias=True)
            self.conv3_lpd = nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=True)

    # input x: # [B,1,num,num_dims]
    # output x: # [b,emb_dims,num,1]
    def forward(self, x):
        x=torch.squeeze(x,dim=1).transpose(2, 1) # [B,num_dims,num]
        batch_size, num_dims, num_points = x.size()
        # 单独对坐标进行T-Net旋转
        if num_dims > 3 or self.use_mFea:
            x, feature = x.transpose(2, 1).split([3,5], dim=2)  # [B,num,3]  [B,num,num_dims-3]
            xInit3d = x.transpose(2, 1)
            # 是否进行3D坐标旋转
            if self.t3d:
                trans = self.t_net3d(x.transpose(2, 1))
                x = torch.bmm(x, trans)
                x = torch.cat([x, feature], dim=2).transpose(2, 1) # [B,num_dims,num]
            else:
                x = torch.cat([x, feature], dim=2).transpose(2, 1)  # [B,num_dims,num]
        else:
            xInit3d = x
            if self.t3d:
                trans = self.t_net3d(x)
                x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        if self.useBN:
            x = self.act_f(self.bn1_lpd(self.conv1_lpd(x)))
            x = self.act_f(self.bn2_lpd(self.conv2_lpd(x)))
        else:
            x = self.act_f(self.conv1_lpd(x))
            x = self.act_f(self.conv2_lpd(x))

        if self.tfea:
            trans_feat = self.t_net_fea(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        # Serial structure
        # Danymic Graph cnn for feature space
        if cat_or_stack:
            x = get_graph_feature(x, k=self.k) # [b,64*2,num,20]
        else:
            x = get_graph_feature(x, k=self.k) # [B, num_dims, num, k+1]
        x = self.convDG1(x) # [b,128,num,20]
        x1 = x.max(dim=-1, keepdim=True)[0] # [b,128,num,1]
        x = self.convDG2(x) # [b,128,num,20]
        x2 = x.max(dim=-1, keepdim=True)[0] # [b,128,num,1]

        # Spatial Neighborhood fusion for cartesian space
        idx = knn(xInit3d, k=self.k)
        x = get_graph_feature(x2, idx=idx, k=self.k) # [b,128*2,num,20]
        x = self.convSN1(x) # [b,256,num,20]
        x3 = x.max(dim=-1, keepdim=True)[0] # [b,256,num,1]

        x = torch.cat((x1, x2, x3), dim=1).squeeze(-1) # [b,512,num]
        if self.useBN:
            x = self.act_f(self.bn3_lpd(self.conv3_lpd(x))).view(batch_size, -1, num_points) # [b,emb_dims,num]
        else:
            x = self.act_f(self.conv3_lpd(x)).view(batch_size, -1, num_points) # [b,emb_dims,num]
        # [b,emb_dims,num]
        x = x.unsqueeze(-1)
        # [b,emb_dims,num,1]
        return x

# TranformNet
# input x [B,num_dims,num]
class TranformNet(nn.Module):
    def __init__(self, k=3, negative_slope=1e-2, use_relu=True):
        super(TranformNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        if use_relu:
            self.relu = nn.ReLU
        else:
            self.relu = nn.LeakyReLU(negative_slope=negative_slope,inplace=True)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)),inplace=True)
        x = F.relu(self.bn2(self.conv2(x)),inplace=True)
        x = F.relu(self.bn3(self.conv3(x)),inplace=True)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)),inplace=True)
        x = F.relu(self.bn5(self.fc2(x)),inplace=True)
        x = self.fc3(x)

        device = torch.device('cuda')

        iden = torch.eye(self.k, dtype=torch.float32, device=device).view(1, self.k * self.k).repeat(batchsize, 1)

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
# input  [b,3,num]
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x) # [b,num,num]
    # 求坐标（维度空间）的平方和
    xx = torch.sum(x ** 2, dim=1, keepdim=True) # [b,1,num] #x ** 2 表示点平方而不是x*x
    # 2x1x2+2y1y2+2z1z2-x1^2-y1^2-z1^2-x2^2-y2^2-z2^2=-[(x1-x2)^2+(y1-y2)^2+(z1-z2)^2]
    pairwise_distance = -xx - inner
    del inner,x
    pairwise_distance = pairwise_distance - xx.transpose(2, 1) # [b,num,num]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

# input x [B,num_dims,num]
# output [B, num_dims*2, num, k] 领域特征tensor
def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    with torch.no_grad():
        device = torch.device('cuda')
        # 获得索引阶梯数组
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # (batch_size, 1, 1) [0 num_points ... num_points*(B-1)]
        # 以batch为单位，加到索引上
        idx = idx + idx_base  # (batch_size, num_points, k)
        # 展成一维数组，方便后续索引
        idx = idx.view(-1)  # (batch_size * num_points * k)
        # 获得特征维度
        _, num_dims, _ = x.size()
        x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)
        # 改变x的shape，方便索引。被索引数组是所有batch的所有点的特征，索引数组idx为所有临近点对应的序号，从而索引出所有领域点的特征
        feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size * num_points * k,num_dims)
        # 统一数组形式
        feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
        if cat_or_stack:
            # 重复k次，以便k个邻域点每个都能和中心点做运算
            x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # [B, num, k, num_dims]
            # 领域特征的表示，为(feature - x, x)，这种形式可以详尽参见dgcnn论文
            feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)  # [B, num_dims*2, num, k]
        else:
            x = x.view(batch_size, num_points, 1, num_dims)  # [B, num, 1, num_dims]
            # 领域特征的表示，为(feature - x, x)，这种形式可以详尽参见dgcnn论文
            feature = torch.cat((feature, x), dim=2).permute(0, 3, 1, 2)     # [B, num_dims, num, k+1]
    # del x,idx,idx_base
    return feature