#-*-coding:utf-8-*-
class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

c = np.load( "seq0.npz" )
print(type(c))
print(c['pc1'].shape)
print(c['pc2'].shape)
print(c['gt'].shape)
a=np.random.rand(3,3)
b = np.random.permutation(a)
c=np.asarray([1,2,1]).reshape(3,-1)
print(a)
print(a+c)
# x=c['pc1'][0,:,0].squeeze(0).squeeze(1)
# print(x.shape)
# fig=plt.figure(dpi=120)
# ax=fig.add_subplot(111,projection='3d')
# #标题
# plt.title('point cloud')
# #利用xyz的值，生成每个点的相应坐标（x,y,z）
# ax.scatter(x,y,z,c='b',marker='.',s=8,linewidth=0,alpha=1,cmap='spectral')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.scatter(x,y,z,c='b',marker='.',s=8,linewidth=0,alpha=1,cmap='spectral')
# #显示
# plt.show()