#-*-coding:utf-8-*-
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


def plot3d(data1,data2,s=0.2):
    fig=plt.figure(figsize=(10,30))
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(data1[0], data1[1], data1[2], color='blue', s=s)
    ax.scatter(data2[0], data2[1], data2[2], color='red', s=s)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

if __name__ == '__main__':
    rawdata = np.load('seq0.npz')
    pc1=rawdata['pc1'][3402,:,0:3].T
    pc2=rawdata['pc2'][3402,:,0:3].T
    gt=rawdata['gt'][3402].T
    R_ab = np.asarray([gt[0:3], gt[4:7], gt[8:11]]).reshape(3, 3)
    translation_ab = np.asarray([gt[3], gt[7], gt[11]]).reshape(3,1)
    t=np.asarray([4.276802385584e-04,-9.999672484946e-01,-8.084491683471e-03,-1.198459927713e-02,-7.210626507497e-03,8.081198471645e-03,-9.999413164504e-01,-5.403984729748e-02,9.999738645903e-01,4.859485810390e-04,-7.206933692422e-03,-2.921968648686e-01])
    t_r = np.asarray([t[0:3], t[4:7], t[8:11]]).reshape(3, 3)
    t_t = np.asarray([t[3], t[7], t[11]]).reshape(3,1)
    pc1=t_r.dot(pc1)+t_t
    pc2=t_r.dot(pc2)+t_t
    print(gt.reshape(3,4))
    plot3d(pc1,pc2)
    pc1=R_ab.dot(pc1)+translation_ab
    plot3d(pc1,pc2)
