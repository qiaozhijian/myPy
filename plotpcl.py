#-*-coding:utf-8-*-
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA


def toCameraCoord(pose_mat):
    '''
        Convert the pose of lidar coordinate to camera coordinate
    '''
    R_C2L = np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]])
    inv_R_C2L = np.linalg.inv(R_C2L)
    R = np.dot(inv_R_C2L, pose_mat)
    rot = np.dot(R, R_C2L)
    return rot
def plot3d(data1,data2,s=1):
    fig=plt.figure(figsize=(10,30))
    ax=fig.add_subplot(111,projection='3d')
    # plt.scatter(x2, y2, s=area, c=area, cmap='rainbow', alpha=0.7)
    # area=np.random.rand(data1.shape[1])*1
    # ax.scatter(data1[0], data1[1], data1[2], c=area, s=s,cmap='Blues')
    ax.scatter(data1[0], data1[1], data1[2], color='blue', s=s)
    ax.scatter(data2[0], data2[1], data2[2], color='red', s=s)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
def velodyne2camera_pose(gt_v):
    t = np.asarray( [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
                    -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
                    9.999738645903e-01,4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
                    0,                  0,                  0,                   1                  ]).reshape(4,4)
    gt_c=t.dot(gt_v)
    gt_c = gt_c.dot(np.linalg.inv(t))
    return gt_c
def velodyne2camera(pc1,pc2):
    t = np.asarray([4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02, -7.210626507497e-03,8.081198471645e-03,
                    -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01, 4.859485810390e-04,-7.206933692422e-03, -2.921968648686e-01])
    t_r = np.asarray([t[0:3], t[4:7], t[8:11]]).reshape(3, 3)
    t_t = np.asarray([t[3], t[7], t[11]]).reshape(3, 1)

    pc1 = t_r.dot(pc1) + t_t
    pc2 = t_r.dot(pc2) + t_t
    return pc1,pc2
def velodyne2camera_inv(pc1,pc2):
    t = np.asarray([4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02, -7.210626507497e-03,8.081198471645e-03,
                    -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01, 4.859485810390e-04,-7.206933692422e-03, -2.921968648686e-01,0,0,0,1]).reshape(4,4)
    t=np.linalg.inv(t)
    t_r = t[0:3,0:3].reshape(3, 3)
    t_t = t[0:3,3].reshape(3, 1)

    pc1 = t_r.dot(pc1) + t_t
    pc2 = t_r.dot(pc2) + t_t
    return pc1,pc2
def camera2velodyne(gt_c):
    t = np.asarray( [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,-7.210626507497e-03,  \
                    8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01,4.859485810390e-04,  \
                            -7.206933692422e-03, -2.921968648686e-01,0,0,0,1]).reshape(4,4)
    gt_v = np.linalg.inv(t).dot(gt_c)
    gt_v=gt_v.dot(t)
    return gt_v

if __name__ == '__main__':
    modifyP=False
    # rawdata = np.load('../dcp/data/kitti/seq4.npz')
    rawdata = np.load('seq0.npz')
    num=2400
    pc1=rawdata['pc1'][num,:,0:3].T
    pc2=rawdata['pc2'][num,:,0:3].T
    gt=rawdata['gt'][num].T
    R_ab = np.asarray([gt[0:3], gt[4:7], gt[8:11]]).reshape(3, 3)
    translation_ab = np.asarray([gt[3], gt[7], gt[11]]).reshape(3)
    gt_raw=np.eye(4);gt_raw[0:3,0:3]=R_ab;gt_raw[0:3,3]=translation_ab
    gt=camera2velodyne(gt_raw).flatten()
    R_ab = np.asarray([gt[0:3], gt[4:7], gt[8:11]]).reshape(3, 3)
    translation_ab = np.asarray([gt[3], gt[7], gt[11]]).reshape(3,1)

    if modifyP:
        pc1, pc2 = velodyne2camera(np.copy(pc1), pc2)
        plot3d(pc1, pc2)
        pc1 = R_ab.dot(pc1) + translation_ab
        plot3d(pc1, pc2)
    else:
        gt_c = np.eye(4, 4)
        gt_c[0:3,0:3]=R_ab
        gt_c[0:3,3]=translation_ab.reshape(3)
        gt_v=velodyne2camera_pose(gt_c)
        R_ab = gt_v[0:3,0:3]
        translation_ab = gt_v[0:3,3]
        # plot3d(pc1, pc2)
        pc1 = R_ab.dot(pc1) + translation_ab.reshape(3,1)
        # plot3d(pc1, pc2)
        print(gt_v)
        print(np.mean(pc1--pc2))
        gt_v=toCameraCoord(gt_c)
        R_ab = gt_v[0:3,0:3]
        translation_ab = gt_v[0:3,3]
        # plot3d(pc1, pc2)
        pc1 = R_ab.dot(pc1) + translation_ab.reshape(3,1)
        print(np.mean(pc1--pc2))

        # plot3d(pc1, pc2)
        print(gt_v)