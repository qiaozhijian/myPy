import numpy as np
import time
import pcl
import copy
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import cKDTree
path='/media/qzj/Document/grow/research/slamDataSet/kitti/data_odometry_velodyne/dataset/downsample4096/'+str(0).zfill(2)+'/'+str(0).zfill(6)+'.bin'
    # path = '/media/qzj/My Book/KITTI/data_odometry_velodyne/dataset/sequences/' + str(seqN).zfill(2) + '/velodyne/' + str(binNum).zfill(6) + '.bin'
pc = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4]);
scan1 = pc[:, :3]
p1 = pcl.PointCloud(scan1)

path='/media/qzj/Document/grow/research/slamDataSet/kitti/data_odometry_velodyne/dataset/downsample4096/'+str(0).zfill(2)+'/'+str(5).zfill(6)+'.bin'
    # path = '/media/qzj/My Book/KITTI/data_odometry_velodyne/dataset/sequences/' + str(seqN).zfill(2) + '/velodyne/' + str(binNum).zfill(6) + '.bin'
pc = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4]);
scan2 = pc[:, :3]
p2 = pcl.PointCloud(scan2)

icp = pcl.IterativeClosestPoint()
T = icp.icp(p1, p2)

print(list(T))
