# -*- coding: utf-8 -*-
'''
process kitti

remove ground, downsample, compute pose groundtruth

author: syl
created: 10/23/19
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import multiprocessing as multiproc


PCBaseDir='/media/shenyl/SYL-MHD/datatset/kitti_odometry_velodyne/data_odometry_velodyne/dataset/sequences/'
PoseBaseDir='/media/shenyl/SYL-MHD/datatset/kitti_odometry_velodyne/dataset/poses/'
OutDir='/media/shenyl/SYL-MHD/datatset/kitti_odometry_velodyne/processed_data/'
RmgroundOutDir='/media/shenyl/SYL-MHD/datatset/kitti_odometry_velodyne/rmground/'
PCDir=os.listdir(PCBaseDir)
PoseDir=os.listdir(PoseBaseDir)


#####

def RandomSample(pc,npoints):
	n=pc.shape[0]
	sample_idx = np.random.choice(n, npoints, replace=False)
	return sample_idx

#计算三点构成的平面参数
def estimate_plane(xyz,normalize=True):
	vector1 = xyz[1,:] - xyz[0,:]
	vector2 = xyz[2,:] - xyz[0,:]

	
	#判断vector1是否为0
	if not np.all(vector1):
		return None
	#共线性检查,如果vector1和vector2三维同比例，则三点共线
	dy1dy2 = vector2 / vector1
	if  not ((dy1dy2[0] != dy1dy2[1])  or  (dy1dy2[2] != dy1dy2[1])):
		return None

	a = (vector1[1]*vector2[2]) - (vector1[2]*vector2[1])
	b = (vector1[2]*vector2[0]) - (vector1[0]*vector2[2])
	c = (vector1[0]*vector2[1]) - (vector1[1]*vector2[0])

	#normalize
	if normalize:
		r = math.sqrt(a ** 2 + b ** 2 + c ** 2)
		a = a / r
		b = b / r
		c = c / r
	d = -(a*xyz[0,0] + b*xyz[0,1] + c*xyz[0,2])
	return np.array([a,b,c,d])

#remove ground by ransac
def RemoveGround(pc,distance_threshold,sample_size,max_iterations):
	random.seed(12345)
	max_point_num=-999
	i=0
	R_L=range(pc.shape[0])
	max_pc_ground=np.empty([0,4])
	pc3d=pc[:,0:3]

	while i<max_iterations:
		s3=random.sample(R_L,sample_size)
		coeffs = estimate_plane(pc3d[s3,:], normalize=False) #计算平面方程系数
		if coeffs is None:
			continue

		#计算平面法向量的模
		r = np.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2 )
		#若平面法向量与Z轴的夹角大于45度则可能为墙壁，剔除这种情况
		zaxis = np.array([0,0,1])
		nor = abs(coeffs[:3])
		if math.acos(np.dot(zaxis.T,nor)/(r))>math.pi/4:
			#print('find a wall!continue')
			continue

		#计算每个点和平面的距离，距离小于阈值的点作为平面上的点
		d = np.divide(np.abs(np.matmul(coeffs[:3], pc3d.T) + coeffs[3]) , r)
		d_filt = np.array(d < distance_threshold)
		pc_ground = pc[d_filt,:]
		near_point_num = pc_ground.shape[0]
		d_filt = np.array(d >= distance_threshold)
		pc_rmground = pc[d_filt,:]

		#选出内点数最多的平面
		if near_point_num > max_point_num:
			max_point_num = near_point_num
			max_pc_ground = pc_ground
			max_pc_rmground = pc_rmground

		i=i+1
	return max_pc_rmground,max_pc_ground



def plot3d(data1,data2,s=0.2):
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(data1[:,0],data1[:,1],data1[:,2],color='red',s=s)
	ax.scatter(data2[:,0],data2[:,1],data2[:,2],color='green',s=s)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	plt.show()



def LoadData(FrameGap,npoints,seq):
	print('start run seq'+str(seq))

	#去地面参数
	distance_threshold=0.3
	sample_size=3
	max_iterations=300

	PointCloudDir=PCBaseDir+PCDir[seq]+'/velodyne/'
	PCFileDir=os.listdir(PointCloudDir)
	SeqPose=np.loadtxt(PoseBaseDir+PoseDir[seq])
	RmgroundDir=RmgroundOutDir+PCDir[seq]+'/'

	pc1list=np.empty([0,4096,4],np.float32)
	pc2list=np.empty([0,4096,4],np.float32)
	gtlist=np.empty([0,12],np.float32)


	#for frame in range(0,len(PCFileDir)-FrameGap):
	#for frame in range(100,101):
	for frame in range(len(PCFileDir)-FrameGap,len(PCFileDir)):
		pc1=np.fromfile(PointCloudDir+PCFileDir[frame],dtype=np.float32,count=-1).reshape([-1,4])
			
		#plot3d(pc1,pc2,0.3)
		##remove ground
		pc1_rmground,pc1_ground=RemoveGround(pc1,distance_threshold,sample_size,max_iterations)


		print('process frame'+str(frame))
		pc1_rmground.tofile(RmgroundDir+PCFileDir[frame])
		#read=np.fromfile(RmgroundDir+PCFileDir[frame],dtype=np.float32,count=-1).reshape([-1,4])
		#plot3d(pc1_rmground,read,0.3)


def run_all_processes(all_p):
		try:
			for p in all_p:
				p.start()
			for p in all_p:
				p.join()
		except KeyboardInterrupt:
			for p in all_p:
				if p.is_alive():
					p.terminate()
				p.join()
			exit(-1)

if __name__ == '__main__':
    FrameGap=5     #用于回归位姿的两帧之间的间隔
    npoints=4096   #采样点个数

    all_p = []
    for seq in range(1,11):
    	all_p.append(multiproc.Process(target=LoadData,args=(FrameGap,npoints,seq)))
    run_all_processes(all_p)	

    ###read from npz
    # data = np.load(OutDir+'seq5.npz')
    # pc1list=data['pc1']
    # pc2list=data['pc2']
    # gtlist=data['gt']

    # pc1=pc1list[0]
    # pc2=pc2list[0]
    # plot3d(pc11,pc2)
