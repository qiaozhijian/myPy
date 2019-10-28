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

	pc1list=np.empty([0,4096,4],np.float32)
	pc2list=np.empty([0,4096,4],np.float32)
	gtlist=np.empty([0,12],np.float32)


	for frame in range(0,len(PCFileDir)-FrameGap):
	#for frame in range(100,101):
		frame2=frame+FrameGap
		pc1=np.fromfile(PointCloudDir+PCFileDir[frame],dtype=np.float32,count=-1).reshape([-1,4])
		pc2=np.fromfile(PointCloudDir+PCFileDir[frame2],dtype=np.float32,count=-1).reshape([-1,4])
			
		#plot3d(pc1,pc2,0.3)
		##remove ground
		pc1_rmground,pc1_ground=RemoveGround(pc1,distance_threshold,sample_size,max_iterations)
		pc2_rmground,pc2_ground=RemoveGround(pc2,distance_threshold,sample_size,max_iterations)

		#plot3d(pc1_ground,pc1_rmground,0.3)

		##random sapmle
		sample_idx1=RandomSample(pc1_rmground,npoints)
		sample_idx2=RandomSample(pc2_rmground,npoints)
		pc1_sampled = pc1_rmground[sample_idx1, :]
		pc2_sampled = pc2_rmground[sample_idx2, :]

		t = np.asarray([4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02, -7.210626507497e-03,
		 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01, 4.859485810390e-04,
		 -7.206933692422e-03, -2.921968648686e-01])
		t_r = np.asarray([t[0:3], t[4:7], t[8:11]]).reshape(3, 3)
		t_t = np.asarray([t[3], t[7], t[11]]).reshape(3, 1)
		pc1 = (t_r.dot(pc1_sampled.T) + t_t).T
		pc2 = (t_r.dot(pc2_sampled.T) + t_t).T

		#plot3d(pc1_sampled,pc2_sampled,s=0.2)
		#print('pc1_sampled'+str(pc1_sampled.shape)+'\n')
		#print('pc1list'+str(pc1list.shape)+'\n')
		#put into list pc1list and pc2list
		pc1list = np.append(pc1list, np.expand_dims(pc1,axis=0), axis = 0)
		pc2list = np.append(pc2list, np.expand_dims(pc2,axis=0), axis = 0)
			
		#compute T21 groundtruth and put into list gtlist
		Tw1=np.zeros([4,4])
		Tw2=np.zeros([4,4])
		Tw1[0:3,0:4]=SeqPose[frame].reshape([3,4])
		Tw2[0:3,0:4]=SeqPose[frame2].reshape([3,4])
		Tw1[3,3]=1
		Tw2[3,3]=1
		T21=(np.dot(np.linalg.inv(Tw2),Tw1))
		gt=(T21[0:3,0:4]).reshape([1,12])
		gtlist = np.append(gtlist, gt, axis = 0)
		print('process frame'+str(frame))

		##*********经过正确位姿旋转后，再可视化看一下**********##
		R_ab = np.asarray([gt[0:3], gt[4:7], gt[8:11]]).reshape(3, 3)
		translation_ab = np.asarray([gt[3], gt[7], gt[11]]).reshape(3, 1)
		plot3d(pc1, pc2)
		pc1 = R_ab.dot(pc1.T) + translation_ab
		plot3d(pc1, pc2)

	#save as npz for each seq
	np.savez(OutDir+'seq'+str(seq)+'.npz',pc1=pc1list,pc2=pc2list,gt=gtlist)
	print('save to '+OutDir+'seq'+str(seq)+'.npz')

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
    for seq in range(0,11):
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
