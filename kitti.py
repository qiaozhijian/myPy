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


isRmGround=False
if isRmGround:
	PCBaseDir = '/media/qzj/My Book/KITTI/data_odometry_velodyne/rmground/'
else:
	PCBaseDir='/media/qzj/My Book/KITTI/data_odometry_velodyne/dataset/sequences/'
RmgroundOutDir='./rmground/'
GroundOutDir='./ground/'
PoseBaseDir='/media/qzj/My Book/KITTI/data_odometry_poses/'
OutDir='/media/qzj/Ubuntu 18.0/kitti/'

if not os.path.exists(OutDir):
	os.mkdir(OutDir)
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
def RemoveGround(pc,distance_threshold=0.3,sample_size=3,max_iterations=300):
	i=0
	random.seed(1234)
	max_point_num=-999
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
		if math.acos(abs(coeffs[2])/r)>math.pi/4:
			continue

		#计算每个点和平面的距离，距离小于阈值的点作为平面上的点
		d = np.divide(np.abs(np.matmul(coeffs[:3], pc3d.T) + coeffs[3]) , r)
		near_point_num = pc[np.array(d < distance_threshold),:].shape[0]

		coeffs2 = np.copy(coeffs);
		if coeffs[2]<0:
			coeffs2[3] = coeffs[3] + distance_threshold * r
			d = np.matmul(coeffs2[:3], pc3d.T) + coeffs2[3]
			pc_ground = pc[np.array(d >= 0), :]
			pc_rmground = pc[np.array(d < 0), :]
		else:
			coeffs2[3] = coeffs[3] - distance_threshold * r
			d = np.matmul(coeffs2[:3], pc3d.T) + coeffs2[3]
			pc_ground = pc[np.array(d < 0), :]
			pc_rmground = pc[np.array(d >= 0), :]
		#选出内点数最多的平面
		if near_point_num > max_point_num:
			max_coeffs=coeffs
			max_point_num = near_point_num
			max_pc_ground = pc_ground
			max_pc_rmground = pc_rmground

		i=i+1
	return max_pc_rmground,max_pc_ground



def plot3d(data1,data2,s=0.2):
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(data1[:,0],data1[:,1],data1[:,2],color='blue',s=s)
	ax.scatter(data2[:,0],data2[:,1],data2[:,2],color='red',s=s)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	plt.show()
def plotpc(data1,s=0.1):
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(data1[:,0],data1[:,1],data1[:,2],color='blue',s=s)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	plt.show()

# pc1
def velodyne2camera(pc1,pc2):
    t = np.asarray([4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02, -7.210626507497e-03,8.081198471645e-03,
                    -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01, 4.859485810390e-04,-7.206933692422e-03, -2.921968648686e-01])
    t_r = np.asarray([t[0:3], t[4:7], t[8:11]]).reshape(3, 3)
    t_t = np.asarray([t[3], t[7], t[11]]).reshape(3, 1)

    pc1 = t_r.dot(pc1) + t_t
    pc2 = t_r.dot(pc2) + t_t
    return pc1,pc2

def generateRmGroud(seq):
	print('start run seq'+str(seq))
	# 看一下这些是否有必要改成直接写字符串
	PointCloudDir=PCBaseDir+PCDir[seq]+'/velodyne/'
	bin_num=len(os.listdir(PointCloudDir))
	RmgroundDir=RmgroundOutDir+PCDir[seq]+'/'
	GroundDir=GroundOutDir+PCDir[seq]+'/'

	for frame in range(bin_num):
		path=PointCloudDir+str(frame).zfill(6)+'.bin'
		pc=np.fromfile(path,dtype=np.float32,count=-1).reshape([-1,4])
		pc_rmground,pc_ground=RemoveGround(pc)
		if frame%100==0:
			print('seq: '+str(seq)+'  frame: '+str(frame))
		pc_rmground.tofile(RmgroundDir+str(frame).zfill(6)+'.bin')
		pc_ground.tofile(GroundDir+str(frame).zfill(6)+'.bin')
	print('end run seq'+str(seq))

# def sampleUniform(pointCloud):
# 	pointCloud=pointCloud[:,0:3]
def camera2velodyne(gt_c):
    t = np.asarray( [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,-7.210626507497e-03,  \
                    8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01,4.859485810390e-04,  \
                            -7.206933692422e-03, -2.921968648686e-01,0,0,0,1]).reshape(4,4)
    gt_v = np.linalg.inv(t).dot(gt_c)
    gt_v=gt_v.dot(t)
    return gt_v


def LoadData(FrameGap,npoints,seq):
	print('start run seq'+str(seq))
	if isRmGround:
		PointCloudDir=PCBaseDir+PCDir[seq]+'/'
	else:
		PointCloudDir=PCBaseDir+PCDir[seq]+'/velodyne/'
	bin_num=len(os.listdir(PointCloudDir))
	SeqPose=np.loadtxt(PoseBaseDir+PoseDir[seq])

	pc1list=np.empty([0,npoints,4],np.float32)
	pc2list=np.empty([0,npoints,4],np.float32)
	gtlist=np.empty([0,12],np.float32)


	for frame in range(0,bin_num-FrameGap):
	# for frame in np.linspace(0,bin_num-FrameGap,2,endpoint=False).astype(int):
	# for frame in [0]:
		frame2=frame+FrameGap
		path1=PointCloudDir+str(frame).zfill(6)+'.bin';path2=PointCloudDir+str(frame2).zfill(6)+'.bin';
		pc1_raw=np.fromfile(path1,dtype=np.float32,count=-1).reshape([-1,4])
		pc2_raw=np.fromfile(path2,dtype=np.float32,count=-1).reshape([-1,4])

		##remove ground
		if not isRmGround:
			pc2_rmground,pc2_ground=RemoveGround(pc2_raw)
			pc1_rmground,pc1_ground=RemoveGround(pc1_raw)

		##random sapmle
		pc1_sampled=np.random.permutation(pc1_rmground)[0:npoints]
		pc2_sampled=np.random.permutation(pc2_rmground)[0:npoints]

		pc1list = np.append(pc1list, np.expand_dims(pc1_sampled,axis=0), axis = 0)
		pc2list = np.append(pc2list, np.expand_dims(pc2_sampled,axis=0), axis = 0)

		#compute T21 groundtruth and put into list gtlist
		Tw1=np.eye(4,4)
		Tw2=np.eye(4,4)
		Tw1[0:3,0:4]=SeqPose[frame].reshape([3,4])
		Tw2[0:3,0:4]=SeqPose[frame2].reshape([3,4])
		T21=(np.dot(np.linalg.inv(Tw2),Tw1))
		T21=camera2velodyne(T21)
		gt=(T21[0:3,0:4]).reshape([1,12])
		gtlist = np.append(gtlist, gt, axis = 0)
		if frame%100==0:
			print('seq: '+str(seq)+'  frame: '+str(frame))

		####*********经过正确位姿旋转后，再可视化看一下**********##
		# R_ab = np.asarray([gt[0,0:3], gt[0,4:7], gt[0,8:11]]).reshape(3, 3)
		# translation_ab = np.asarray([gt[0,3], gt[0,7], gt[0,11]]).reshape(3, 1)
		# plot3d(pc1_sampled, pc2_sampled)
		# pc1_sampled[:,0:3] = (R_ab.dot(pc1_sampled[:,0:3].T) + translation_ab).T
		# plot3d(pc1_sampled, pc2_sampled)

	#save as npz for each seq
	print('end run seq'+str(seq))
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

    all_p = []
    for seq in range(0,11):
		all_p.append(multiproc.Process(target=LoadData,args=(5,4096,seq)))
		# all_p.append(multiproc.Process(target=generateRmGroud,args=(seq)))
    run_all_processes(all_p)


