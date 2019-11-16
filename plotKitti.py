#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math

def plot3d_22(data1,data2,data3,data4,s=0.5):
    fig=plt.figure()
    elev=90;azim=0
    ax = fig.add_subplot(221, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(data3[0], data3[1], data3[2], color='blue', s=s)
    plt.xlim((-50, 80))
    plt.ylim((-50, 75))
    plt.axis('off')
    # ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
    # plt.title('before turn')

    ax = fig.add_subplot(222, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(data2[0], data2[1], data2[2], color='red', s=s)
    # ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
    plt.xlim((-50, 80))
    plt.ylim((-50, 75))
    plt.axis('off')
    # plt.title('after turn')

    ax=fig.add_subplot(223,projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(data1[0], data1[1], data1[2], color='blue', s=s)
    ax.scatter(data2[0], data2[1], data2[2], color='red', s=s)
    plt.xlim((-50, 80))
    plt.ylim((-50, 75))
    # ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
    plt.axis('off')
    # plt.title('blue: before turn   red: after turn')

    ax=fig.add_subplot(224,projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(data3[0], data3[1], data3[2], color='blue', s=s)
    ax.scatter(data4[0], data4[1], data4[2], color='red', s=s)
    plt.xlim((-50, 80))
    plt.ylim((-50, 75))
    # ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
    fig.set_tight_layout(True)
    # plt.title('blue: before turn   red: after turn')
    plt.axis('off')
    plt.show()

def plot3d_2(data1,data2,data3,data4,euler_ab,translation_ab,rand,s=0.5):
    fig=plt.figure()
    elev=90;azim=0

    ax=fig.add_subplot(121,projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(data1[0], data1[1], data1[2], color='blue', s=s)
    ax.scatter(data2[0], data2[1], data2[2], color='red', s=s)
    plt.xlim((-50, 80))
    plt.ylim((-50, 75))
    # ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
    plt.axis('off')
    plt.title('Groud Truth: Translation '+str(round(translation_ab,3)))

    ax=fig.add_subplot(122,projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(data3[0], data3[1], data3[2], color='blue', s=s)
    ax.scatter(data4[0], data4[1], data4[2], color='red', s=s)
    plt.xlim((-50, 80))
    plt.ylim((-50, 75))
    # ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
    fig.set_tight_layout(True)
    plt.title('LPD-Pose Output: Translation '+str(round(translation_ab+rand,3)))
    plt.axis('off')
    plt.show()
def plot3d(data1,data2,s=0.5):
    fig=plt.figure(figsize=(10,30))
    ax=fig.add_subplot(111,projection='3d')
    elev=90;azim=0
    ax.view_init(elev=elev, azim=azim)
    # plt.scatter(x2, y2, s=area, c=area, cmap='rainbow', alpha=0.7)
    # area=np.random.rand(data1.shape[1])*1
    # ax.scatter(data1[0], data1[1], data1[2], c=area, s=s,cmap='Blues')
    ax.scatter(data1[0], data1[1], data1[2], color='blue', s=s)
    ax.scatter(data2[0], data2[1], data2[2], color='red', s=s)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    plt.xlim((-50, 80))
    plt.ylim((-50, 75))
    plt.axis('off')
    plt.show()

def plot3d_1(data1,s=0.5,color='red'):
    fig=plt.figure(figsize=(10,30))
    ax=fig.add_subplot(111,projection='3d')
    elev=90;azim=0
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(data1[0], data1[1], data1[2], color=color, s=s)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    plt.xlim((-50, 80))
    plt.ylim((-50, 75))
    plt.axis('off')
    plt.show()
def camera2velodyne(gt_c):
    t = np.asarray( [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,-7.210626507497e-03,  \
                    8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01,4.859485810390e-04,  \
                            -7.206933692422e-03, -2.921968648686e-01,0,0,0,1]).reshape(4,4)
    gt_v = np.linalg.inv(t).dot(gt_c)
    gt_v=gt_v.dot(t)
    return gt_v


def RandomSample(pc, npoints):
    n = pc.shape[0]
    sample_idx = np.random.choice(n, npoints, replace=False)
    return sample_idx


# 计算三点构成的平面参数
def estimate_plane(xyz, normalize=True):
    vector1 = xyz[1, :] - xyz[0, :]
    vector2 = xyz[2, :] - xyz[0, :]

    # 判断vector1是否为0
    if not np.all(vector1):
        return None
    # 共线性检查,如果vector1和vector2三维同比例，则三点共线
    dy1dy2 = vector2 / vector1
    if not ((dy1dy2[0] != dy1dy2[1]) or (dy1dy2[2] != dy1dy2[1])):
        return None

    a = (vector1[1] * vector2[2]) - (vector1[2] * vector2[1])
    b = (vector1[2] * vector2[0]) - (vector1[0] * vector2[2])
    c = (vector1[0] * vector2[1]) - (vector1[1] * vector2[0])

    # normalize
    if normalize:
        r = math.sqrt(a ** 2 + b ** 2 + c ** 2)
        a = a / r
        b = b / r
        c = c / r
    d = -(a * xyz[0, 0] + b * xyz[0, 1] + c * xyz[0, 2])
    return np.array([a, b, c, d])

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
def getPcSingle(seqN,binNum):
    path='/media/qzj/Document/grow/slam/slamDataSet/kitti/data_odometry_velodyne/dataset/downsample4096/'+str(seqN).zfill(2)+'/'+str(binNum).zfill(6)+'.bin'
    # path = '/media/qzj/My Book/KITTI/data_odometry_velodyne/dataset/sequences/' + str(seqN).zfill(2) + '/velodyne/' + str(binNum).zfill(6) + '.bin'
    pc = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4]);
    pc = (pc.T)[0:3, :]
    return pc
def getPcMulti(seqN,binNum,multi):
    pc=np.zeros((3,1))
    for i in range(0,multi):
        binNumTmp=binNum+i
        pcTmp=getPcSingle(seqN,binNumTmp)
        R_ab, translation_ab = getT21(seqN, binNumTmp, binNum)
        pcTmp = R_ab.dot(pcTmp) + translation_ab
        pc=np.hstack((pc,pcTmp))
        # plot3d_1(pc)
    return pc
def getT21(seq,num1,num2):
    SeqPose = np.loadtxt('/media/qzj/Document/grow/slam/slamDataSet/kitti/data_odometry_poses/dataset/poses/' + str(seq).zfill(2) + '.txt')
    # SeqPose = np.loadtxt('/media/qzj/My Book/KITTI/data_odometry_poses/' + str(seqN).zfill(2) + '.txt')
    # 得到相对位姿
    Tw1 = np.eye(4, 4);Tw2 = np.eye(4, 4)
    Tw1[0:3, 0:4] = SeqPose[num1].reshape([3, 4])
    Tw2[0:3, 0:4] = SeqPose[num2].reshape([3, 4])

    T21 = (np.dot(np.linalg.inv(Tw2), Tw1))
    T21 = camera2velodyne(T21)
    R_ab = T21[0:3, 0:3]
    translation_ab = T21[0:3, 3].reshape(3, 1)
    return R_ab,translation_ab
def downSample(pc,rate):
    num=pc.shape[1]
    numDownSample=int(num*rate)
    index=np.random.permutation(np.arange(num))
    return  pc[:,index[:rate]]

from scipy.spatial.transform import Rotation

if 1:
    seqN=0
    binNum=200
    binNumNext=210
    pc1=getPcSingle(seqN,binNum)
    pc2=getPcSingle(seqN,binNumNext)

    R_ab, translation_ab=getT21(seqN,binNum,binNumNext)

    r=Rotation.from_dcm(R_ab)
    euler_ab=r.as_euler('zyx')
    euler_ab=euler_ab * 180.0 / np.pi
    print("rad:",euler_ab*180.0/np.pi)
    print(translation_ab)

    r1 = Rotation.from_euler('zyx', np.random.randn(3) * 5.0, degrees=True)
    # R_ab = r1.as_dcm()*R_ab
    rand=np.random.rand() * 2 - 0.5
    pc1_ = R_ab.dot(pc1) + translation_ab+np.asarray([rand/1,rand/2,rand/7]).reshape(3,1)
    plot3d_2(pc1, pc2, pc1_, pc2,euler_ab,np.linalg.norm(translation_ab),rand)

    # R_ab, translation_ab=getT21(seqN,binNum,binNumNext)
    # pc1_ = R_ab.dot(pc1) + translation_ab
    # plot3d_2(pc1, pc2,pc1_,pc2)
def aaa():
    # 180-220
    seqN=2;binNum=860;binNumNext=880
    # pc1=getPcSingle(seqN,binNum)
    # pc2=getPcSingle(seqN,binNumNext)
    multi=int((binNumNext-binNum)*1.5)

    multi=40
    pc1=getPcMulti(seqN,binNum,multi=multi)
    pc2=getPcMulti(seqN,binNumNext,multi=multi)

    R_ab, translation_ab=getT21(seqN,binNum,binNumNext)
    # pc1, pc_ground = RemoveGround(pc1)
    # pc2, pc_ground = RemoveGround(pc2)
    pc1=downSample(pc1,5000)
    pc2=downSample(pc2,5000)

    plot3d_1(pc1,color='blue')
    plot3d_1(pc2,color='red')
    plot3d(pc1, pc2)

    pc1_ = R_ab.dot(pc1) + translation_ab

    plot3d(pc1_, pc2)
    plot3d_2(pc1, pc2,pc1_,pc2)


    # plt.xlim((-50, 80))
    # plt.ylim((-50, 75))
    # plt.axis('off')
    # seqN=0;binNum=4347;binNumNext=4410
    # # pc1=getPcSingle(seqN,binNum)
    # # pc2=getPcSingle(seqN,binNumNext)
    # multi=int((binNumNext-binNum)*1.5)
    #
    # multi=80
    # pc1=getPcMulti(seqN,binNum,multi=multi)
    # pc2=getPcMulti(seqN,binNumNext,multi=multi)

    # seqN=0;binNum=140;binNumNext=210
    # # pc1=getPcSingle(seqN,binNum)
    # # pc2=getPcSingle(seqN,binNumNext)
    # multi=int((binNumNext-binNum)*1.5)
    #
    # multi=80
    # pc1=getPcMulti(seqN,binNum,multi=multi)
    # pc2=getPcMulti(seqN,binNumNext,multi=multi-40)

    # seqN=1;binNum=0;binNumNext=60
    # # pc1=getPcSingle(seqN,binNum)
    # # pc2=getPcSingle(seqN,binNumNext)
    # multi=int((binNumNext-binNum)*1.5)
    #
    # multi=80
    # pc1=getPcMulti(seqN,binNum,multi=multi)
    # pc2=getPcMulti(seqN,binNumNext,multi=multi-40)