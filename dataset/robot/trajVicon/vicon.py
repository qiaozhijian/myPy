import numpy as np
import yaml
import cv2
from scipy.spatial.transform import Rotation
import math
import matplotlib.pyplot as plt
import os
from paras import T_c0_robot, T_imu_c0, T_odo_imu, T_odo_robot, T_robot_odo, T_robot_c0

scale = 0.95
seq = "01"
slam = 'plo'
type_slam = 'vo'
path = '/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/{}'.format(seq)
debug = True
scaleAlign = False
Align = False
suffix = ""
if scaleAlign:
    suffix = suffix + " -s"
if Align:
    suffix = suffix + " -va"


def main():
    # time x y z qx qy qz w
    slamTxt = os.path.join(path, "robot{}_{}_stereo_{}.txt").format(seq, slam, type_slam)
    # time x y z qx qy qz w
    viconTxt = os.path.join(path, "vicon_{}.txt".format(seq))  # full mode: 时间，轴角，旋转矩阵，四元数，欧拉角
    # time x y z qx qy qz w
    odometryTxt = os.path.join(path, "odometry{}.txt".format(seq))
    # scipy.spatial.transform.Rotation的四元数顺序是(x,y,z,w)

    slamTraj = np.loadtxt(slamTxt, dtype=np.float64)
    viconTraj = np.loadtxt(viconTxt, dtype=np.float64)
    odometryTraj = np.loadtxt(odometryTxt, dtype=np.float64)

    slamTraj[:,1:4] = slamTraj[:,1:4] * scale

    slamTraj = trans_robot_slam(slamTraj)
    viconTraj = trans_robot_vicon(viconTraj)
    odometryTraj = trans_robot_odometry(odometryTraj)

    # plot3d(slamTraj[:,1:4], viconTraj[:,1:4])
    # plot3d(odometryTraj[:,1:4], viconTraj[:,1:4])
    plot3d(slamTraj[:, 1:4], odometryTraj[:, 1:4])

    eval_name = slamTxt[:-4] + "_BodyFrame.txt"
    gt_file_new = viconTxt[:-4] + "_BodyFrame.txt"
    odo_file_new = odometryTxt[:-4] + "_BodyFrame.txt"
    saveTum(eval_name, slamTraj)
    saveTum(gt_file_new, viconTraj)
    saveTum(odo_file_new, odometryTraj)

    if (False):
        # os.system(
        #     "evo_traj tum " + eval_name + " --ref " + gt_file_new + " --save_plot {}_{}_{}.pgf --plot --plot_mode=xyz".format(
        #         seq, slam, type_slam) + suffix)
        # angle_deg
        os.system("evo_ape tum " + gt_file_new + " " + eval_name + " -r trans_part -p" + suffix)
        os.system("evo_ape tum " + gt_file_new + " " + eval_name + " -r angle_deg" + suffix)
        # os.system("evo_rpe tum "+ gt_file_new+ " " + eval_name + suffix)


def debugVicon(Traj, End1, End2):
    if End1 == 0:
        print("perfect start")
    else:
        print("debug start")
        # plot3d(Traj[:End1,1:4])

    if End2 >= Traj.shape[0] - 1:
        print("perfect end")
    else:
        print("debug end")
        # plot3d(Traj[End2:,1:4])
    for i in range(Traj.shape[0]):
        T_vicon_robot = tum2kitti(Traj[i, 1:])
        T_robot_vicon = np.linalg.inv(T_vicon_robot)
        # # 返回欧拉角也是zyx顺序
        print(Rotation.from_dcm(T_robot_vicon[:3, :3]).as_euler("zyx") / math.pi * 180.0)


def saveTum(file_name, traj):
    w = open(file_name, 'w')
    for i in range(traj.shape[0]):
        ss = str(traj[i, 0])
        for j in range(1, 8):
            ss = ss + " " + str(traj[i, j])
        w.write(ss + '\n')
    w.close()


def alignTime(slamTraj, viconTraj):
    timeMax = slamTraj[-1, 0]
    timeMin = slamTraj[0, 0]
    num = viconTraj.shape[0]
    timeVicon = np.linspace(timeMin, timeMax, num)
    viconTraj[:, 0] = timeVicon.reshape(-1)
    return slamTraj, viconTraj


def trans_robot_slam(slamTraj):
    num, _ = slamTraj.shape
    for i in range(num):
        T = tum2kitti(slamTraj[i, 1:])
        T = np.dot(T_robot_c0, T)
        T = np.dot(T, T_c0_robot)
        slamTraj[i, 1:] = np.asarray(kitti2tum(T))
    return slamTraj


def trans_robot_vicon(viconTraj):
    T_vicon_robot = tum2kitti(viconTraj[0, 1:])
    T_robot_vicon = np.linalg.inv(T_vicon_robot)
    # print(T_vicon_robot[:3,:3])
    # print(T_robot_vicon[:3,:3])
    # 返回欧拉角也是zyx顺序
    # print(Rotation.from_dcm(T_robot_vicon[:3,:3]).as_euler("zyx")/math.pi*180.0)
    # 返回欧拉角也是zyx顺序
    # print(Rotation.from_dcm(T_vicon_robot[:3,:3]).as_euler("zyx")/math.pi*180.0)
    num, _ = viconTraj.shape
    for i in range(num):
        try:
            T = tum2kitti(viconTraj[i, 1:])
        except ValueError:
            print(viconTraj[i, 1:])
        T_ = np.dot(T_robot_vicon, T)
        # print(T[:3,3],T_[:3,3])
        # 返回欧拉角也是zyx顺序
        # print(Rotation.from_dcm(T[:3,:3]).as_euler("zyx")/math.pi*180.0)
        # T = np.dot(T, T_robot_vicon)
        viconTraj[i, 1:] = np.asarray(kitti2tum(T_))
    return viconTraj


def trans_robot_odometry(odometryTraj):
    num, _ = odometryTraj.shape
    for i in range(num):
        T = tum2kitti(odometryTraj[i, 1:])
        T = np.dot(T_robot_odo, T)
        T = np.dot(T, T_odo_robot)
        odometryTraj[i, 1:] = np.asarray(kitti2tum(T))
    return odometryTraj


def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


def kitti2tum(kitti):
    dcm = kitti[:3, :3]
    t = kitti[:3, 3].tolist()
    q = Rotation.from_dcm(dcm).as_quat()
    tum = list(flat([t, q.tolist()]))
    return np.asarray(tum)


def tum2kitti(tum):
    t = tum[:3]
    q = tum[3:]
    rotation = Rotation.from_quat(q)
    r = rotation.as_dcm()
    T = np.eye(4)
    T[0:3, 0:3] = r
    T[0:3, 3] = t
    return T


def getUsefulPart(traj, vicon=False):
    trans = traj[:, 1:4]
    num, _ = traj.shape;
    endIdx1 = 0
    if vicon:
        thresold = 0.00002
    else:
        thresold = 0.00055
    for i in range(num):
        egoMotion = np.sqrt(np.sum(np.square(trans[i + 1, :] - trans[i, :])))
        # print(egoMotion)
        if egoMotion < thresold:
            endIdx1 = i
        else:
            break
    traj = traj[endIdx1:]

    trans = traj[:, 1:4]
    num, _ = traj.shape;
    endIdx2 = num - 1
    for i in range(num):
        j = num - i - 1;
        egoMotion = np.sqrt(np.sum(np.square(trans[j - 1, :] - trans[j, :])))
        # print(egoMotion)
        if egoMotion < thresold:
            endIdx2 = j
        else:
            break
    traj = traj[:endIdx2]

    return traj, endIdx1, endIdx2


def plot3d(data1, data2=None, s=1):
    if data1.shape[0] > data1.shape[1]:
        data1 = data1.T;
    if data2 is not None and data2.shape[0] > data2.shape[1]:
        data2 = data2.T;
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[0], data1[1], data1[2], color='blue', s=s)
    ax.scatter(data1[0, 0], data1[1, 0], data1[2, 0], s=200, marker='v', edgecolors='b', linewidths=2)
    if data2 is not None:
        ax.scatter(data2[0], data2[1], data2[2], color='red', s=s)
        ax.scatter(data2[0, 0], data2[1, 0], data2[2, 0], s=200, marker='v', edgecolors='r', linewidths=2)
        # print(data1[:,0],data2[:,0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def checkQuatReal():
    # vicon数据是x y z w
    a = np.asarray([-0.019602, 0.003027, 0.765291, 0.643378])
    print(math.acos(a[3]))
    b = np.sqrt(np.sum(np.square(a[:3])))
    print(math.asin(b))

    # print(math.acos(a[0]))
    # b= np.sqrt(np.sum(np.square(a[1:])))
    # print(math.asin(b))

    # 返回欧拉角也是zyx顺序
    euler = Rotation.from_quat(a).as_euler("zyx") / math.pi * 180.0
    print(euler)
    # dcm = Rotation.from_quat(a).as_dcm()
    # print(dcm)


if __name__ == "__main__":
    # checkQuatReal()
    main()
