import numpy as np
import yaml
import cv2
from scipy.spatial.transform import Rotation
import math

def transferMat(mat):
    mat = mat.flatten();
    matList = ""
    for i in mat:
        matList = matList + str(i)+", "
    return matList[:-2]

PATH = '/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/cali/camera_imu/camera_imu/result/1/stereo no time'
f = open(PATH + "/camchain-imucam-1.yaml",'r')
cont = f.read()
x = yaml.load(cont, Loader=yaml.FullLoader)
T_ic0 = np.linalg.inv(np.asarray(x['cam0']['T_cam_imu']))
T_ic1 = np.linalg.inv(np.asarray(x['cam1']['T_cam_imu']))
T_c0c1 = np.linalg.inv(np.asarray(x['cam1']['T_cn_cnm1']))
print(T_ic0)
T = Rotation.from_dcm(T_ic0[:3,:3])
euler = T.as_euler("xyz")*180.0/math.pi
print(euler)

euler_ex = np.asarray([-90,0,90])/180.0*math.pi
T_ic0[:3,:3] = Rotation.from_euler("xyz",euler_ex).as_dcm()
T_ic0[0,3] = -0.005
T_ic0[1,3] = -0.06
T_ic0[2,3] = 0.13
T_ic1 = np.dot(T_ic0,T_c0c1)

print(T_ic0)
print(T_ic1)

print((np.sum(np.dot(np.linalg.inv(T_ic0),T_ic1) - T_c0c1)<0.1) == True)

# print(T_c0i)
# print(T_c1i)
# print(T_c0c1)
# print(transferMat(T_c1i))


