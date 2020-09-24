import numpy as np
import yaml
import cv2
from scipy.spatial.transform import Rotation
import math
import matplotlib.pyplot as plt
import os

def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

def kitti2tum(kitti):
    dcm = kitti[:3,:3]
    t = kitti[:3,3].tolist()
    q = Rotation.from_dcm(dcm).as_quat()
    tum = list(flat([t,q.tolist()]))
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

def se_err(pose1, pose2):
    T1 = tum2kitti(pose1)
    T2 = tum2kitti(pose2)

    deltaT = np.dot(np.linalg.inv(T1),T2)
    deltaTum = kitti2tum(deltaT)

    t = deltaTum[:3]
    q = deltaTum[3:]
    euler = Rotation.from_quat(q).as_euler("zyx") / math.pi * 180.0
    
    return t, euler

if __name__ == "__main__":
    # [  3.89254942 -14.87413043  -0.39500856] [-1.86403593  0.13357317  0.28846216] gt
    # [-0.20786871  0.03017495  0.03170735] [-0.30888754  4.21099767 -0.5630786 ] orb
    # [-0.14751887  0.09579507  0.04550303] [0.03512486 5.2488449  0.53295697] plo
    # [-0.11194196  0.12289331  0.05768358] [ 1.90013484  4.8721695  -0.19431023] po
    poseGt = [0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,1.000000000]
    pose1 = [0.131265208,-0.120222114,-0.049600281,-0.009651445,-0.039765373,0.000532430,0.999162257]
    t, euler = se_err(pose1,poseGt)
    print(t, euler)
    t= [  3.89254942,-14.87413043, -0.39500856]
    euler=[-1.86403593,0.13357317,0.28846216]
    t = np.asarray(t)/1000.0
    euler = np.asarray(euler)
    print(np.sqrt(np.sum(np.square(t))),np.sqrt(np.sum(np.square(euler))))
    t=[-0.20786871,0.03017495,0.03170735]
    euler=[-0.30888754,4.21099767,-0.5630786]
    t = np.asarray(t)
    euler = np.asarray(euler)
    print(np.sqrt(np.sum(np.square(t))),np.sqrt(np.sum(np.square(euler))))
    t=[-0.12668887,0.11927983,0.06218]
    euler=[-0.10527323,4.55717357,1.11105581]
    t = np.asarray(t)
    euler = np.asarray(euler)
    print(np.sqrt(np.sum(np.square(t))),np.sqrt(np.sum(np.square(euler))))
    t=[-0.11194196,0.12289331,0.05768358]
    euler=[1.90013484,4.8721695,-0.19431023]
    t = np.asarray(t)
    euler = np.asarray(euler)
    print(np.sqrt(np.sum(np.square(t))),np.sqrt(np.sum(np.square(euler))))