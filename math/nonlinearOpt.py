from scipy.optimize import minimize
import numpy as np
from scipy.spatial.transform import Rotation
from itertools import combinations,permutations
import itertools
np.set_printoptions(linewidth=1000)
isUseSe3=True
def antisym(vec):
    v1=vec[0]
    v2=vec[1]
    v3=vec[2]
    return np.asarray([0,-v3,v2,v3,0,-v1,-v2,v1,0]).reshape(3,3)
def se3FromRt(R,t,isOpted=True):
    r=Rotation.from_dcm(R)
    rotvec = r.as_rotvec()
    theta=np.linalg.norm(rotvec)
    axis=np.divide(rotvec,theta)
    J=np.sin(theta)/theta*np.eye(3)+(1-np.sin(theta)/theta)*axis.dot(axis.T)+(1-np.cos(theta))/theta*antisym(axis)

    p=np.linalg.inv(J).dot(t)
    se3=[np.asarray([rotvec[0],rotvec[0],rotvec[0],p[0,0],p[1,0],p[2,0]]),isOpted]

    return se3

def RtFromse3(se3):
    rotvec=np.asarray(se3[0:3])
    r=Rotation.from_rotvec(rotvec)

    p=np.asarray(se3[3:6])
    theta=np.linalg.norm(rotvec)
    axis=np.divide(rotvec,theta)

    J=np.sin(theta)/theta*np.eye(3)+(1-np.sin(theta)/theta)*axis.dot(axis.T)+(1-np.cos(theta))/theta*antisym(axis)
    t=J.dot(p)
    return r.as_dcm(),t

def vtFromRt(R,t,isOpted=True):
    r=Rotation.from_dcm(R)
    rotvec = r.as_rotvec()
    vt=[np.asarray([rotvec[0],rotvec[0],rotvec[0],t[0,0],t[1,0],t[2,0]]),isOpted]
    return vt
def RtFromVt(vt):
    rotvec=np.asarray(vt[0:3])
    r=Rotation.from_rotvec(rotvec)
    t=np.asarray(vt[3:6])
    return r.as_dcm(),t
def rotationError(pose_error):
    a = pose_error[0,0]
    b = pose_error[1,1]
    c = pose_error[2,2]
    d = 0.5*(a+b+c-1.0)
    return np.arccos(max(min(d,1.0),-1.0))

def translationError(pose_error):
    dx = pose_error[0,3]
    dy = pose_error[1,3]
    dz = pose_error[2,3]
    return np.sqrt(dx**2+dy**2+dz**2)

def getRelaPose(pose1, pose2):
    if isUseSe3:
        R1, t1 = RtFromse3(pose1[0])
    else:
        R1, t1 = RtFromVt(pose1[0])

    T1 = np.eye(4);
    T1[0:3, 0:3] = R1;
    T1[0:3, 3] = t1

    if isUseSe3:
        R2, t2 = RtFromse3(pose2[0])
    else:
        R2, t2 = RtFromVt(pose2[0])
    T2 = np.eye(4);
    T2[0:3, 0:3] = R2;
    T2[0:3, 3] = t2

    pose_error = np.dot(T1, np.linalg.inv(T2))

    return pose_error
def getPoseLoss(pose1,pose2):
    pose_error = np.dot(np.linalg.inv(pose1), pose2)

    r_err = rotationError(pose_error)
    t_err = translationError(pose_error)

    return r_err+0.1*t_err
def smallDisturb():
    T2 = np.eye(4);
    r1=Rotation.from_euler('zyx', np.random.rand(3)*0.1, degrees=True)
    T2[0:3, 0:3] = r1.as_dcm()
    T2[0:3, 3] = np.random.rand(3)*0.2
    return T2


def getGraphEdges(pose,isOpted=False,isDisturbed=False):
    pose_key = [k for k, v in pose.items()]
    c = list(combinations(pose_key, 2))

    relaPose = {}
    for framePair in c:
        if isOpted:
            if pose[framePair[0]][1] or pose[framePair[1]][1]:
                if isDisturbed:
                    relaPose[framePair] = getRelaPose(pose[framePair[0]], pose[framePair[1]]).dot(smallDisturb())
                else:
                    relaPose[framePair] = getRelaPose(pose[framePair[0]], pose[framePair[1]])
        else:
            if isDisturbed:
                relaPose[framePair] = getRelaPose(pose[framePair[0]], pose[framePair[1]]).dot(smallDisturb())
            else:
                relaPose[framePair] = getRelaPose(pose[framePair[0]], pose[framePair[1]])
    return relaPose

def fun(relaPose,notOptPose,poseIsOpted):
    def v(optPose_float):
        pose={}
        i=0
        for key,v in poseIsOpted.items():
            # 如果是优化量
            if not v:
                pose[key] = notOptPose[key]
            else:
                pose[key] = [optPose_float[i:i+6],True]
                i=i+6
        relaPose2=getGraphEdges(pose,isDisturbed=False,isOpted=True)

        total_loss=0;edges=0
        for framePair in relaPose2.keys():
            edges=edges+1
            loss=getPoseLoss(relaPose2[framePair],relaPose[framePair])
            total_loss+=loss
        aver_loss=total_loss/edges
        return aver_loss
    return v

def poseGraph(pose,relaPose):
    notOptPose={}
    poseIsOpted={}
    optPose_float=[]
    for pose_key,poseT in pose.items():
        poseIsOpted[pose_key]=poseT[1]
        if poseT[1]:
            optPose_float.append(poseT[0])
        else:
            notOptPose[pose_key]=poseT

    res = minimize(fun(relaPose,notOptPose,poseIsOpted), np.asarray(optPose_float).reshape(1,-1), method='SLSQP')
    print(res.fun)
    print(res.success)
    Opted_float=res.x
    Opted={};i=0
    for pose_key,poseT in pose.items():
        if poseT[1]:
            Opted[pose_key] = [Opted_float[i:i+6],True]
            i=i+6
    all_opted = notOptPose.copy()
    all_opted.update(Opted)
    all_edges=getGraphEdges(all_opted,isOpted=False,isDisturbed=False)
    return all_opted,all_edges
if __name__ == "__main__":
    pose={};i=0
    for euler in range(10,100,25):
        i=i+2
        r1=Rotation.from_euler('zyx', [euler, euler, euler], degrees=True)
        t=np.asarray([euler,euler*0.5,euler*0.1]).reshape(3,1)
        if isUseSe3:
            pose[i] = se3FromRt(r1.as_dcm(),t,isOpted=True)
        else:
            pose[i] = vtFromRt(r1.as_dcm(), t, isOpted=True)
    relaPose=getGraphEdges(pose,isOpted=False,isDisturbed=True)

    all_opted,all_edges=poseGraph(pose,relaPose)
    print(pose)
    print(all_opted)