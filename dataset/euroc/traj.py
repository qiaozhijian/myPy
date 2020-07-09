from scipy.spatial.transform import Rotation
import numpy as np
import os
# 不太行啊，不知道怎么转换的


# file_name = "orbstereo_102.txt"
file_name = "vicon101easy.txt"
# file_name = "orbstereo_101.txt"
# file_name = "orbstereo_216_vo.txt"
# file_name = "orbmono_101.txt"
file_write = "new.txt"
gt_file = "v101.txt"
# gt_file = "V102.txt"
# gt_file = "vicon101easy.csv"
# gt_file = "orbstereo_216_slam.txt"
f = open(file_name, 'r')
w = open(file_write, 'w')
s = f.readlines()
f.close()

T_bv = np.asarray([0.33638, -0.01749, 0.94156, 0.06901,
                   -0.02078, -0.99972, -0.01114, -0.02781,
                   0.94150, -0.01582, -0.33665, -0.12395,
                   0.0, 0.0, 0.0, 1.0]).reshape((4, 4))
T_bc = np.asarray([0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                   0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                   -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                   0.0, 0.0, 0.0, 1.0]).reshape((4, 4))
T_vc=np.dot(np.linalg.inv(T_bv), T_bc)


gt = [8.786119999999999486e-01, 2.142469999999999875e+00, 9.472620000000000484e-01]
gq = [-8.284589999999999455e-01, -5.895600000000000146e-02, -5.536410000000000498e-01, 6.051399999999999835e-02]

traj = []
timeStamps = []
for cnt, line in enumerate(s):
    a = line.split()
    timeStamps.append(float(line.split()[0]))
    P = np.eye(4)
    line_split = [float(i) for i in line.split()]
    t = line_split[1:4]
    q = line_split[4:]
    rotation = Rotation.from_quat(q)
    r = rotation.as_dcm()
    T = np.eye(4)
    T[0:3, 0:3] = r
    T[0:3, 3] = t
    # print(T)
    # T = np.dot(np.dot(np.linalg.inv(T_vc), T), T_vc)
    # T = np.dot(np.dot(np.linalg.inv(T_bv), T_bc), T)
    # T = np.dot(T_bv, T)
    traj.append(T)
traj = np.asarray(traj).reshape((-1, 4, 4))

# gT = np.eye(4)
# gr = Rotation.from_quat(gq).as_dcm()
# gT[0:3, 0:3] = gr
# gT[0:3, 3] = gt
# deltaT =  np.dot(gT,np.linalg.inv(traj[0]))
# # print(deltaT)
# delta_t=gt - traj[0][0:3, 3]
# deltaT = np.zeros((4,4))
# deltaT[0:3, 3] = delta_t
# deltaT = np.eye(4)
for i in range(traj.shape[0]):
    # T = np.dot(deltaT, traj[i])
    # T = deltaT + traj[i]
    T = traj[i]
    t = T[0:3, 3]
    q = Rotation.from_dcm(T[0:3, 0:3]).as_quat()
    ss = str(timeStamps[i])
    for i in range(3):
        ss = ss + " " + str(t[i])
    for i in range(4):
        ss = ss + " " + str(q[i])
    w.write(ss + '\n')
w.close()

os.system("evo_traj tum " + file_write + " --ref "+ gt_file+ " -p")
os.system("evo_ape tum "+ gt_file+ " " + file_write + " --plot")
# os.system("evo_traj tum " + file_name + " --ref "+ gt_file+ " --plot_mode xyz -p ")
# os.system("evo_ape tum "+ gt_file+ " " + file_name + " -r trans_part -a --align_origin")


# T0 = np.asarray([0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
#          0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
#         -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
#          0.0, 0.0, 0.0, 1.0]).reshape((4, 4))
# T1 = np.asarray([0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
#          0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
#         -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
#          0.0, 0.0, 0.0, 1.0]).reshape((4, 4))
# print(np.linalg.inv(T0).dot(T1))
# print(np.linalg.inv(T1).dot(T0))
# print(1.10074138e-01*435.654)
# baseline = np.sqrt(np.sum(np.square(np.asarray([-114.1875516864081,-0.1165879437668266,-18.24557290425103])/1000)))
# print(baseline)
# print(baseline*735.215354396876)

# evo_traj tum new.txt --ref data.tum --plot_mode xyz -p
# evo_ape tum data.tum new.txt --plot_full_ref -a --align_origin -p -r angle_deg
# evo_ape tum data.tum new.txt -r angle_deg -a --align_origin
