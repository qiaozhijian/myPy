from scipy.spatial.transform import Rotation
import numpy as np

t=[8.786119999999999486e-01,2.142469999999999875e+00,9.472620000000000484e-01]
q=[-8.284589999999999455e-01,-5.895600000000000146e-02,-5.536410000000000498e-01,6.051399999999999835e-02]

# t=[0.515356,1.996773,0.971104]
# q=[0.161996,0.789985,-0.205376,0.554528]

rotation = Rotation.from_quat(q)
euler=rotation.as_euler("zyx",degrees=True)
# print(euler)

T_bv=np.asarray([ 0.33638, -0.01749,  0.94156,  0.06901,
         -0.02078, -0.99972, -0.01114, -0.02781,
          0.94150, -0.01582, -0.33665, -0.12395,
              0.0,      0.0,      0.0,      1.0]).reshape((4,4))
T_bc=np.asarray([0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]).reshape((4,4))
T_bc2=np.asarray([0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
         0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
         0.0, 0.0, 0.0, 1.0]).reshape((4,4))

Tv=np.eye(4)
Tv[0:3,0:3]=rotation.as_dcm()
Tv[0:3,3]=t
print(Tv)

Tc=np.linalg.inv(T_bc).dot(T_bv).dot(Tv)
print(Tc)

rotation = Rotation.from_dcm(Tc[0:3,0:3])
euler=rotation.as_euler("zyx",degrees=True)
print(euler)

rotation = Rotation.from_euler("zyx",euler)
quat=rotation.as_quat()
print(quat)