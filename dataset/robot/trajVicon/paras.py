import numpy as np

T_c0_robot = np.asarray([0.9998199476005879, 0.01896351998860963, 0.0006762319126025606, 0.04216179343179696,
                         0.0002432940438150989, 0.02282305763279319, -0.9997394904915473, 0.08578733170877284,
                         -0.01897401349125333, 0.9995596495206938, 0.02281433457504149, -0.06909713473004092,
                         0, 0, 0, 1]).reshape(4, 4)
# T_imu_c0 = np.asarray([-0.01404787, -0.00799913, -0.99986933, -0.01712035, 0.9998648, 0.00843394, -0.01411528, 0.10571385, 0.00854575,
#      -0.99993244, 0.00787957, 0.11432723, 0., 0., 0., 1.]).reshape(4, 4)
T_imu_c0=np.asarray([0.0, -0.0, -1, -0.005,1, 0, 0.0, -0.06,0, -1, 0.0, 0.13,0, 0, 0, 1]).reshape(4,4)

T_odo_imu = np.asarray([-1, 0, 0, 0.133, 0, -1, 0, 0.0, 0, 0, 1, 0.02564, 0., 0., 0., 1.]).reshape(4, 4)

T_robot_c0 = np.linalg.inv(T_c0_robot)
T_odo_robot = np.dot(T_odo_imu, np.dot(T_imu_c0, T_c0_robot))
T_robot_odo = np.linalg.inv(T_odo_robot)