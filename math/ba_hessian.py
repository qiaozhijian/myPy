import numpy as np

def get_hessian():
    n = 7 # 总变量数
    l = 4 # 路标点数
    m = n - l # 相机位姿数
    """
    n = [l, m]
    """
    # 1. 构建雅克比矩阵
    J1 = np.array([1, 0, 0, 0, 0, 1, 0]).reshape(1, n)
    J2 = np.array([0, 1, 0, 0, 1, 0, 0]).reshape(1, n)
    J3 = np.array([0, 0, 1, 0, 0, 1, 0]).reshape(1, n)
    J4 = np.array([0, 0, 0, 1, 0, 0, 1]).reshape(1, n)
    J5 = np.array([0, 0, 0, 0, 1, 1, 0]).reshape(1, n)
    J6 = np.array([0, 0, 0, 0, 0, 1, 1]).reshape(1, n)
    J7 = np.array([0, 0, 0, 0, 1, 0, 0]).reshape(1, n)

    J = np.concatenate((J1, J2, J3, J4, J5, J6, J7), axis=0)
    
    cov = np.diag(np.random.rand(n))
    cov_inv = np.linalg.inv(cov)
    H = J.T @ cov_inv @ J

    print(H.shape)
    print(H != 0)
    return H

if __name__ == "__main__":
    H = get_hessian()
