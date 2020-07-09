import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

a=np.random.rand(2,200)
print(len(a.shape))
point=np.linspace(0,1,1000)
# plt.scatter(point,np.zeros(1000),s=2,c=np.linspace(0,1,1000),cmap='Blues')
plt.scatter(a[0],a[1],s=2)
plt.show()

