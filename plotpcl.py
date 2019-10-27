#-*-coding:utf-8-*-
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
pointclouds_pl = tf.placeholder(tf.float32, shape=(32, 1024, 3))
w = tf.Variable(tf.random_normal([1,20,3],mean=1,stddev=2,dtype=tf.float32))
sess = tf.Session()
#初始化所有变量
init = tf.initialize_all_variables()
sess.run(init)
# print(w.eval(session=sess))
sess2=tf.Session()
sess2.run(init)
# print(w.eval(session=sess2))
# print(type(w.eval(session=sess2)[0,0,:]))

x=w.eval(session=sess2)[0,:,0]
y=w.eval(session=sess2)[0,:,1]
z=w.eval(session=sess2)[0,:,2]
#开始绘图
fig=plt.figure(dpi=120)
ax=fig.add_subplot(111,projection='3d')
#标题
plt.title('point cloud')
#利用xyz的值，生成每个点的相应坐标（x,y,z）
ax.scatter(x,y,z,c='b',marker='.',s=8,linewidth=0,alpha=1,cmap='spectral')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.scatter(x,y,z,c='b',marker='.',s=8,linewidth=0,alpha=1,cmap='spectral')
#显示
plt.show()