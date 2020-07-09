import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import matplotlib.pyplot as plt

def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s
re=[]
x=np.arange(-3,3,0.1).squeeze()
for i in x:
    y= softmax(np.asarray([i,0]).reshape((1,-1)))[0,0]
    re.append(y)
re=np.asarray(re).squeeze()
print(re.shape,x.shape)
plt.plot(x,re)
plt.show()