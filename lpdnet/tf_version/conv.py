import torch
import torch.nn.functional as F
# import config as cfg
import tensorflow as tf
# from torch.autograd import Variable
# import numpy as np
import loupe as lp
from lpd_FNSF import forward



if __name__ == '__main__':

    input = tf.zeros([2,16,4096,13])
    forward(input)