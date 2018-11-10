import os
import os.path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DB_dir = '/home/x/DB/MSRA HandPoseDataset/cvpr15_MSRAHandGestureDB/P0/1/000001_depth.bin'

# a = os.listdir(DB_dir)
# b = sorted(a)
# c = os.listdir(os.path.join(DB_dir, a[0]))
# d = sorted(c)
# print(c)
# print(d)
a = np.arange(0, 5, 1)
b = np.arange(1, 6, 1)
c = np.empty([2, 5])
c[1] = a
c[0] = b
d = np.empty([3, 2, 5])
d[0] = c
d[1] = c
d[2] = c
print(d)
