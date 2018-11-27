import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

joints = np.load('./1.npy')


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for j in range(2):
    sj = joints[j]
    plt.cla()
    ax.scatter(sj[:, 0], sj[:, 1], -sj[:, 2], marker='o', color='r')
    # The 21 hand joints are: wrist,
    # index_mcp, index_pip, index_dip, index_tip,
    # middle_mcp, middle_pip, middle_dip, middle_tip,
    # ring_mcp, ring_pip, ring_dip, ring_tip,
    # little_mcp, little_pip, little_dip, little_tip,
    #  thumb_mcp, thumb_pip, thumb_dip, thumb_tip.
    base = np.arange(5)
    c = ['b', 'y', 'c', 'm', 'g']  # b,g,r,c,m,y,k,w
    for i in range(5):
        f_idx = base + 4 * i
        f_idx[0] = 0
        ax.plot(sj[f_idx, 0], sj[f_idx, 1], -
                sj[f_idx, 2], color=c[i], linestyle='--', linewidth=0.8)

    ax.axis('square')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.pause(0.5)

plt.ioff()
plt.show()
