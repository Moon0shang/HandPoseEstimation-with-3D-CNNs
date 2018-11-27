import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_joints(joint, out):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joint[:, 0], joint[:, 1], joint[:, 2], marker='^', color='k')
    ax.scatter(out[:, 0], out[:, 1], out[:, 2], marker='o', color='r')
    base = np.arange(5)
    c = ['b', 'y', 'c', 'm', 'g']  # b,g,r,c,m,y,k,w
    ax.plot(joint[base, 0], joint[base, 1], joint[base, 2],
            color=c[0], linestyle='--', linewidth=0.8, label='origin')
    for i in range(1, 5):
        f_idx = base + 4 * i
        f_idx[0] = 0
        ax.plot(joint[f_idx, 0], joint[f_idx, 1], joint[f_idx, 2],
                color=c[i], linestyle='--', linewidth=0.8)

    ax.plot(out[base, 0], out[base, 1], out[base, 2],
            color=c[0], label='augment')
    for i in range(1, 5):
        f_idx = base + 4 * i
        f_idx[0] = 0
        ax.plot(out[f_idx, 0], out[f_idx, 1], out[f_idx, 2], color=c[i])

    ax.axis('equal')
    ax.set_title('ground truth compare')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    joints = np.load('./output/all/joint-1.npy')
    outs = np.load('./output/all/out-1.npy')
    kk = 0
    joint = joints[kk]
    out = outs[kk]
    visualize_joints(joint, out)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(joint[:, 0], joint[:, 1], joint[:, 2], marker='^', color='k')
# ax.scatter(out[:, 0], out[:, 1], out[:, 2], marker='o', color='r')
# base = np.arange(5)
# c = ['b', 'y', 'c', 'm', 'g']  # b,g,r,c,m,y,k,w
# for i in range(5):
#     f_idx = base + 4 * i
#     f_idx[0] = 0
#     ax.plot(joint[f_idx, 0], joint[f_idx, 1], joint[f_idx, 2],
#             color=c[i], linestyle='--', linewidth=0.8)

# for i in range(5):
#     f_idx = base + 4 * i
#     f_idx[0] = 0
#     ax.plot(out[f_idx, 0], out[f_idx, 1], out[f_idx, 2], color=c[i])

# ax.axis('scaled')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()


# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for j in range(2):
#     sj = joints[j]
#     plt.cla()
#     ax.scatter(sj[:, 0], sj[:, 1], -sj[:, 2], marker='o', color='r')
#     # The 21 hand joints are: wrist,
#     # index_mcp, index_pip, index_dip, index_tip,
#     # middle_mcp, middle_pip, middle_dip, middle_tip,
#     # ring_mcp, ring_pip, ring_dip, ring_tip,
#     # little_mcp, little_pip, little_dip, little_tip,
#     #  thumb_mcp, thumb_pip, thumb_dip, thumb_tip.
# base = np.arange(5)
# c = ['b', 'y', 'c', 'm', 'g']  # b,g,r,c,m,y,k,w
# for i in range(5):
#     f_idx = base + 4 * i
#     f_idx[0] = 0
#     ax.plot(sj[f_idx, 0], sj[f_idx, 1], -
#             sj[f_idx, 2], color=c[i], linestyle='--', linewidth=0.8)

# ax.axis('scaled')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.pause(0.5)

# plt.ioff()
# plt.show()
