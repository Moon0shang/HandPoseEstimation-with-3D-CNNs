import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize(self, hand_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        hand_points[:, :, 0],
        hand_points[:, :, 1],
        hand_points[:, :, 2],
        marker='.',
        s=2,
        linewidth=0,
        alpha=1,
        cmap='spectral'
    )
    ax.axis('scaled')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
