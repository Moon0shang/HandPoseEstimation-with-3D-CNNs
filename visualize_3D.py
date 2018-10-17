import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

hand_points_raw = sio.loadmat('./results/P0/1/hand_points.mat')
hand_points = hand_points_raw['hand_points']


def visualize(hand_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        hand_points[:, 0],
        hand_points[:, 1],
        hand_points[:, 2]

    )
    ax.axis('scaled')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


show_3d = visualize(hand_points)
