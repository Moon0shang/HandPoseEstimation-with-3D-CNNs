import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize(hand_points):

    if len(hand_points.shape) == 2:
        x = hand_points[:, 0]
        y = hand_points[:, 1]
        z = hand_points[:, 2]
    elif len(hand_points.shape) == 3:
        x = None
        y = None
        z = None
    else:
        print('hand_points is not the right matrix!')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        x, y, z,
        marker='.'  # show as a point
    )
    ax.axis('scaled')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
