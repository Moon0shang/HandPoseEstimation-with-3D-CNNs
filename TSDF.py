import numpy as np


""" 
def set_m(resolution=32):

    return resolution
 """


def z_TSDF(hand_points):

    M = 32

    voxel_center = [160, 120, 0]

    x = hand_points[:, 0]
    y = hand_points[:, 1]
    z = hand_points[:, 2]

    l_x = np.max(x) - np.min(x)
    l_y = np.max(y) - np.min(y)
    l_z = np.max(z) - np.min(z)

    l_voxel = max(l_x, l_y, l_z) / M
    truncation = 3*l_voxel

    volume = np.zeros((M, M, M))

    signed_distance = np.linalg.norm(hand_points - voxel_center, axis=1)

    for xi in range(32):
        for yi in range(32):
            for zi in range(32):
                volume[xi, yi, zi] = np.min(np.max(signed_distance[] / truncation, -1), 1)
