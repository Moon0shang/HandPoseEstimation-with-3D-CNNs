import numpy as np


""" 
def set_m(resolution=32):

    return resolution
 """


def tsdf(hand_points):


def z_TSDF(hand_points):

    M = 32
    fFocal = 242.1
    voxel_center = [160, 120, 0]

    x = hand_points[:, 0]
    y = hand_points[:, 1]
    z = hand_points[:, 2]

    point_max = np.array([np.max(x), np.max(y), np.max(z)], dtype=np.float32)
    point_min = np.array([np.min(x), np.min(y), np.min(z)], dtype=np.float32)
    point_mid = (point_min + point_max) / 2
    len_e = point_max - point_min
    max_l = np.max(len_e)
    voxel_len = max_l / M
    truncation = voxel_len * 3
    vox_ori = point_mid-max_l/2+voxel_len/2
    tsdf_v = np.ones((M, M, M))

    signed_distance = np.linalg.norm(hand_points - voxel_center, axis=1)

    for zi in range(32):
        for yi in range(32):
            for xi in range(32):
                vox_center = vox_ori + \
                    np.array([xi * voxel_len, yi * voxel_len, zi * voxel_len])
                vox_depth = -vox_center[2]
                q = fFocal / vox_depth
                pix_x = int(round(vox_center[0] * q + 160))
                pix_y = int(round(-vox_center[1] * q + 120))
                if pix_x < l or pix_x >= r or pix_y < t or pix_y >= b:
                    continue
                idx = (pix_y-t)*b_w+pix_x-l
                # print vox_center,x,y,z,pix_x,pix_y,idx
                pix_depth = s['data'][idx]
                if pix_depth == 0:
                    continue
                diff = (pix_depth - vox_depth)/truncation
                if diff >= 1 or diff <= -1:
                    continue
                tsdf_v[z, y, x] = diff

    return tsdf_v


"""     l_x = np.max(x) - np.min(x)
    l_y = np.max(y) - np.min(y)
    l_z = np.max(z) - np.min(z)

    l_voxel = max(l_x, l_y, l_z) / M
    truncation = 3*l_voxel

    volume = np.zeros((M, M, M)) """
