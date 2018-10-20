import numpy as np


""" 
def set_m(resolution=32):

    return resolution
 """


def tsdf(depth_ori, pic_info):

    voxel_res = 32
    [fFocal_msra, img_height, img_width, bb_top, bb_bottom,
        bb_left, bb_right, bb_width, bb_height] = pic_info
    point_max_min = max_min_point(depth_ori, pic_info)
    # 不知道用途，也许可以删除
    if any(point_max_min):
        point_max_min = point_max_min[:-1]

    point_max = np.max(point_max_min[:, 3:6], axis=0)
    point_min = np.min(point_max_min[:, :3], axis=0)
    point_mid = (point_max + point_min) / 2
    len_pixel = point_max - point_min
    max_lenth = np.max(len_pixel)
    voxel_len = max_lenth / voxel_res
    truncation = voxel_len * 3
    vox_ori = point_mid - max_lenth / 2 + voxel_len / 2

    # tsdf calculation
    tsdf = tsdf_cal(depth_ori, pic_info, vox_ori, voxel_len, truncation)


def max_min_point(hand_points, pic_info):

    [fFocal_msra, img_height, img_width, bb_top, bb_bottom,
        bb_left, bb_right, bb_width, bb_height] = pic_info

    x = hand_points[:, 0]
    y = hand_points[:, 1]
    z = hand_points[:, 2]

    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    z_max = np.max(z)
    z_min = np.min(z)
    point_max_min = np.array([x_min, y_min, z_min, x_max, y_max, z_max])

    return point_max_min


def tsdf_cal(depth_ori, pic_info, vox_ori, voxel_len, truncation):

    [fFocal_msra, img_height, img_width, bb_top, bb_bottom,
        bb_left, bb_right, bb_width, bb_height] = pic_info

    x_center = 160
    y_center = 120
    volume_size = 32 * 32 * 32

    for z in range(32):
        for y in range(32):
            for x in range(32):
                voxel_idx = x + y * 32 + z * 32 * 32
                if voxel_idx >= volume_size:
                    print('out of range')
                    return
                # voxel center
                voxel_x = vox_ori[0] + x * voxel_len
                voxel_y = vox_ori[1] + y * voxel_len
                voxel_z = vox_ori[2] + z * voxel_len

                # voxel center in image frame
                coeff = -fFocal_msra / voxel_z
                pixel_x = int((voxel_x * coeff + x_center))
                pixel_y = int((voxel_y * coeff + y_center))
                # need to modify
                tsdf_v = np.zeros(3, z, y, x)  # not 100
                if pixel_x < bb_left or pixel_x >= bb_right or pixel_y < bb_top or pixel_y >= bb_bottom:
                    print('out of valid area')
                    return
                idx = (pixel_y - bb_top) * bb_width + pixel_x - bb_left
                pixel_depth = depth_ori[idx]
                if abs(pixel_depth) < 1:
                    print('wrong depth')
                    return
                # closest surface point in world frame
                coeff1 = pixel_depth / fFocal_msra
                world_x = (pixel_x - x_center) * coeff1
                world_y = (pixel_y - y_center) * coeff1
                world_z = -pixel_depth
                tsdf_x = abs(voxel_x - world_x) / truncation
                tsdf_y = abs(voxel_y - world_y) / truncation
                tsdf_z = abs(voxel_z - world_z) / truncation
                dis_to_sur_min = pow(tsdf_x * tsdf_x + tsdf_y *
                                     tsdf_y + tsdf_z * tsdf_z, 0.5)
                if dis_to_sur_min > 1:
                    tsdf_x = 1
                    tsdf_y = 1
                    tsdf_z = 1

                tsdf_x = min(tsdf_x, 1)
                tsdf_y = min(tsdf_y, 1)
                tsdf_z = min(tsdf_z, 1)

                if world_z > voxel_z:
                    tsdf_x = - tsdf_x
                    tsdf_y = - tsdf_y
                    tsdf_z = - tsdf_z

                tsdf_v[0, z, y, x] = tsdf_x
                tsdf_v[1, z, y, x] = tsdf_y
                tsdf_v[2, z, y, x] = tsdf_z

                return tsdf_v


""" 
def z_TSDF(hand_points,pic_info):

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
"""

"""     l_x = np.max(x) - np.min(x)
    l_y = np.max(y) - np.min(y)
    l_z = np.max(z) - np.min(z)

    l_voxel = max(l_x, l_y, l_z) / M
    truncation = 3*l_voxel

    volume = np.zeros((M, M, M)) """
