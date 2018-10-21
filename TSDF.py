import numpy as np
import scipy.io as sio

""" 
def set_m(resolution=32):

    return resolution
 """


def tsdf(hand_points, hand_ori, pic_info):

    voxel_res = 32
    """ 
    [fFocal_msra, img_height, img_width, bb_top, bb_bottom,
        bb_left, bb_right, bb_width, bb_height] = pic_info """
    # blockdim = (hand_ori.size+128-1)//128
    depth_ori = hand_ori[:, 2]
    point_max, point_min = max_min_point(hand_ori)

    # 不知道用途，也许可以删除
    # if any(point_max_min):
    # point_max_min = point_max_min[:-1] """

    # point_max = np.max(point_max_min[:, 3:6], axis=0)
    # point_min = np.min(point_max_min[:, :3], axis=0)

    point_mid = (point_max + point_min) / 2
    len_pixel = point_max - point_min
    max_lenth = np.max(len_pixel)
    voxel_len = max_lenth / voxel_res
    truncation = voxel_len * 3
    vox_ori = point_mid - max_lenth / 2 + voxel_len / 2
    # tsdf calculation

    tsdf_v = tsdf_cal(depth_ori, pic_info, vox_ori,
                      voxel_len, truncation, point_mid)

    sio.savemat('./tsdf/tsdf.mat', {'tsdf': tsdf_v})

    return tsdf_v

# need to be rewrite


def max_min_point(hand_ori):

    # trying
    x = hand_ori[:, 0]
    y = hand_ori[:, 1]
    z = hand_ori[:, 2]
    # 防止出现max(z)= 0 的情况
    z = z[z != 0]

    point_max = np.empty([3], dtype=np.float32)
    point_min = np.empty([3], dtype=np.float32)

    point_max[0] = np.max(x)
    point_max[1] = np.max(y)
    point_max[2] = np.max(z)
    point_min[0] = np.min(x)
    point_min[1] = np.min(y)
    point_min[2] = np.min(z)

    return point_max, point_min


""" 
wrong method that takes too much time, don't use it!
def max_min_point(blockdim, hand_ori, pic_info):

    [fFocal_msra, img_height, img_width, bb_top, bb_bottom,
        bb_left, bb_right, bb_width, bb_height] = pic_info

    # mm_threadperblock = 128
    center_x = 160
    center_y = 120
    point_max_min = np.empty([blockdim, 6], dtype=np.float32)

    point_min = np.empty((128, 3), dtype=np.float32)
    point_max = np.empty((128, 3), dtype=np.float32)
    # '''此处需要一个循环'''
    for bid in range(blockdim):
        for tid in range(128):
            # cuda.grid = threadidx.x+blockidx.x*blockdim.x
            # threadidx.x =0~127; blockidx.x=0~blockdim-1
            # blockdim.x=128
            pos = tid+bid*128
            x = pos % bb_width + bb_left
            y = pos / bb_width + bb_top
            cam_z = hand_ori[pos]
            if abs(cam_z) < 1:
                point_min[tid][0], point_min[tid][1], point_min[tid][2] = 9999, 9999, 9999
                point_max[tid][0], point_max[tid][1], point_max[tid][2] = - \
                    9999, -9999, -9999
            else:
                coeff = cam_z / fFocal_msra
                cam_x = coeff * (x - center_x)
                cam_y = coeff * (y - center_y)
                point_min[tid][0], point_min[tid][1], point_min[tid][2] = cam_x, cam_y, cam_z
                point_max[tid][0], point_max[tid][1], point_max[tid][2] = cam_x, cam_y, cam_z

            s = 1
            while s < 128:
                index = 2 * s * tid
                if index + s < 128:
                    point_min[index][0] = min(
                        point_min[index + s][0], point_min[index][0])
                    point_min[index][1] = min(
                        point_min[index + s][1], point_min[index][1])
                    point_min[index][2] = min(
                        point_min[index + s][2], point_min[index][2])
                    point_max[index][0] = min(
                        point_max[index + s][0], point_max[index][0])
                    point_max[index][1] = min(
                        point_max[index + s][1], point_max[index][1])
                    point_max[index][2] = min(
                        point_max[index+s][2], point_max[index][2])

                s *= 2

            if tid == 0:
                point_max_min[bid][0] = point_min[0][0]
                point_max_min[bid][1] = point_min[0][1]
                point_max_min[bid][2] = point_min[0][2]
                point_max_min[bid][3] = point_max[0][0]
                point_max_min[bid][4] = point_max[0][1]
                point_max_min[bid][5] = point_max[0][2]

    return point_max_min

 """
""" 
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
 """


def tsdf_cal(depth_ori, pic_info, vox_ori, voxel_len, truncation, point_mid):

    [fFocal_msra, img_height, img_width, bb_top, bb_bottom,
        bb_left, bb_right, bb_width, bb_height] = pic_info

    x_center = 160
    y_center = 120
    volume_size = 32 * 32 * 32

    tsdf_v = np.empty([3, 32, 32, 32])
    # sdf = np.empty([32, 32, 32])
    for x in range(32):
        for y in range(32):
            for z in range(32):

                voxel_idx = x + y * 32 + z * 32 * 32
                if voxel_idx >= volume_size:
                    # print('out of range')
                    continue

                # voxel center
                voxel_x = vox_ori[0] + x * voxel_len
                voxel_y = vox_ori[1] + y * voxel_len
                voxel_z = vox_ori[2] + z * voxel_len

                # voxel center in image frame
                coeff = -fFocal_msra / voxel_z
                pixel_x = int((voxel_x * coeff + x_center))
                pixel_y = int((-voxel_y * coeff + y_center))
                '''need to modify'''

                if pixel_x < bb_left or pixel_x >= bb_right or pixel_y < bb_top or pixel_y >= bb_bottom:
                    # print('out of valid area')
                    continue

                idx = int((pixel_y - bb_top) * bb_width + pixel_x - bb_left)
                pixel_depth = depth_ori[idx]

                if abs(pixel_depth) < 1:
                    # print('wrong depth')
                    continue

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
