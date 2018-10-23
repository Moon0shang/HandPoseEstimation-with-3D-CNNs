import numpy as np


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


def tsdf_cal(depth_ori, pic_info, vox_ori, voxel_len, truncation, point_mid):

    [fFocal_msra, img_height, img_width, bb_top, bb_bottom,
        bb_left, bb_right, bb_width, bb_height] = pic_info

    x_center = 160
    y_center = 120
    volume_size = 32 * 32 * 32

    tsdf_v = np.empty([3, 32, 32, 32])
    tsdf_v1 = np.empty([3, 32, 32, 32])
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

                tsdf_x = abs(voxel_x - world_x) / truncation-7
                tsdf_y = abs(voxel_y - world_y) / truncation-2
                tsdf_z = (abs(voxel_z - world_z) / truncation-43)/6  # (37-48)
                tsdf_v1[0, z, y, x] = tsdf_x
                tsdf_v1[1, z, y, x] = tsdf_y
                tsdf_v1[2, z, y, x] = tsdf_z
                dis_to_sur_min = pow(tsdf_x * tsdf_x + tsdf_y *
                                     tsdf_y + tsdf_z * tsdf_z, 0.5)
                if dis_to_sur_min > 1:
                    tsdf_x = 1
                    tsdf_y = 1
                    tsdf_z = 1

                tsdf_x = min(tsdf_x, 1)
                tsdf_y = min(tsdf_y, 1)
                tsdf_z = min(tsdf_z, 1)

                """  if world_z > voxel_z:
                    tsdf_x = - tsdf_x
                    tsdf_y = - tsdf_y
                    tsdf_z = - tsdf_z
                """
                tsdf_v[0, z, y, x] = tsdf_x
                tsdf_v[1, z, y, x] = tsdf_y
                tsdf_v[2, z, y, x] = tsdf_z

    return tsdf_v
