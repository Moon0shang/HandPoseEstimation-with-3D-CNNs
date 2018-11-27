import numpy as np


class DataProcess(object):

    def __init__(self, data, ground_truth, point_num=6000, aug=False):
        self.fFocal_msra = 241.42
        self.data = data
        self.ground_truth = ground_truth
        self.point_num = point_num
        self.aug = aug

    def process(self):

        hand_points = self.point_cloud()
        point_clouds = self.set_length(hand_points)
        tsdf, max_l, mid_p = self.tsdf_f(point_clouds)

        if self.aug:
            hand_aug, ground_truth_aug = self.data_aug(hand_points)
            point_clouds_aug = self.set_length(hand_aug)
            tsdf_aug, max_l_aug, mid_p_aug = self.tsdf_f(point_clouds_aug)
            data_pre = [point_clouds, tsdf, max_l, mid_p,
                        point_clouds_aug, tsdf_aug, max_l_aug, mid_p_aug, ground_truth_aug]
        else:
            data_pre = [point_clouds, tsdf, max_l, mid_p]

        return data_pre

    def point_cloud(self):
        "get point cloud from depth informations"
        header = self.data['header']
        depth = self.data['depth']
        valid_pixel_num = depth.size
        img_width = header[0]
        img_height = header[1]
        bb_left = header[2]
        bb_top = header[3]
        bb_right = header[4]
        bb_bottom = header[5]
        bb_height = bb_bottom - bb_top
        bb_width = bb_right - bb_left

        hand_3d = np.zeros((valid_pixel_num, 3))
        # '-' get on the right position
        hand_3d[:, 2] = - depth
        depth = depth.reshape(bb_height, bb_width)
        h_matrix = np.array([i for i in range(bb_height)], dtype=np.float32)
        w_matrix = np.array([i for i in range(bb_width)], dtype=np.float32)

        for h in range(bb_height):
            hand_3d[(h * bb_width):((h + 1) * bb_width), 0] = np.multiply(
                (w_matrix + bb_left - (img_width / 2)), depth[h, :]) / self.fFocal_msra

        for w in range(bb_width):
            idx = [(hi * bb_width + w) for hi in range(bb_height)]
            # '-' get on the right position
            hand_3d[idx, 1] = -np.multiply(
                (h_matrix+bb_top - (img_height / 2)), depth[:, w]) / self.fFocal_msra
            # drop the useless point
        valid_idx = []
        for num in range(hand_3d.shape[0]):
            if any(hand_3d[num, :]):
                valid_idx.append(num)

        hand_points = hand_3d[valid_idx, :]

        return hand_points

    def set_length(self, hand_points):
        "set the point number"
        point_num = self.point_num
        point_shape = hand_points.shape[0]

        if point_shape < point_num:
            rand_idx = np.arange(0, point_num, 1, dtype=np.int32)
            rand_idx[point_shape:] = np.random.randint(0, point_shape,
                                                       size=point_num - point_shape)
        else:
            rand_idx = np.random.randint(0, point_shape, size=point_num)

        point_cloud = hand_points[rand_idx, :]

        return point_cloud

    def tsdf_f(self, point_cloud):

        voxel_res = 32
        point_max, point_min = self.max_min_point(point_cloud)

        mid_point = (point_max + point_min) / 2
        len_pixel = point_max - point_min
        max_lenth = np.max(len_pixel)
        voxel_len = max_lenth / voxel_res
        truncation = voxel_len * 3
        vox_ori = mid_point - max_lenth / 2 + voxel_len / 2
        # tsdf calculation
        tsdf_v = self.tsdf_cal(vox_ori, voxel_len, truncation)

        return tsdf_v, max_lenth, mid_point

    def max_min_point(self, point_cloud):

        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
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

    def tsdf_cal(self, vox_ori, voxel_len, truncation):

        depth = self.data['depth']
        header = self.data['header']
        bb_left = header[2]
        bb_top = header[3]
        bb_right = header[4]
        bb_bottom = header[5]
        # bb_height = bb_bottom - bb_top
        bb_width = bb_right - bb_left

        x_center = 160
        y_center = 120
        volume_size = 32 * 32 * 32

        tsdf_v = np.zeros([3, 32, 32, 32])
        # tsdf_v1 = np.empty([3, 32, 32, 32])
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
                    coeff = -self.fFocal_msra / voxel_z
                    pixel_x = int(voxel_x * coeff + x_center)
                    pixel_y = int(-voxel_y * coeff + y_center)

                    if pixel_x < bb_left or pixel_x >= bb_right or pixel_y < bb_top or pixel_y >= bb_bottom:
                        # print('out of valid area')
                        continue

                    idx = int((pixel_y - bb_top) *
                              bb_width + pixel_x - bb_left)
                    pixel_depth = depth[idx]

                    if abs(pixel_depth) < 1:
                        # print('wrong depth')
                        continue

                    # closest surface point in world frame
                    coeff1 = pixel_depth / self.fFocal_msra
                    world_x = (pixel_x - x_center) * coeff1
                    world_y = -(pixel_y - y_center) * coeff1
                    world_z = -pixel_depth

                    tsdf_x = abs(voxel_x - world_x) / truncation
                    tsdf_y = abs(voxel_y - world_y) / truncation
                    tsdf_z = abs(voxel_z - world_z) / truncation
                    dis_to_sur_min = np.sqrt(
                        tsdf_x * tsdf_x + tsdf_y * tsdf_y + tsdf_z * tsdf_z)
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

                    tsdf_v[0, x, y, z] = tsdf_x
                    tsdf_v[1, x, y, z] = tsdf_y
                    tsdf_v[2, x, y, z] = tsdf_z

        return tsdf_v

    def data_aug(self, point_clouds):
        """
        data augmentation: contain stretch and rotation
        stretch factor: x, y: [2/3,3/2]; z: 1
        rotation factor: theta x,y: [-30d,30d]; theta z: [-180d,180d]
        """
        # stretch factor
        stretch_xy = np.random.uniform(2 / 3, 3 / 2)
        stretch = np.array([stretch_xy, stretch_xy, 1])
        # diag matrix
        S = np.diag(stretch)

        # rotate factor
        rot_xy = np.random.randint(-30, 30)
        rot_z = np.random.randint(-30, 30)
        # x, y matrix
        xy_c = np.cos(np.pi*rot_xy/180)
        xy_s = np.sin(np.pi*rot_xy/180)
        R_x = np.array([[1, 0, 0], [0, xy_c, xy_s], [0, -xy_s, xy_c]])
        R_y = np.array([[xy_c, 0, -xy_s], [0, 1, 0], [xy_s, 0, xy_c]])
        z_c = np.cos(rot_z)
        z_s = np.sin(rot_z)
        R_z = np.array([[xy_c, xy_s, 0], [-xy_s, xy_c, 0], [0, 0, 1]])
        # rotation matrix
        R = np.dot(R_x, R_y)
        R = np.dot(R, R_z)

        # ground truth augmentation
        # stretch
        joints = self.ground_truth(-1, 3)
        joint_stretch = np.dot(joints, S)

        # normalize the points to center point
        center_mean = np.mean(joint_stretch, axis=2)
        center_point = np.empty(
            [center_mean.shape[0], center_mean.shape[1], 3])
        center_point[:, :, 0] = center_mean
        center_point[:, :, 1] = center_mean
        center_point[:, :, 2] = center_mean
        coor_nor = joint_stretch - center_point
        # rotate
        joint_rot = np.dot(coor_nor, R)
        # unnormalize the points to coordinate system
        joint_coor = joint_rot + center_point
        ground_truth_aug = joint_coor.reshape(-1, 63)

        # point cloud augmentation
        pc_stretch = np.dot(point_clouds, S)
        pc_mean = np.mean(pc_stretch, axis=2)
        pc_center = np.empty([pc_mean.shape[0], pc_mean.shape[1], 3])
        pc_center[:, :, 0] = pc_mean
        pc_center[:, :, 1] = pc_mean
        pc_center[:, :, 2] = pc_mean
        pc_nor = pc_stretch-pc_center
        pc_rot = np.dot(pc_nor, R)
        point_clouds_aug = pc_rot+pc_center

        return point_clouds_aug, ground_truth_aug
