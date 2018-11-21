import numpy as np

fFocal_msra = 241.42


def point_cloud(data, point_num):

    header = data['header']
    depth = data['depth']
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
            (w_matrix + bb_left - (img_width / 2)), depth[h, :]) / fFocal_msra

    for w in range(bb_width):
        idx = [(hi * bb_width + w) for hi in range(bb_height)]
        # '-' get on the right position
        hand_3d[idx, 1] = -np.multiply(
            (h_matrix+bb_top - (img_height / 2)), depth[:, w]) / fFocal_msra
    # drop the useless point
    # valid_idx = []
    # for num in range(valid_pixel_num):
    #     if any(hand_3d[:, num]):
    #         valid_idx.append(num)

    # hand_points = hand_3d[:, valid_idx]
    point_clouds = set_length(hand_3d, point_num)

    return hand_3d, point_clouds


def gt_pca(pc):
    pass


def set_length(data, point_num):
    "set the point number"
    point_shape = data.shape[0]

    if point_shape < point_num:
        rand_idx = np.arange(0, point_num, 1, dtype=np.int32)
        rand_idx[point_shape:] = np.random.randint(0, point_shape,
                                                   size=point_num - point_shape)
    else:
        rand_idx = np.random.randint(0, point_shape, size=point_num)

    point_cloud = data[:, rand_idx]

    return point_cloud
