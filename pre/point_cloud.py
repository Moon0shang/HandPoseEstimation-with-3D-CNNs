import numpy as np
import scipy.io as sio

from visualize import visualize

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

    hand_3d = np.zeros((3, valid_pixel_num))
    # '-' get on the right position
    hand_3d[2, :] = - depth
    depth = depth.reshape(bb_height, bb_width)
    h_matrix = np.array([i for i in range(bb_height)], dtype=np.float32)
    w_matrix = np.array([i for i in range(bb_width)], dtype=np.float32)

    for h in range(bb_height):
        hand_3d[0, (h * bb_width):((h + 1) * bb_width)] = np.multiply(
            (w_matrix + bb_left - (img_width / 2)), depth[h, :]) / fFocal_msra

    for w in range(bb_width):
        idx = [(hi * bb_width + w) for hi in range(bb_height)]
        # '-' get on the right position
        hand_3d[1, idx] = -np.multiply(
            (h_matrix+bb_top - (img_height / 2)), depth[:, w]) / fFocal_msra
    # drop the useless point
    # valid_idx = []
    # for num in range(valid_pixel_num):
    #     if any(hand_3d[:, num]):
    #         valid_idx.append(num)

    # hand_points = hand_3d[:, valid_idx]
    point_clouds = set_length(hand_3d, point_num)

    return hand_3d, point_clouds


def set_length(data, point_num):
    "set the point number"
    point_shape = data.shape[1]

    if point_shape < point_num:
        rand_idx = np.arange(0, point_num, 1, dtype=np.int32)
        rand_idx[point_shape:] = np.random.randint(0, point_shape,
                                                   size=point_num - point_shape)
    else:
        rand_idx = np.random.randint(0, point_shape, size=point_num)

    point_cloud = data[:, rand_idx]

    return point_cloud


if __name__ == "__main__":

    data = sio.loadmat('./result/P0.mat')
    header = data['1'][0][0][0]
    depth = data['1'][0][1][0]
    sample = {
        'header': header,
        'depth': depth
    }
    pc = point_cloud(sample)
    sio.savemat('./result/pc1.mat', {
        'pc': pc
    })
    visualize(pc)
