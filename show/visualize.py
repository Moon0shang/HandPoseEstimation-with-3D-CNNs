import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_point(hand_points, lable):

    x = hand_points[:, 0]
    y = hand_points[:, 1]
    z = hand_points[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        x, y, z,
        marker='.'  # show as a point
    )
    ax.axis('equal')
    ax.set_title('%s' % lable)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.pause(0.5)


def visualize_joints(joint, out, factors):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joint[:, 0], joint[:, 1], -joint[:, 2], marker='^', color='k')
    ax.scatter(out[:, 0], out[:, 1], -out[:, 2], marker='o', color='r')
    base = np.arange(5)
    c = ['b', 'y', 'c', 'm', 'g']  # b,g,r,c,m,y,k,w
    ax.plot(joint[base, 0], joint[base, 1], -joint[base, 2],
            color=c[0], linestyle='--', linewidth=0.8, label='origin')
    for i in range(1, 5):
        f_idx = base + 4 * i
        f_idx[0] = 0
        ax.plot(joint[f_idx, 0], joint[f_idx, 1], -joint[f_idx, 2],
                color=c[i], linestyle='--', linewidth=0.8)

    ax.plot(out[base, 0], out[base, 1], -out[base, 2],
            color=c[0], label='augment')
    for i in range(1, 5):
        f_idx = base + 4 * i
        f_idx[0] = 0
        ax.plot(out[f_idx, 0], out[f_idx, 1], -out[f_idx, 2], color=c[i])

    ax.axis('scaled')
    f1, f2, f3 = factors
    ax.set_title('ground truth compare\nstrech=%.2f,XYrot=%d,Zrot=%d' %
                 (f1, f2, f3))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()


def data_aug(point_clouds, joints):
    """
    data augmentation: contain stretch and rotation
    stretch factor: x, y: [2/3,3/2]; z: 1
    rotation factor: theta x,y: [-30d,30d]; theta z: [-180d,180d]
    """
    # stretch factor
    # stretch_xy = np.random.uniform(2 / 3, 3 / 2)
    stretch_xy = 1.2
    stretch = np.array([stretch_xy, stretch_xy, 1])
    # diag matrix
    S = np.diag(stretch)

    # rotate factor
    rot_xy = 10  # np.random.randint(-30, 30)
    rot_z = 30  # np.random.randint(-180, 180)
    # x, y matrix
    xy_c = np.cos(np.pi*rot_xy/180)
    xy_s = np.sin(np.pi*rot_xy/180)
    R_x = np.array([[1, 0, 0], [0, xy_c, xy_s], [0, -xy_s, xy_c]])
    R_y = np.array([[xy_c, 0, -xy_s], [0, 1, 0], [xy_s, 0, xy_c]])
    z_c = np.cos(rot_z)
    z_s = np.sin(rot_z)
    R_z = np.array([[z_c, z_s, 0], [-z_s, z_c, 0], [0, 0, 1]])
    # rotation matrix
    R = np.dot(R_x, R_y)
    R = np.dot(R, R_z)

    # ground truth augmentation
    # stretch
    joint_stretch = np.dot(joints, S)

    # normalize the points to center point
    # center_mean = np.mean(joint_stretch, axis=1)
    # center_point = np.empty(
    #     [center_mean.shape[0],  3])
    # center_point[:,  0] = center_mean
    # center_point[:,  1] = center_mean
    # center_point[:,  2] = center_mean
    # coor_nor = joint_stretch - center_point
    # rotate
    joint_rot = np.dot(joint_stretch, R)
    # unnormalize the points to coordinate system
    # joint_coor = joint_rot + center_point
    ground_truth_aug = joint_rot  # .reshape(-1, 63)

    # point cloud augmentation
    pc_stretch = np.dot(point_clouds, S)
    # pc_mean = np.mean(pc_stretch, axis=1)
    # pc_center = np.empty([pc_mean.shape[0],  3])
    # pc_center[:,  0] = pc_mean
    # pc_center[:,  1] = pc_mean
    # pc_center[:, 2] = pc_mean
    # pc_nor = pc_stretch-pc_center
    pc_rot = np.dot(pc_stretch, R)
    point_clouds_aug = pc_rot

    # point_clouds_aug = pc_rot+pc_center
    factors = [stretch_xy, rot_xy, rot_z]

    return point_clouds_aug, ground_truth_aug, factors


if __name__ == "__main__":

    joints = np.load('./datas/ground_truth-1.npy')
    pcs = np.load('./datas/point_cloud-1.npy')
    joint = joints[0].reshape(-1, 3)
    pc = pcs[0]
    pc_aug, joint_aug, factors = data_aug(pc, joint)
    plt.ion()
    visualize_point(pc, 'point cloud')
    visualize_point(pc_aug, 'point cloud augmentation')
    visualize_joints(joint, joint_aug, factors)
    plt.ioff()
    plt.show()
