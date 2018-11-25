"""
rebuild the data precess.
read all MSRA bin files and convert them to point cloud 
also, we should get the data transfer them into the tsdf volume representation

I assume here need to have the following parts:
1. read the data and store them 
2. convert the store data into 3D cloud point
3. convert the store data into tsdf data
4. data augmentation(rotate and stretch)
5. visualize the point cloud and tsdf

this preprocess have 2 related files, process and the pca
the other files exist before the merge, now the previous two files contains all of them.
"""
import os
import os.path
import numpy as np
import scipy.io as sio

from process import DataProcess
from joint_pca import *

# files directions
DB_dir = '/home/x/DB/MSRA HandPoseDataset/cvpr15_MSRAHandGestureDB/'
SAVE_dir = './result'
# set whether to do data augmentation
AUG = False


def main():
    "main part"
    try:
        os.mkdir(SAVE_dir)
        # os.mkdir(PC_dir)
        print('create directory "result".')
    except:
        print('directory "result" already exist!')
    # get the file name of each file
    # there are 9 subjects and 17 gestures in each subject
    subject = sorted(os.listdir(DB_dir))
    gesture = sorted(os.listdir(os.path.join(DB_dir, subject[0])))
    # in each gesture files there are 1 joint.txt file that
    # stores the number of bin files and the ground truth infomation
    for sub in subject:
        sub_dir = os.path.join(SAVE_dir, sub)
        # pc_dir = os.path.join(PC_dir, sub)
        try:
            os.mkdir(sub_dir)
            print('create directory "%s".' % sub)
        except:
            print('directory "%s" already exist!' % sub)

        try:
            os.mkdir(os.path.join(sub_dir, 'Point_Cloud'))
            os.mkdir(os.path.join(sub_dir, 'TSDF'))
            os.mkdir(os.path.join(sub_dir, 'ground_truth'))
        except:
            print('failed create saved files')

        if AUG:
            try:
                os.mkdir(os.path.join(sub_dir, 'Point_Cloud_aug'))
                os.mkdir(os.path.join(sub_dir, 'TSDF_aug'))
                os.mkdir(os.path.join(sub_dir, 'ground_truth_aug'))
            except:
                print('failed create aug files')

        total_num = 0
        for ges in gesture:
            file_dir = os.path.join(DB_dir, sub, ges)
            [bin_num, ground_truth] = read_joint(file_dir)
            # ground_truth = ground_truth.reshape(bin_num, 21, 3)
            # ground_truth[:, :, 2] = -ground_truth[:, :, 2]
            total_num += bin_num

            point_num = 6000
            if AUG:
                POINT_CLOUD_AUG = np.empty([bin_num, point_num, 3])
                TSDF_AUG = np.empty([bin_num, 3, 32, 32, 32])
                MAX_L_AUG = np.empty(bin_num)
                MID_P_AUG = np.empty([bin_num, 3])
                ground_truth_aug = np.empty(ground_truth.shape)

            POINT_CLOUD = np.empty([bin_num, point_num, 3])
            TSDF = np.empty([bin_num, 3, 32, 32, 32])
            MAX_L = np.empty(bin_num)
            MID_P = np.empty([bin_num, 3])

            for i in range(bin_num):
                file_name = os.path.join(file_dir, '%06d_depth.bin' % i)
                # for the header, the order is image width, image height,
                # boundbox left,bb top, bb right, bb bottom
                [header, depth] = read_bin(file_name)
                data_single = {'header': header, 'depth': depth}
                DP = DataProcess(data_single, ground_truth[i, :],
                                 point_num=point_num, aug=AUG)
                data_pre = DP.process()

                POINT_CLOUD[i] = data_pre[0]
                TSDF[i] = data_pre[1]
                MAX_L[i] = data_pre[2]
                MID_P[i] = data_pre[3]
                if AUG:
                    POINT_CLOUD_AUG[i] = data_pre[4]
                    TSDF_AUG[i] = data_pre[5]
                    MAX_L_AUG[i] = data_pre[6]
                    MID_P_AUG[i] = data_pre[7]
                    ground_truth_aug[i] = data_pre[8]

            np.save(os.path.join(sub_dir, 'Point_Cloud', '%s.npy' % ges),
                    POINT_CLOUD)
            np.savez(os.path.join(sub_dir, 'TSDF', '%s.npz' % ges),
                     tsdf=TSDF, max_l=MAX_L, mid_p=MID_P)
            np.save(os.path.join(sub_dir, 'ground_truth', '%s.npy' % ges),
                    ground_truth)

            if AUG:
                np.save(os.path.join(sub_dir, 'Point_Cloud_aug', '%s.npy' % ges),
                        POINT_CLOUD_AUG)
                np.savez(os.path.join(sub_dir, 'TSDF_aug', '%s.npz' % ges),
                         tsdf=TSDF_AUG, max_l=MAX_L_AUG, mid_p=MID_P_AUG)
                np.save(os.path.join(sub_dir, 'ground_truth_aug', '%s.npy' % ges),
                        ground_truth_aug)

            print('%s-%s files saved.' % (sub, ges))
        np.save(os.path.join(SAVE_dir, 'data_num-%s.npy' % sub), total_num)
        print('%s total number saved.' % sub)


def read_joint(f_dir):
    "read the joint files and return the ground truth"
    # in the joint.txt, the first line stores the number of bin files
    # and the rest stores the ground truth
    f_name = os.path.join(f_dir, 'joint.txt')
    with open(f_name, 'r') as f:
        bin_num = int(f.readline())
    ground_truth = np.loadtxt(f_name, dtype=np.float32, skiprows=1)

    return bin_num, ground_truth


def read_bin(f_name):
    "read all bin files"
    with open(f_name, 'r') as f:
        # in the bin fils, the first 6 informations are image width,
        # image height, box left, box top, box right and box bottom
        # and the rest are the depth information of the image
        header = np.fromfile(f, dtype=np.int32, count=6)
        depth = np.fromfile(f, dtype=np.float32)

    return header, depth


if __name__ == '__main__':

    main()
    joint_pca()

    "after use Matlab process the ground truth data"
    Use_Matlab = False
    if Use_Matlab:
        merge_pca()
