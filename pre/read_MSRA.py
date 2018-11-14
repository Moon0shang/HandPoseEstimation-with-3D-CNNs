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
"""
import os
import os.path
import numpy as np
import scipy.io as sio

from point_cloud import point_cloud
from tsdf import tsdf_cal
# from data_aug import data_augmentation


DB_dir = '/home/x/DB/MSRA HandPoseDataset/cvpr15_MSRAHandGestureDB/'
SAVE_dir = './result'


def main():
    "main part"
    try:
        os.mkdir(SAVE_dir)
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
        try:
            os.mkdir(sub_dir)
            print('create directory "%s".' % sub)
        except:
            print('directory "%s" already exist!')
        ground_truth = []
        for ges in gesture:
            # ges_dir = os.path.join(sub_dir, ges)
            # try:
            #     os.mkdir(ges_dir)
            #     print('create directory "%s".' % ges)
            # except:
            #     print('directory "%s" already exist!')
            file_dir = os.path.join(DB_dir, sub, ges)
            [bin_num, truth_single] = read_joint(file_dir)
            ground_truth.append(truth_single)

            POINT_CLOUD = np.empty([bin_num, 3, ])
            TSDF = np.empty([bin_num, 3, 32, 32, 32])
            AUG_tsdf = np.empty([bin_num, 3, 32, 32, 32])
            for i in range(bin_num):
                file_name = os.path.join(file_dir, '%06d_depth.bin' % i)
                [header, depth] = read_bin(file_name)
                # for the header, the order is image width, image height,
                # boundbox left,bb top, bb right, bb bottom
                data_single = [header, depth]
                pc = point_cloud(data_single)
                POINT_CLOUD[i] = pc
                tsdf = tsdf_cal(header, pc)
                TSDF[i] = tsdf
                # aug_pc = data_augmentation(pc)
                # aug_tsdf = tsdf_cal(header, aug_pc)
                # AUG_tsdf[i] = aug_tsdf

            sio.savemat(os.path.join(sub_dir, 'Point_Cloud', '%s.mat' % ges),
                        {'PC': POINT_CLOUD})
            print('file %s-point_cloud saved' % ges)
            sio.savemat(os.path.join(sub_dir, "TSDF", '%s.mat' % ges),
                        {'TSDF': TSDF})
            print('file % s-TSDF.mat saved' % ges)
            # sio.savemat(os.path.join(sub_dir, "TSDF", '%s.mat' % ges),
            #             {'TSDF': AUG_tsdf})
            # print('file % s-AUG_tsdf.mat saved' % ges)
        sio.savemat(os.path.join(SAVE_dir, '%s-ground_truth.mat' % sub),
                    {'ground_truth': ground_truth})
        print('gound_truth file saved.')


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
