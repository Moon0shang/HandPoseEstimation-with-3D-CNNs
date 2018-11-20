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
from tsdf import tsdf_f
# from data_aug import data_augmentation


DB_dir = '/home/x/DB/MSRA HandPoseDataset/cvpr15_MSRAHandGestureDB/'
SAVE_dir = './result'
PC_dir = './PC'


def main():
    "main part"
    try:
        os.mkdir(SAVE_dir)
        os.mkdir(PC_dir)
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
        pc_dir = os.path.join(PC_dir, sub)
        try:
            os.mkdir(sub_dir)
            os.mkdir(pc_dir)
            print('create directory "%s".' % sub)
        except:
            print('directory "%s" already exist!' % sub)
        # ground_truth = []
        total_num = 0
        for ges in gesture:
            # ges_dir = os.path.join(sub_dir, ges)
            # try:
            #     os.mkdir(ges_dir)
            #     print('create directory "%s".' % ges)
            # except:
            #     print('directory "%s" already exist!')
            file_dir = os.path.join(DB_dir, sub, ges)
            [bin_num, ground_truth] = read_joint(file_dir)
            total_num += bin_num

            point_num = 9000
            POINT_CLOUD = np.empty([bin_num, 3, point_num])
            TSDF = np.empty([bin_num, 3, 32, 32, 32])
            MAX_L = np.empty(bin_num)
            MID_P = np.empty([bin_num, 3])
            # AUG_tsdf = np.empty([bin_num, 3, 32, 32, 32])
            for i in range(bin_num):
                file_name = os.path.join(file_dir, '%06d_depth.bin' % i)
                # for the header, the order is image width, image height,
                # boundbox left,bb top, bb right, bb bottom
                [header, depth] = read_bin(file_name)
                data_single = {'header': header, 'depth': depth}
                [hand_3d, pc] = point_cloud(data_single, point_num)
                POINT_CLOUD[i] = pc
                tsdf, max_l, mid_point = tsdf_f(header, depth, hand_3d)
                TSDF[i] = tsdf
                MAX_L[i] = max_l
                MID_P[i] = mid_point
                # aug_pc = data_augmentation(pc)
                # aug_tsdf = tsdf_cal(header, aug_pc)
                # AUG_tsdf[i] = aug_tsdf
            np.save(os.path.join(pc_dir, 'Point_Cloud-%s.mat' % ges), POINT_CLOUD)
            # print('file %s-point_cloud saved' % ges)
            np.savez(os.path.join(sub_dir, 'TSDF-%s.mat' % ges),
                     tsdf=TSDF, max_l=MAX_L, mid_p=MID_P)
            # print('file % s-TSDF.mat saved' % ges)
            np.save(os.path.join(sub_dir, 'ground_truth-%s.mat' % ges),
                    ground_truth)
            # print('gound_truth file saved.')
            print('%s-%s files saved.' % (sub, ges))
        np.save(os.path.join(SAVE_dir, 'data_num-%s.mat' % sub), total_num)
        print('%s total number saved.' % sub)
        #     sio.savemat(os.path.join(pc_dir, 'Point_Cloud-%s.mat' % ges),
        #                 {'pc': POINT_CLOUD})
        #     print('file %s-point_cloud saved' % ges)
        #     sio.savemat(os.path.join(sub_dir, 'TSDF-%s.mat' % ges),
        #                 {
        #                     'tsdf': TSDF,
        #                     'max_l': MAX_L,
        #                     'mid_p': MID_P
        #     })
        #     print('file % s-TSDF.mat saved' % ges)
        #     # sio.savemat(os.path.join(sub_dir, "TSDF", '%s.mat' % ges),
        #     #             {'TSDF': AUG_tsdf})
        #     # print('file % s-AUG_tsdf.mat saved' % ges)
        #     sio.savemat(os.path.join(sub_dir, 'ground_truth-%s.mat' % ges),
        #                 {'ground_truth': ground_truth})
        #     print('gound_truth file saved.')
        # sio.savemat(os.path.join(SAVE_dir, 'data_num-%s.mat' % sub),
        #             {'num': total_num})
        # print('total number saved.')


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
