"""
read all MSRA bin files and convert them to point cloud and tsdf
"""
import os
import os.path
import numpy as np

from process import DataProcess

# files directions
DB_dir = '/media/hp2/96e32a29-1cf9-4ce8-ba6c-706bd9d5f8fa/hp2/xuyuanquan/code/MSRA HandPoseDataset/cvpr15_MSRAHandGestureDB'
SAVE_dir = './result'
# set whether to do data augmentation
AUG = True


def main():
    "main part"
    try:
        os.mkdir(SAVE_dir)
        print('create directory "result".')
    except:
        print('directory "result" already exist!')
    # get the file name of each file
    # there are 9 subjects and 17 gestures in each subject
    subject = sorted(os.listdir(DB_dir))[:3]
    gesture = sorted(os.listdir(os.path.join(DB_dir, subject[0])))
    # in each gesture files there are 1 joint.txt file that
    # stores the number of bin files and the ground truth infomation
    for sub in subject:
        sub_dir = os.path.join(SAVE_dir, sub)
        try:
            os.mkdir(sub_dir)
            print('create directory "%s".' % sub)
        except:
            print('directory "%s" already exist!' % sub)

        try:
            os.mkdir(os.path.join(sub_dir, 'Point_Cloud'))
            os.mkdir(os.path.join(sub_dir, 'TSDF'))
            os.mkdir(os.path.join(sub_dir, 'ground_truth'))
            os.mkdir(os.path.join(sub_dir, 'num'))
            os.mkdir(os.path.join(sub_dir, 'joint_nor'))
            print('create data directories')
        except:
            print('failed create saved files')

        if AUG:
            try:
                os.mkdir(os.path.join(sub_dir, 'Point_Cloud_aug'))
                os.mkdir(os.path.join(sub_dir, 'TSDF_aug'))
                os.mkdir(os.path.join(sub_dir, 'ground_truth_aug'))
                os.mkdir(os.path.join(sub_dir, 'num_aug'))
                os.mkdir(os.path.join(sub_dir, 'joint_nor_aug'))
                print('create data augmentation directories')
            except:
                print('failed create aug files')

        total_num = 0
        for ges in gesture:
            file_dir = os.path.join(DB_dir, sub, ges)
            [bin_num, ground_truth] = read_joint(file_dir)
            ground_truth = ground_truth.reshape(bin_num, 21, 3)
            # ground_truth[:, :, 2] = -ground_truth[:, :, 2]
            # total_num += bin_num

            points_num = 6000
            if AUG:
                POINT_CLOUD_AUG = np.empty([bin_num, points_num, 3])
                TSDF_AUG = np.empty([bin_num, 3, 32, 32, 32])
                MAX_L_AUG = np.empty(bin_num)
                MID_P_AUG = np.empty([bin_num, 3])
                JOINT_NOR_AUG = np.empty([bin_num, 63])
                ground_truth_aug = np.empty(ground_truth.shape)

            POINT_CLOUD = np.empty([bin_num, points_num, 3])
            TSDF = np.empty([bin_num, 3, 32, 32, 32])
            MAX_L = np.empty(bin_num)
            MID_P = np.empty([bin_num, 3])
            JOINT_NOR = np.empty([bin_num, 63])

            for i in range(bin_num):
                file_name = os.path.join(file_dir, '%06d_depth.bin' % i)
                # for the header, the order is image width, image height,
                # boundbox left,bb top, bb right, bb bottom
                [header, depth] = read_bin(file_name)
                data_single = {'header': header, 'depth': depth}
                DP = DataProcess(data_single, points_num, aug=AUG)
                # data_pre = DP.process()

                POINT_CLOUD[i] = DP.get_points()
                TSDF[i], MAX_L[i], MID_P[i] = DP.tsdf_f(POINT_CLOUD[i])
                JOINT_NOR[i] = DP.normalize(
                    ground_truth[i], MAX_L[i], MID_P[i])

                if AUG:
                    POINT_CLOUD_AUG[i], ground_truth_aug[i] = DP.data_aug(
                        ground_truth[i], POINT_CLOUD[i])
                    TSDF_AUG[i], MAX_L_AUG[i], MID_P_AUG[i] = DP.tsdf_f(
                        POINT_CLOUD_AUG[i])
                    JOINT_NOR_AUG = DP.normalize(
                        ground_truth_aug[i], MAX_L_AUG[i], MID_P_AUG[i])

            np.save(os.path.join(sub_dir, 'Point_Cloud', '%s.npy' % ges),
                    POINT_CLOUD)
            np.savez(os.path.join(sub_dir, 'TSDF', '%s.npz' % ges),
                     tsdf=TSDF, max_l=MAX_L, mid_p=MID_P)
            np.save(os.path.join(sub_dir, 'ground_truth', '%s.npy' % ges),
                    ground_truth)
            np.save(os.path.join(sub_dir, 'num', '%s.npy' % ges),
                    bin_num)
            np.save(os.path.join(sub_dir, 'joint_nor', '%s.npy' % ges),
                    JOINT_NOR)

            if AUG:
                np.save(os.path.join(sub_dir, 'Point_Cloud_aug', '%s.npy' % ges),
                        POINT_CLOUD_AUG)
                np.savez(os.path.join(sub_dir, 'TSDF_aug', '%s.npz' % ges),
                         tsdf=TSDF_AUG, max_l=MAX_L_AUG, mid_p=MID_P_AUG)
                np.save(os.path.join(sub_dir, 'ground_truth_aug', '%s.npy' % ges),
                        ground_truth_aug)
                np.save(os.path.join(sub_dir, 'num_aug', '%s.npy' % ges),
                        bin_num)
                np.save(os.path.join(sub_dir, 'joint_nor_aug', '%s.npy' % ges),
                        JOINT_NOR_AUG)

            print('%s-%s files saved.' % (sub, ges))
        # np.save(os.path.join(SAVE_dir, 'data_num-%s.npy' % sub), total_num)
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
