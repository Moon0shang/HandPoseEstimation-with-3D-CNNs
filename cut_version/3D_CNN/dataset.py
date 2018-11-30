"""
load the tsdf data and set the sequence
"""
import os
import os.path
import torch
import numpy as np
import torch.utils.data as data
# from torch.utils.data import DataLoader

# set the GPU devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


class MSRA_Dataset(data.Dataset):
    def __init__(self, root_path, opt, train=True, aug=False):  # opt,
        self.root_path = root_path
        self.train = train
        self.size = opt.size  # load 样本的数量，full /small 'small'  #
        self.test_idx = opt.test_index  # 1  #
        self.PCA_SZ = opt.PCA_SZ  # PCA 成分数量大小，默认为 63(int)63  #
        self.AUG = aug

        if self.size == 'full':
            self.SUBJECTS = 3
            self.GESTURES = 17
        elif self.size == 'small':
            self.SUBJECTS = 3
            self.GESTURES = 3

        self.total_num = self.__getlength()

        # 初始化 python 数组
        self.tsdf = np.empty([self.total_num, 3, 32, 32, 32], dtype=np.float32)
        self.ground_truth = np.empty([self.total_num, 21, 3], dtype=np.float32)
        self.max_l = np.empty(self.total_num, dtype=np.float32)
        self.mid_p = np.empty([self.total_num, 3], dtype=np.float32)
        self.joint_nor = np.empty([self.total_num, 63], dtype=np.float32)

        self.start_index = 0
        self.end_index = 0
        # 获取训练数据路径并加载数据
        results = sorted(os.listdir(self.root_path))[:9]
        if self.train:
            for sub in range(self.SUBJECTS):
                if sub != self.test_idx:
                    data_dir = os.path.join(self.root_path, results[sub])
                    print("Training: " + data_dir)
                    self.__loaddata(data_dir)
        else:
            data_dir = os.path.join(self.root_path, results[self.test_idx])
            print("Testing: " + data_dir)
            self.__loaddata(data_dir)

        # add augmentation dataset
        if self.AUG:
            for sub in range(self.SUBJECTS):
                data_dir = os.path.join(self.root_path, results[sub])
                print('Training aug: ' + data_dir)
                self.__loaddata(data_dir, laug=True)

    def __getitem__(self, index):
        """return index data"""

        return self.tsdf[index, :, :, :, :], self.ground_truth[index, :, :], self.max_l[index], self.mid_p[index, :], self.joint_nor[index, :]

    def __len__(self):
        """to get all data number"""
        return self.tsdf.shape[0]

    def __loaddata(self, data_dir, laug=False):
        "load data from preprocess files"

        if laug:
            aug = '_aug'
        else:
            aug = ''

        tsdf_file = sorted(os.listdir(os.path.join(data_dir, 'TSDF%s' % aug)))
        g_file = sorted(os.listdir(
            os.path.join(data_dir, 'ground_truth%s' % aug)))
        n_file = sorted(os.listdir(
            os.path.join(data_dir, 'joint_nor%s' % aug)))

        for ges in range(self.GESTURES):
            # tsdf, max_l, mid_p
            data = np.load(os.path.join(data_dir, 'TSDF%s' %
                                        aug, tsdf_file[ges]))
            tsdf = data['tsdf'].astype(np.float32)
            max_l = data['max_l'].astype(np.float32)
            mid_p = data['mid_p'].astype(np.float32)
            # ground truth
            ground_truth = np.load(os.path.join(
                data_dir, 'ground_truth%s' % aug, g_file[ges]))
            ground_truth = ground_truth.astype(np.float32)
            # ground_truth[:, :, 2] = -ground_truth[:, :, 2]
            # g_t = ground_truth.reshape(-1, 63)
            # joint normalized
            joint_nor = np.load(os.path.join(
                data_dir, 'joint_nor%s' % aug, n_file[ges]))
            joint_nor = joint_nor.astype(np.float32)
            joint_nor = joint_nor.reshape(-1, 63)
            # index order
            self.start_index = self.end_index
            self.end_index = self.end_index + tsdf.shape[0]

            self.tsdf[self.start_index:self.end_index, :, :, :, :] = tsdf
            self.ground_truth[self.start_index:self.end_index,
                              :, :] = ground_truth
            self.max_l[self.start_index:self.end_index] = max_l
            self.mid_p[self.start_index:self.end_index, :] = mid_p
            self.joint_nor[self.start_index:self.end_index, :] = joint_nor

    def __getlength(self):
        "get length of each subject"
        results = sorted(os.listdir(self.root_path))[:9]
        total_num = 0
        if self.train:
            for sub in range(self.SUBJECTS):
                if self.AUG:
                    sub_dir = os.path.join(
                        self.root_path, results[sub], 'num')
                    ges_files = sorted(os.listdir(sub_dir))
                    for ges in range(self.GESTURES):
                        num_dir = os.path.join(sub_dir, ges_files[ges])
                        nums = np.load(num_dir)
                        total_num += nums
                else:
                    if sub != self.test_idx:
                        sub_dir = os.path.join(
                            self.root_path, results[sub], 'num')
                        ges_files = sorted(os.listdir(sub_dir))
                        for ges in range(self.GESTURES):
                            num_dir = os.path.join(sub_dir, ges_files[ges])
                            nums = np.load(num_dir)
                            total_num += nums

        else:
            sub_dir = os.path.join(
                self.root_path, results[self.test_idx], 'num')
            ges_files = sorted(os.listdir(sub_dir))
            for ges in range(self.GESTURES):
                num_dir = os.path.join(sub_dir, ges_files[ges])
                nums = np.load(num_dir)
                total_num += nums

        return total_num


if __name__ == "__main__":

    # opt = None
    # D = MSRA_Dataset('./result/', opt)
    # td = torch.utils.data.DataLoader(D, batch_size=8, num_workers=0)
    # for i, data in enumerate(td):
    #     tsdf_x, tsdf_y, tsdf_z, ground_truth, volume_arg = data
    #     tsdf = torch.stack((tsdf_x, tsdf_y, tsdf_z), dim=1)
    #     b_size = len(tsdf)
    #     joint = ground_truth[0]
    #     joint_pca = ground_truth[1]
    #     max_l = volume_arg[0]
    #     mid_p = volume_arg[1]
    #     # if i > 400:
    #     #     break
    #     # print('d')
    # print('d')
    pass
