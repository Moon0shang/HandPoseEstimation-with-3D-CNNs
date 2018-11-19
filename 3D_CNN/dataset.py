"""
load the tsdf data and set the sequence
"""
import os
import os.path
import torch
import numpy as np
import scipy.io as sio
import torch.utils.data as data


class TsdfDataset(data.dataset):
    def __init__(self, root_path, opt, train=True):
        self.root_path = root_path
        self.train = train
        self.size = opt.size  # load 样本的数量，full /small
        self.test_idx = opt.test_idx
        self.JOINT_NUM = opt.JOINT_NUM  # joint 数量，默认为 21

        if self.size == 'full':
            self.SUBJECTS = 9
            self.GESTURES = 17
        elif self.size == 'small':
            self.SUBJECTS = 3
            self.GESTURES = 2

        # self.total_frame_num = self.__total_frame_num()  # 从Volume Length中获取的长度

        # 初始化 python 数组
        self.tsdf = np.empty()
        self.ground_truth = np.empty()
        self.max_l = np.empty()
        self.mid_p = np.empty()

        self.start_index = 0
        self.end_index = 0
        # 获取训练数据路径并加载数据
        if self.train:
            for sub in range(self.SUBJECTS):
                if sub != self.test_idx:
                    for ges in range(self.GESTURES):
                        data_dir = os.path.join(self.root_path,)
                        print("Training: " + data_dir)
                        self.__loaddata(data_dir)
        else:
            for ges in range(self.GESTURES):
                data_dir = os.path.join(self.root_path,)
                print("Testing: " + data_dir)
                self.__loaddata(data_dir)

        self.tsdf = torch.from_numpy(self.tsdf)
        self.ground_truth = torch.from_numpy(self.ground_truth)

    def __getitem__(self, index):
        """still have problems"""
        return self.tsdf[index, :, :, :, :], self.ground_truth[index, :, :], self.max_l[index], slef.mid_p[index, :]

    def __len__(self):
        """still have problems"""
        return self.tsdf.size(0)

    def __loaddata(self, data_dir):
        data = sio.loadmat(os.path.join(data_dir, 'tsdf.mat'))
        tsdf = data['tsdf']
        ground_truth = sio.loadmat(os.path.join(data_dir, "ground truth.mat"))
        self.start_index = self.end_index + 1
        self.end_index = self.end_index + len(tsdf)

        self.tsdf[(self.start_index - 1):self.end_index, :, :, :, :] = tsdf.astype(
            np.float32)
        self.ground_truth[(self.start_index - 1):self.end_index,
                          :, :] = ground_truth['ground_truth'].astype(np.float32)
        self.max_l[(self.start_index - 1):self.end_index] = data['max_l'].astype(np.float32)
        self.mid_p[(self.start_index - 1):self.end_index,
                   :] = data['mid_p'].astype(np.float32)

    def __getlength(self, data_dir):
