"""
load the tsdf data and set the sequence
"""
import os
import os.path
import torch
import numpy as np
import scipy.io as sio
import torch.utils.data as data


class MSRA_Dataset(data.dataset):
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

        self.total_num = self.__getlength(self.root_path)

        # 初始化 python 数组
        self.tsdf = np.empty([self.total_num, 3, 32, 32, 32], dtype=np.float32)
        self.ground_truth = np.empty([self.total_num, 63])
        self.max_l = np.empty(self.total_num)
        self.mid_p = np.empty([self.total_num, 3])

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

        self.tsdf = torch.from_numpy(self.tsdf)
        self.ground_truth = torch.from_numpy(self.ground_truth)

    def __getitem__(self, index):
        """return index data"""
        return self.tsdf[index, :, :, :, :], self.ground_truth[index, :], self.max_l[index], self.mid_p[index, :]

    def __len__(self):
        """still have problems"""
        return self.tsdf.size(0)

    def __loaddata(self, data_dir):
        "load data from preprocess files"
        files = sorted(os.listdir(data_dir))

        for ges in self.GESTURES:
            # data = sio.loadmat(os.path.join(data_dir, files[ges+17]))
            data = np.load(os.path.join(data_dir, files[ges+17]))
            tsdf = data['tsdf']
            # ground_truth = sio.loadmat(os.path.join(data_dir, files[ges]))
            ground_truth = np.load(os.path.join(data_dir, files[ges]))
            self.start_index = self.end_index
            self.end_index = self.end_index + tsdf.shape[0]

            self.tsdf[self.start_index:self.end_index, :, :, :, :] = tsdf
            self.ground_truth[self.start_index:self.end_index,
                              :] = ground_truth
            self.max_l[self.start_index:self.end_index] = data['max_l']
            self.mid_p[self.start_index:self.end_index, :] = data['mid_p']

    def __getlength(self, data_dir):
        "need modify"
        files = sorted(os.listdir(data_dir))[9:]

        if self.train:
            total_num = 0
            for idx, name in enumerate(files):
                if idx != self.test_idx:
                    # num = sio.loadmat(os.path.join(data_dir, name))
                    num = np.load(os.path.join(data_dir, name))
                    total_num += num
        else:
            total_num = np.load(os.path.join(data_dir, files[self.test_idx]))

        return total_num
