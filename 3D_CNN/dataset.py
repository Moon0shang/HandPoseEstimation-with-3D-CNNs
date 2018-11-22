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
        self.PCA_SZ = opt.PCA_SZ  # PCA 成分数量大小，默认为 42(int)

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
        # add augmentation dataset
        if self.AUG:
            for sub in range(self.SUBJECTS):
                data_dir = os.path.join(self.root_path, results[sub])
                print('Training aug: ' + data_dir)
                self.__loaddata(data_dir, aug=True)

        self.tsdf = torch.from_numpy(self.tsdf)
        self.ground_truth = torch.from_numpy(self.ground_truth)
        self.__loadPCA()

    def __getitem__(self, index):
        """return index data"""
        return self.tsdf[index, :, :, :, :], self.ground_truth[index, :], self.max_l[index], self.mid_p[index, :]

    def __len__(self):
        """to get all data number"""
        return self.tsdf.size(0)

    def __loaddata(self, data_dir, aug=False):
        "load data from preprocess files"

        if aug:
            s = '_aug'
        else:
            s = ''

        tsdf_file = sorted(os.listdir(os.path.join(data_dir, 'TSDF%s' % s)))
        g_file = sorted(os.listdir(
            os.path.join(data_dir, 'ground_truth%s' % s)))

        for ges in self.GESTURES:
            # data = sio.loadmat(os.path.join(data_dir, files[ges+17]))
            data = np.load(os.path.join(data_dir, tsdf_file[ges+17]))
            tsdf = data['tsdf']
            # ground_truth = sio.loadmat(os.path.join(data_dir, files[ges]))
            ground_truth = np.load(os.path.join(data_dir, g_file[ges]))
            self.start_index = self.end_index
            self.end_index = self.end_index + tsdf.shape[0]

            self.tsdf[self.start_index:self.end_index, :, :, :, :] = tsdf
            self.ground_truth[self.start_index:self.end_index,
                              :] = ground_truth
            self.max_l[self.start_index:self.end_index] = data['max_l']
            self.mid_p[self.start_index:self.end_index, :] = data['mid_p']

    def __getlength(self, data_dir):
        "get length of each subject"
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

    def __loadPCA(self):

        files = os.listdir('./PCA')
        data = np.load(os.path.join('./PCA', files[self.test_idx]))
        self.PCA_mean = torch.from_numpy(data['pca_mean'])
        self.PCA_coeff = torch.from_numpy(data['coeff'][:, 0:self.PCA_SZ])
        # 将一列的值扩展至 joint_num 列
        tmp = self.PCA_mean.expand(self.total_frame_num, self.JOINT_NUM * 3)
        tmp_demean = self.ground_truth - tmp
        # 矩阵相乘
        self.ground_truth_pca = torch.mm(tmp_demean, self.PCA_coeff)
        # 转置 transpose(dim1, dim2)
        self.PCA_coeff = self.PCA_coeff.transpose(0, 1).cuda()
        self.PCA_mean = self.PCA_mean.cuda()
