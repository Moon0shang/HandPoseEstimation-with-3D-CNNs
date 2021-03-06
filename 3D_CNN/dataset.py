"""
load the tsdf data and set the sequence
"""
import os
import os.path
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

# set the GPU devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


class MSRA_Dataset(data.Dataset):
    def __init__(self, root_path, opt, train=True, aug=False):  # opt,
        self.root_path = root_path
        self.train = train
        self.size = 'small'  # opt.size  # load 样本的数量，full /small 'small'  #
        self.test_idx = 2  # opt.test_index  # 1  #
        self.PCA_SZ = 63  # opt.PCA_SZ  # PCA 成分数量大小，默认为 63(int)63  #
        self.AUG = aug

        if self.size == 'full':
            self.SUBJECTS = 9
            self.GESTURES = 17
        elif self.size == 'small':
            self.SUBJECTS = 4
            self.GESTURES = 5

        self.total_num = self.__getlength()

        # 初始化 python 数组
        self.tsdf = np.empty([self.total_num, 3, 32, 32, 32], dtype=np.float32)
        self.ground_truth = np.empty([self.total_num, 63], dtype=np.float32)
        self.max_l = np.empty(self.total_num, dtype=np.float32)
        self.mid_p = np.empty([self.total_num, 3], dtype=np.float32)

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

        # load PCA data

        # transfer to torch
        # self.tsdf = torch.from_numpy(self.tsdf)
        # self.ground_truth = torch.from_numpy(self.ground_truth)
        # self.max_l = torch.from_numpy(self.max_l)
        # self.mid_p = torch.from_numpy(self.mid_p)

        if self.size == 'full':
            self.__loadPCA()

    def __getitem__(self, index):
        """return index data"""

        if self.size == 'full':
            return self.tsdf[index, :, :, :, :], self.ground_truth[index, :], self.max_l[index], self.mid_p[index, :], self.ground_truth_pca[index, :]
        else:
            return self.tsdf[index, :, :, :, :], self.ground_truth[index, :], self.max_l[index], self.mid_p[index, :]

    def __len__(self):
        """to get all data number"""
        return self.tsdf.shape[0]

    def __loaddata(self, data_dir, laug=False):
        "load data from preprocess files"

        if laug:
            s = '_aug'
        else:
            s = ''

        tsdf_file = sorted(os.listdir(os.path.join(data_dir, 'TSDF%s' % s)))
        g_file = sorted(os.listdir(
            os.path.join(data_dir, 'ground_truth%s' % s)))

        for ges in range(self.GESTURES):
            # tsdf, max_l, mid_p
            data = np.load(os.path.join(data_dir, 'TSDF', tsdf_file[ges]))
            tsdf = data['tsdf'].astype(np.float32)
            max_l = data['max_l'].astype(np.float32)
            mid_p = data['mid_p'].astype(np.float32)
            # ground truth
            ground_truth = np.load(os.path.join(
                data_dir, 'ground_truth', g_file[ges]))
            ground_truth = ground_truth.astype(np.float32)
            if len(ground_truth.shape) == 3:
                ground_truth[:, :, 2] = -ground_truth[:, :, 2]
                g_t = ground_truth.reshape(-1, 63)
            # index order
            self.start_index = self.end_index+1
            self.end_index = self.end_index + tsdf.shape[0]

            self.tsdf[(self.start_index-1):self.end_index, :, :, :, :] = tsdf
            self.ground_truth[(self.start_index-1):self.end_index, :] = g_t
            self.max_l[(self.start_index-1):self.end_index] = max_l
            self.mid_p[(self.start_index-1):self.end_index, :] = mid_p

    def __getlength(self):
        "get length of each subject"
        results = sorted(os.listdir(self.root_path))[:9]
        total_num = 0
        if self.train:
            for sub in range(self.SUBJECTS):
                if sub != self.test_idx:
                    sub_dir = os.path.join(self.root_path, results[sub], 'num')
                    ges_files = sorted(os.listdir(sub_dir))
                    for ges in range(self.GESTURES):
                        num_dir = os.path.join(sub_dir, ges_files[ges])
                        nums = np.load(num_dir)
                        total_num += nums
                    if self.AUG:
                        sub_dir = os.path.join(
                            self.root_path, results[sub], 'num_aug')
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

        # files = sorted(os.listdir(data_dir))[9:9+self.SUBJECTS]

        # if self.train:
        #     total_num = 0
        #     for idx, name in enumerate(files):
        #         if idx != self.test_idx:
        #             num = np.load(os.path.join(data_dir, name))
        #             total_num += num
        #     if self.AUG:
        #         t_num = np.load(os.path.join(data_dir, files[self.test_idx]))
        #         total_num = total_num*2+t_num*2
        # else:
        #     total_num = np.load(os.path.join(data_dir, files[self.test_idx]))

        return total_num

    def __loadPCA(self):

        all_f = os.listdir('./PCA')
        if self.AUG:
            files = all_f[:9]
        else:
            files = all_f[9:]
        data = np.load(os.path.join('./PCA', files[self.test_idx]))

        self.PCA_mean = data['pca_mean'].astype(np.float32)
        self.PCA_coeff = data['coeff'][:, 0:self.PCA_SZ].astype(np.float32)
        tmp = self.ground_truth - self.PCA_mean
        self.ground_truth_pca = np.dot(tmp, self.PCA_coeff)
        self.PCA_coeff = self.PCA_coeff.transpose(0, 1)
        self.PCA_coeff = torch.from_numpy(self.PCA_coeff)
        self.PCA_mean = torch.from_numpy(self.PCA_mean)


if __name__ == "__main__":

    D = MSRA_Dataset('./result/')
    td = torch.utils.data.DataLoader(D, batch_size=8, num_workers=0)
    for i, data in enumerate(td):
        tsdf_x, tsdf_y, tsdf_z, ground_truth, volume_arg = data
        tsdf = torch.stack((tsdf_x, tsdf_y, tsdf_z), dim=1)
        b_size = len(tsdf)
        joint = ground_truth[0]
        joint_pca = ground_truth[1]
        max_l = volume_arg[0]
        mid_p = volume_arg[1]
        # if i > 400:
        #     break
        # print('d')
    print('d')
