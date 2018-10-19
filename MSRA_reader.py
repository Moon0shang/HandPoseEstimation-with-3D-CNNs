import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sensor import cal_angle_from_pos
import os
import glob
from params import *
from heatmap import *
from tsdf import *
from time import time
import warnings
import tables
from PIL import Image, ImageOps


# joint  sequence: 21*3
# wrist, index_mcp, index_pip, index_dip, index_tip, middle_mcp, middle_pip, middle_dip, middle_tip, ring_mcp,
# ring_pip, ring_dip, ring_tip, little_mcp, little_pip, little_dip, little_tip, thumb_mcp, thumb_pip,
# thumb_dip, thumb_tip
class MSRADataSet(Dataset):
    def __init__(self, root_path, use_sensor=False, use_preprocessing=False):
        self.root_path = root_path
        self.subjects = filter(lambda x: os.path.isdir(
            os.path.join(root_path, x)), os.listdir(root_path))
        self.subjects.sort()
        self.subjects = self.subjects[:2]
        self.gestures = GESTURES
        self.samples = []
        self.subjects_length = []  # samples per subject
        self.imgs = []
        self.use_sensor = use_sensor
        self.use_preprocessing = use_preprocessing
        self.use_raw = False
        if use_preprocessing:
            self.f = tables.open_file('tsdf.h5', mode='r')
        for i in range(MAX_SAMPLE_LEN):
            self.samples.append('{:06d}'.format(i)+DATA_EXT)
            self.imgs.append('{:06d}'.format(i)+IMG_EXT)
        self.label = LABEL
        self.data = []
        for sub in self.subjects:
            sub_len = 0
            for ges in self.gestures:
                with open(os.path.join(root_path, sub, ges, LABEL)) as f:
                    lines = int(f.readline())
                    label = np.loadtxt(f, delimiter=' ', dtype=np.float32)
                for i in range(lines):
                    sub_len += 1
                    data = {'path': os.path.join(
                        root_path, sub, ges, self.samples[i]), 'label': label[i, :]}
                    self.data.append(data)
            self.subjects_length.append(sub_len)
        self.length = len(self.data)
        assert np.array(self.subjects_length).sum() == self.length

    # return 32*32*32 tsdf, 63 label, 1 max_l
    def __getitem__(self, item):
        with open(self.data[item]['path']) as f:
            # img_width img_height left top right bottom
            header = np.fromfile(f, np.int32, 6)
            data = np.fromfile(f, np.float32)

        sample = {'header': header,
                  'data': data}

        # tsdf output
        if self.use_preprocessing:
            row = self.f.root.data[item, :]
            return row[:VOXEL_RES**3].reshape((VOXEL_RES, VOXEL_RES, VOXEL_RES)), row[VOXEL_RES**3:-1], row[-1]
        elif self.use_raw:
            label = self.data[item]['label']
            data, heatmaps, n_label = self.pre_processing(sample, label)
            data = [np.array(i) for i in data]
            return data, heatmaps, n_label
        else:
            r = cal_tsdf_cuda(sample)
            if r is None:
                return None
            else:
                tsdf, max_l, mid_p = r
                label = self.data[item]['label']
                if self.use_sensor:
                    angles = cal_angle_from_pos(label.copy())
                    return (tsdf, angles), label, max_l, mid_p
                else:
                    return tsdf, label, max_l, mid_p

    def get_point_cloud(self, index):
        sample = self.get_raw_data(self.data[index]['path'])
        # point cloud output
        pc = cal_pointcloud(sample)
        return pc, self.data[index]['label']

    def pre_processing(self, s, gt, size=IMG_SIZE, map_size=HM_SIZE):
        # normalize image
        l = s['header'][2]
        t = s['header'][3]
        r = s['header'][4]
        b = s['header'][5]
        w = r - l
        h = b - t
        data = s['data']
        # normal to [0,1]
        f_data = data[data > 0]
        max_d = np.max(f_data)
        min_d = np.min(f_data)
        data = (data-min_d)/(max_d-min_d)
        data[data < 0] = 1
        img = Image.fromarray(data.reshape(h, w))
        ratio = float(size)/max(w, h)
        n_w, n_h = int(round(w*ratio)), int(round(h*ratio))
        img = img.resize((n_w, n_h), Image.ANTIALIAS)
        delta_w = size-n_w
        delta_h = size-n_h
        padding = (delta_w // 2, delta_h // 2, delta_w -
                   (delta_w // 2), delta_h - (delta_h // 2))
        new_im = ImageOps.expand(img, padding, 1)

        # generate heatmaps
        gt = gt.reshape(-1, 3)
        gt_pixel = camera2pixel(gt, *get_param('msra'))
        # maps = []
        map_data = np.empty(0, dtype=np.float32)
        images = [new_im, new_im.resize((IMG_SIZE/2, IMG_SIZE/2), Image.ANTIALIAS),
                  new_im.resize((IMG_SIZE/4, IMG_SIZE/4), Image.ANTIALIAS)]
        n_label = np.empty(0, dtype=np.float32)
        ratio_hm = float(map_size)/max(w, h)
        n_w, n_h = int(round(w*ratio_hm)), int(round(h*ratio_hm))
        delta_w = map_size-n_w
        delta_h = map_size-n_h
        for i in range(JOINT_LEN):
            x, y = gt_pixel[i, :2]
            h_data = np.zeros((map_size, map_size), dtype=np.float32)
            x, y = x-l, y-t
            x = int(x * ratio_hm + delta_w // 2)
            y = int(y * ratio_hm + delta_h // 2)
            n_label = np.append(n_label, np.array([y, x], dtype=np.float32))
            if x >= 0 and y >= 0 and x < h_data.shape[1] and y < h_data.shape[0]:
                # if x >= 1 and y >= 1 and x < h_data.shape[1] - 1 and y < h_data.shape[0] - 1:
                h_data[y, x] = 1

            map_data = np.append(map_data, h_data)
            # hm = Image.fromarray(h_data)
            # maps.append(hm)

        return images, map_data, n_label

    def __len__(self):
        # return 100
        return self.length

    def get_train_test_indices(self, test_id=0):
        if test_id >= len(self.subjects_length) or test_id < 0:
            raise ValueError("value must within range (0,{})".format(
                len(self.subjects_length)-1))
        if len(self.subjects_length) < 2:
            raise ValueError("dataset cannot be split")
        n_lens = np.array(self.subjects_length, dtype=np.int32)
        train = []
        test = []
        l = 0
        for i in range(len(self.subjects_length)):
            if i == test_id:
                test.extend(range(l, l+self.subjects_length[i]))
            else:
                train.extend(range(l, l+self.subjects_length[i]))
            l += self.subjects_length[i]
        return train, test


class MSRADataSet3D(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.subjects = filter(lambda x: os.path.isdir(
            os.path.join(root_path, x)), os.listdir(root_path))
        self.subjects.sort()
        # self.subjects = self.subjects[:2]
        self.gestures = GESTURES
        self.samples = []
        self.subjects_length = []  # samples per subject
        self.imgs = []
        for i in range(MAX_SAMPLE_LEN):
            self.samples.append('{:06d}'.format(i)+DATA_EXT)
            self.imgs.append('{:06d}'.format(i)+IMG_EXT)
        self.label = LABEL
        self.data = []
        for sub in self.subjects:
            sub_len = 0
            for ges in self.gestures:
                with open(os.path.join(root_path, sub, ges, LABEL)) as f:
                    lines = int(f.readline())
                    label = np.loadtxt(f, delimiter=' ', dtype=np.float32)
                for i in range(lines):
                    sub_len += 1
                    data = {'path': os.path.join(
                        root_path, sub, ges, self.samples[i]), 'label': label[i, :]}
                    self.data.append(data)
            self.subjects_length.append(sub_len)
        self.length = len(self.data)
        assert np.array(self.subjects_length).sum() == self.length

    # return 44*44*44 voxel, 63 label
    def __getitem__(self, item):
        with open(self.data[item]['path']) as f:
            # img_width img_height left top right bottom
            header = np.fromfile(f, np.int32, 6)
            data = np.fromfile(f, np.float32)

        label = self.data[item]['label']
        # data, heatmaps, n_label = self.pre_processing(header, data, label)
        v = self.pre_processing(header, data, label)
        # data = [np.array(i) for i in data]
        # return data,heatmaps,n_label
        return v

    def get_center_voxel(self, header, data, size=48):
        pc = cal_pointcloud(header, data)
        max_coor = np.max(pc, axis=0)
        min_coor = np.min(pc, axis=0)
        max_size = np.max(max_coor-min_coor)
        mid = (max_coor+min_coor)/2
        v = np.zeros(shape=(size, size, size), dtype=np.float32)
        new_min_coor = mid-np.array([max_size/2, max_size/2, max_size/2])
        new_min_coor = np.expand_dims(new_min_coor, axis=0)
        voxel_size = max_size/size
        n_pc = (pc-np.tile(new_min_coor, (pc.shape[0], 1)))-0.001
        n_pc = (n_pc/voxel_size).astype(int)
        v[n_pc[:, 0], n_pc[:, 1], n_pc[:, 2]] = 1
        return v

    def pre_processing(self, header, data, gt, size=IMG_SIZE, map_size=HM_SIZE):
        # normalize image
        l = header[2]
        t = header[3]
        r = header[4]
        b = header[5]
        w = r - l
        h = b - t
        # convert to point cloud
        v = self.get_center_voxel(header, data)
        # normal to [0,1]
        # f_data = data[data > 0]
        # max_d = np.max(f_data)
        # min_d = np.min(f_data)
        # data = (data-min_d)/(max_d-min_d)
        # data[data<0] = 1
        # img = Image.fromarray(data.reshape(h,w))
        # ratio = float(size)/max(w,h)
        # n_w,n_h = int(round(w*ratio)),int(round(h*ratio))
        # img = img.resize((n_w,n_h), Image.ANTIALIAS)
        # delta_w = size-n_w
        # delta_h = size-n_h
        # padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        # new_im = ImageOps.expand(img, padding, 1)
        #
        # # generate heatmaps
        # gt = gt.reshape(-1, 3)
        # gt_pixel = camera2pixel(gt, *get_param('msra'))
        # # maps = []
        # map_data = np.empty(0,dtype=np.float32)
        # images = [new_im, new_im.resize((48,48),Image.ANTIALIAS),new_im.resize((24,24),Image.ANTIALIAS)]
        # n_label = np.empty(0,dtype=np.float32)
        # ratio_hm = float(map_size)/max(w,h)
        # n_w,n_h = int(round(w*ratio_hm)),int(round(h*ratio_hm))
        # delta_w = map_size-n_w
        # delta_h = map_size-n_h
        # for i in range(JOINT_LEN):
        #     x,y = gt_pixel[i, :2]
        #     h_data = np.zeros((map_size, map_size), dtype=np.float32)
        #     x,y = x-l,y-t
        #     x = int(x * ratio_hm + delta_w // 2)
        #     y = int(y * ratio_hm + delta_h // 2)
        #     n_label = np.append(n_label,np.array([y,x],dtype=np.float32))
        #     if x>=0 and y>=0 and x<h_data.shape[1] and y<h_data.shape[0]:
        #     # if x >= 1 and y >= 1 and x < h_data.shape[1] - 1 and y < h_data.shape[0] - 1:
        #         h_data[y, x] = 1
        #
        #     map_data = np.append(map_data,h_data)
        #     # hm = Image.fromarray(h_data)
        #     # maps.append(hm)

        return v
        # return images,map_data,n_label

    def __len__(self):
        # return 100
        return self.length

    def get_train_test_indices(self, test_id=0):
        if test_id >= len(self.subjects_length) or test_id < 0:
            raise ValueError("value must within range (0,{})".format(
                len(self.subjects_length)-1))
        if len(self.subjects_length) < 2:
            raise ValueError("dataset cannot be split")
        n_lens = np.array(self.subjects_length, dtype=np.int32)
        train = []
        test = []
        l = 0
        for i in range(len(self.subjects_length)):
            if i == test_id:
                test.extend(range(l, l+self.subjects_length[i]))
            else:
                train.extend(range(l, l+self.subjects_length[i]))
            l += self.subjects_length[i]
        return train, test


def get_raw_data(path):
    with open(path) as f:
        # img_width img_height left top right bottom
        header = np.fromfile(f, np.int32, 6)
        data = np.fromfile(f, np.float32)
    return {'header': header,
            'data': data}


def cal_pointcloud(header, data):
    l = header[2]
    t = header[3]
    r = header[4]
    b = header[5]
    b_w = r - l
    p_clouds = []
    for y in range(t, b):
        for x in range(l, r):
            idx = (y - t) * b_w + x - l
            cam_z = data[idx]
            if cam_z == 0:
                continue
            q = cam_z / FOCAL
            cam_x = q * (x - CENTER_X)
            cam_y = -q * (y - CENTER_Y)  # change to right hand axis
            cam_z = -cam_z
            p_clouds.append([cam_x, cam_y, cam_z])
    npa = np.asarray(p_clouds, dtype=np.float32)
    return npa


if __name__ == "__main__":
    # n = np.array([-1.1,2,3])
    # print n.astype(int)
    # n = np.expand_dims(n,axis=0)
    # print n.shape
    # print np.tile(n,(2,1))
    from visualization import *
    m = MSRADataSet3D(DATA_DIR)
    v = m[0]
    pc = np.array(np.where(v > 0)).transpose()
    plot_pointcloud(pc)
    # plot_voxel(v)
    # p = "/home/hfy/data/msra15/P0/4/000001_depth.bin"
    # s = get_raw_data(p)
    # pc = cal_pointcloud(s)
    # plot_pointcloud(pc)
    # # print pc[pc[:,2]>-240]
    # # for i in range(1520,1540):
    # #     t,l,ma = m[i]
    # #     plot_tsdf(t,l)
    #
    # for i in range(10):
    #     t = time()
    #     # pc, label = m.get_point_cloud(i)
    #     # plot_pointcloud(pc, label)
    #     data, label, max_p, mid_p = m[i]
    #     plot_tsdf(data, max_p, mid_p, label,0)
    #     plot_tsdf(data, max_p, mid_p, label,1)
    #     plot_tsdf(data, max_p, mid_p, label,2)
    #
    #     print time()-t
    # data,label = m[0]

    # pc_p = get_project_data(m[0][0])
    # plot_pointcloud(pc)
