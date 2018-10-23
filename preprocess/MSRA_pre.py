import os
import os.path
import scipy.io as sio
import numpy as np
import struct

# read MSRA dataset
if os.name == 'nt':
    try:
        dataset_dir = 'D:/DB/MSRA HandPoseDataset/cvpr15_MSRAHandGestureDB'
    except:
        pass
    else:
        dataset_dir = 'C:/DB/MSRA HandPoseDataset/cvpr15_MSRAHandGestureDB'

else:
    dataset_dir = '/home/x/DB/MSRA HandPoseDataset/cvpr15_MSRAHandGestureDB/'

save_dir = './results'

subject_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
gesture_names = ['1', '2', '3', '4', '5', '6', '7', '8',
                 '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']


class Read_MSRA(object):

    def __init__(self):

        self.MSRA_valid = self.read_valid()

    def read_valid(self):

        raw = sio.loadmat('./msra_valid.mat')
        data = raw['msra_valid']

        return data

    def read_all(self):

        # create results file
        try:
            os.mkdir(save_dir)
        except:
            print('results already exist!')
        else:
            print('create directory results')

        for sub in range(len(subject_names)):

            # create subject files
            try:
                os.mkdir(os.path.join(save_dir, subject_names[sub]))
            except:
                print('%s already exist!' % subject_names[sub])
            else:
                print('create directory %s' % subject_names[sub])

            for ges in range(len(gesture_names)):

                ges_dir = os.path.join(
                    dataset_dir, subject_names[sub], gesture_names[ges])

                self.save_ges_dir = os.path.join(
                    save_dir, subject_names[sub], gesture_names[ges])

                # read ground truth
                [ground_truth, frame_num] = self.read_ground_truth(ges_dir)

                # create gesture files
                try:
                    os.mkdir(self.save_ges_dir)
                except:
                    print('%s already exist!' %
                          (subject_names[sub]+'/'+gesture_names[ges]))
                else:
                    print('create directory %s' %
                          (subject_names[sub]+'/'+gesture_names[ges]))

                # read depth files and save them
                [jnt_xyz, hand_points] = self.read_all_depth(
                    ges_dir, frame_num, sub, ges, ground_truth)

        return jnt_xyz, hand_points

    def read_ground_truth(self, ges_dir):

        joint_txt = os.path.join(ges_dir, 'joint.txt')

        with open(joint_txt, 'r') as f:
            frame_num = int(f.readline())
        # 跳过第一行，不然会显示numpy 列对不上，第一行只有 frame_num 一个数值，比下面的少
        truth_data = np.loadtxt(joint_txt, skiprows=1)
        ground_truth = truth_data.reshape(frame_num, 21, 3)
        # 因为深度图z是正的，但是label里面z却是负数
        ground_truth[:, :, 2] = -ground_truth[:, :, 2]

        return ground_truth, frame_num

    def read_all_depth(self, ges_dir, frame_num, sub, ges, ground_truth):

        num = 0
        for i in os.listdir(ges_dir):
            if os.path.splitext(i)[1] == '.bin':
                num += 1

        valid = self.MSRA_valid[sub, ges]
        # hand_points_3d = np.zeros((frame_num, sample_num, 3))
        for frm in range(num):
            if not valid[frm]:
                continue
            [hand_points, hand_ori, pic_info] = self.read_conv_bin(
                ges_dir, frm)
            self.save_mat('points%03d' % frm, hand_points, hand_ori, pic_info)
            jnt_xyz = np.squeeze(ground_truth[frm, :, :])

        self.save_mat('joint', jnt_xyz)

        return jnt_xyz, hand_points

        # read binary files

    def read_conv_bin(self, ges_dir, frm):

        with open(ges_dir + '/' + str('%06d' % frm) + '_depth.bin', 'rb') as f:
            img_width = struct.unpack('I', f.read(4))[0]
            img_height = struct.unpack('I', f.read(4))[0]

            bb_left = struct.unpack('I', f.read(4))[0]
            bb_top = struct.unpack('I', f.read(4))[0]
            bb_right = struct.unpack('I', f.read(4))[0]
            bb_bottom = struct.unpack('I', f.read(4))[0]
            bb_width = bb_right - bb_left
            bb_height = bb_bottom - bb_top
            valid_pixel_num = bb_width * bb_height

            hand_depth = struct.unpack(
                'f'*valid_pixel_num, f.read(4*valid_pixel_num))

        fFocal_msra = 241.42
        hand_3d = np.zeros((3, valid_pixel_num))

        hand_depth = np.array(hand_depth, dtype=np.float32)

        # '-' get on the right position
        hand_3d[2, :] = - hand_depth

        hand_depth = hand_depth.reshape(bb_height, bb_width)

        h_matrix = np.array([i for i in range(bb_height)], dtype=np.float32)
        w_matrix = np.array([i for i in range(bb_width)], dtype=np.float32)
        for h in range(bb_height):
            hand_3d[0, (h*bb_width):((h+1)*bb_width)] = np.multiply((w_matrix +
                                                                     bb_left-(img_width / 2)), hand_depth[h, :]) / fFocal_msra

        for w in range(bb_width):
            idx = [(hi * bb_width + w) for hi in range(bb_height)]
            # '-' get on the right position
            hand_3d[1, idx] = -np.multiply(
                (h_matrix+bb_top - (img_height / 2)), hand_depth[:, w]) / fFocal_msra

        hand_ori = hand_3d
        """ 
        can use but need more time with CPU threads
        
        for ii in range(bb_height):
            for jj in range(bb_width):

                # 0 < ii < bb_height - 1
                # 0 < jj < bb_weight - 1
                # 0 < idx < h * w -1

                idx = ii * bb_width + jj
                # x lable
                hand_3d[idx, 0] = (jj - img_width/2 + bb_left) * \
                    hand_depth[ii, jj] / fFocal_msra
                # y lable
                hand_3d[idx, 1] = (ii - img_height/2 + bb_top) * \
                    hand_depth[ii, jj] / fFocal_msra
                # z lable
                hand_3d[idx, 2] = hand_depth[ii, jj] 
        """

        valid_idx = []

        for num in range(valid_pixel_num):
            if any(hand_3d[:, num]):
                valid_idx.append(num)

        hand_points = hand_3d[:, valid_idx]

        pic_info = [fFocal_msra, img_height, img_width,
                    bb_top, bb_bottom, bb_left, bb_right, bb_width, bb_height]

        return hand_points, hand_ori, pic_info

    def save_mat(self, name, datas, hand_ori=0, pic_info=0):

        if pic_info == 0:
            sio.savemat(self.save_ges_dir + '/%s.mat' % name, {name: datas})
        else:
            sio.savemat(self.save_ges_dir + '/%s.mat' % name, {'points': datas,
                                                               'pic_info': pic_info,
                                                               'hand_ori': hand_ori})


if __name__ == '__main__':
    r = Read_MSRA()
    r.read_all()
