import os
import os.path
import numpy as np

DB_dir = '/home/x/DB/MSRA HandPoseDataset/cvpr15_MSRAHandGestureDB/'
SAVE_dir = './result'
os.mkdir(SAVE_dir)

subject = sorted(os.listdir(DB_dir))
gesture = sorted(os.listdir(os.path.join(DB_dir, subject[0])))
for sub in subject:
    sub_dir = os.path.join(SAVE_dir, sub)
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, 'num'))
    for ges in gesture:
        f_name = os.path.join(DB_dir, sub, ges, 'joint.txt')
        with open(f_name, 'r') as f:
            bin_num = int(f.readline())
        np.save(os.path.join(sub_dir, 'num', '%s.npy' % ges), bin_num)
