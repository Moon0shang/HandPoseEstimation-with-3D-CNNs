import os
import os.path
import numpy as np

DATA_dir = './result'


def normalize(ground_truth, max_l, mid_p):

    gt_shap = ground_truth.shape
    if len(gt_shap) == 2:
        ground_truth = ground_truth.reshape(-1, 21, 3)

    joint_nor = np.empty(gt_shap)
    for i in range(gt_shap[0]):
        joint_nor[i] = (ground_truth[i] - mid_p[i]) / max_l[i] + 0.5

    return joint_nor


def read_joint(aug=''):

    subjects = sorted(os.listdir(DATA_dir))
    gt_dir = sorted(os.listdir(os.path.join(
        DATA_dir, subjects[0], 'ground_truth%s' % aug)))
    mm_dir = sorted(os.listdir(os.path.join(
        DATA_dir, subjects[0], 'TSDF%s' % aug)))
    gestures = len(gt_dir)
    for sub in subjects:
        sub_dir = os.path.join(DATA_dir, sub)
        try:
            os.mkdir(os.path.join(sub_dir, 'joint_nor%s' % aug))
        except:
            print('failed create joint nor file')
        for ges in range(gestures):
            gt_files = os.path.join(
                sub_dir, 'ground_truth%' % aug, gt_dir[ges])
            mm_files = os.path.join(sub_dir, 'TSDF%s' % aug, mm_dir[ges])
            joint = np.load(gt_files)
            datas = np.load(mm_files)
            max_l = datas['max_l']
            mid_p = datas['mid_p']
            joint_nor = normalize(joint, max_l, mid_p)
            np.save(os.path.join(sub_dir, 'joint_nor%s' % aug), joint_nor)


if __name__ == "__main__":
    read_joint()
    # read_joint(aug='_aug')
