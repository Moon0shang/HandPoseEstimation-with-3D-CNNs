"""
perform PCA on the transformed and normalized joint loacations 
in the ground truth of trainning dataset 

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
untested, if there are any bugs, fix it
"""
import os
import os.path
import numpy as np
# import scipy.io as sio


def joint_pca(aug=False):

    result = './result'
    subjects = sorted(os.listdir(result))[:9]

    gestures = sorted(os.listdir(os.path.join(
        result, subjects[0], 'ground_truth')))

    try:
        os.mkdir('./PCA')
    except:
        print('failed create PCA')

    for test in range(9):

        if aug:
            joints1 = read_joints(subjects, gestures)
            joints2 = read_joints(subjects, gestures, fs='ground_truth_aug')
            joints_pca = np.vstack(joints1, joints2)
            pca_mean, coeff = cal_pca(joint_pca, aug)
            np.savez('./PCA', '%s-aug.npz' % test,
                     pca_mean=pca_mean, coeff=coeff)
        else:
            joints_pca = read_joints(subjects, gestures)
            pca_mean, coeff = cal_pca(joint_pca, aug)
            np.savez('./PCA', '%s.npz' % test, pca_mean=pca_mean, coeff=coeff)

        # np.save(os.path.join('./PCA', test, 'joint-aug.mat'), joints_pca)
        # coeff, score, latent = PCA(joints_pca)
        # pca_mean = np.mean(joints_pca, 0)


def cal_pca(joints_pca, aug):

    pca_mean = np.mean(joints_pca, axis=0)
    nor = joints_pca - pca_mean
    [u, sigma, coeff] = np.linalg.svd(nor, full_matrices=False)
    coeff = np.transpose(coeff)

    return pca_mean, coeff


def nor_joint(subjects, gestures, fs=''):

    joints_pca = np.empty(63)
    for sub in subjects:
        for ges in gestures:
            ground_truth = np.load(os.path.join(
                result, sub, 'ground_truth%s' % fs, ges))
            data = np.load(os.path.join(
                result, sub, 'TSDF%s' % fs, ges))
            max_l = data['max_l']
            mid_p = data['mid_p']

            for data in len(ground_truth):
                gt_nor = (ground_truth[i]-mid_p[i])/max_l[i]+0.5

            gt_nor = gt_nor.reshape(ground_truth.shape[0], 63)

            if subjects[test] != sub:
                joints_pca = np.vstack((joints_pca, gt_nor))

    joints_pca = joints_pca[1:]

    return joints_pca


"""
there are some problem with my pca function
so use matlab pca instead!!!

########### Matlab program #########################
t={'1','2','3','4','5','6','7','8','9'};
pca_dir='/home/x/Codes/mat';
for test = 1:9
    file_dir = [pca_dir '/' t{test} '/joint.mat'];
    load(file_dir);
    [coeff,score,latent]=pca(joint);
    pca_mean=mean(joint,1);
    save([pca_dir '/' t{test} '-pca_mean.mat'],'pca_mean');
    save([pca_dir '/' t{test} '-coeff.mat'],'coeff');
    save([pca_dir '/' t{test} '-latent.mat'],'latent');
end
########### Matlab program #########################
"""


# def PCA(data):
#     """PCA output coeff, score, latent like matlab pca do"""
#     [n, p] = data.shape
#     mean_data = np.mean(data, axis=0)
#     data1 = data - mean_data
#     [u, sigma, coeff] = np.linalg.svd(data1)
#     col = len(sigma)
#     u = u[:, :col]
#     coeff = np.transpose(coeff)
#     score = np.multiply(u, sigma)
#     sigma = sigma / np.sqrt(n - 1)

#     if n <= p:
#         sigma[n:p, 1] = 0
#         score[:, n: p] = 0

#     latent = np.power(sigma, 2).reshape(len(sigma), 1)

#     return coeff, score, latent

# def merge_pca():

#     pca_dir = './mat'
#     try:
#         os.mkdir('./PCA')
#         print('create PCA file')
#     except:
#         print('failed create PCA file')
#     for i in range(1, 10):
#         pca_mean = sio.loadmat(os.path.join(
#             pca_dir, '%d-pca_mean.mat' % i))['pca_mean'][0]
#         coeff = sio.loadmat(os.path.join(pca_dir, '%d-coeff.mat' % i))['coeff']
#         latent = sio.loadmat(os.path.join(
#             pca_dir, '%d-latent.mat' % i))['latent']
#         np.savez(os.path.join('./PCA', '%d.npz' % (i-1)),
#                  pca_mean=pca_mean,
#                  coeff=coeff,
#                  latent=latent)
#         print('save %d files' % (i-1))


if __name__ == "__main__":
    joint_pca()
