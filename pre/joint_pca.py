"""
perform PCA on the transformed and normalized joint loacations 
in the ground truth of trainning dataset 
"""
import os
import os.path
import numpy as np


def PCA(data):
    """PCA output coeff, score, latent like matlab pca do"""
    [n, p] = data.shape
    mean_data = np.mean(data, axis=0)
    data1 = data - mean_data
    [u, sigma, coeff] = np.linalg.svd(data1)
    col = len(sigma)
    u = u[:, :col]
    coeff = np.transpose(coeff)
    score = np.multiply(u, sigma)
    sigma = sigma / np.sqrt(n - 1)

    if n <= p:
        sigma[n:p, 1] = 0
        score[:, n: p] = 0

    latent = np.power(sigma, 2).reshape(len(sigma), 1)

    return coeff, score, latent


def joint_pca():

    result = './result'
    subjects = sorted(os.listdir(result))[:9]
    gestures = sorted(os.listdir(os.path.join(
        result, subjects[0], 'ground_truth')))

    for test in range(9):
        joints_pca = np.empty(63)
        for sub in subjects:
            for ges in gestures:
                ground_truth = np.load(os.path.join(
                    result, sub, 'ground_truth', ges))
                ground_truth = ground_truth.reshape(ground_truth.shape[0], 63)

                if subjects[test] != sub:
                    joints_pca = np.vstack((joints_pca, ground_truth))

        joints_pca = joints_pca[1:]
        coeff, score, latent = PCA(joints_pca)
        pca_mean = np.mean(joints_pca, 0)
        try:
            os.mkdir('./PCA')
        except:
            print('failed create PCA')
        np.savez('./PCA', '%s' % test, pca_mean=pca_mean,
                 coeff=coeff, latent=latent)
