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
    u = u[:, :2]
    coeff = np.transpose(coeff)
    score = np.multiply(u, sigma)
    sigma = sigma / np.sqrt(n - 1)

    if n <= p:
        sigma[n:p, 1] = 0

    latent = np.power(sigma, 2)  # .reshape(len(sigma),1)

    return coeff, score, latent


def joint_pca(joint_dir):

    subjects = sorted(os.listdir(joint_dir))
    gestures = sorted(os.listdir(os.path.join(joint_dir, subjects[0])))

    joints = []
    for sub in subjects:
        for ges in gestures:
            data_dir = os.path.join(joint_dir, sub, ges)
            ground_truth = np.load(data_dir).reshape(len(ground_truth), 3, 21)
            temp1 = None  # permuate()
            temp2 = temp1.reshape(len(ground_truth), 63)
            if sub != subjects[test_index]:
                joints = np.vstack(joints, temp2)
    coeff, score, latent = PCA(joints)
    PCA_mean = np.mean(joints, 0)
    # save datas
    np.savez(save_dir, coeff=coeff, PCA_mean=PCA_mean, latent=latent)
