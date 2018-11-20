'''
PCA
'''
import numpy as np


def PCA(data):
    [n, p] = data.shape
    mean_data = np.mean(data, axis=0)
    data1 = data - mean_data
    r = np.min(n - 1, p)
    [u, sigma, coeff] = np.linalg.svd(data1)
    if n == 1:
        sigma = sigma(1)
    else:
        sigma = np.diag(sigma)
    sigma_t = np.transpose(sigma)
    score = np.multiply(u, sigma_t)
    sigma = np.dot(sigma, np.sqrt(n - 1))
    if n <= p:
        sigma[n:p, 1] = 0
        score[:, n:p] = 0
    latent = np.power(sigma)

    return coeff, score, latent


""" def PCA(X, k):  # k is the components you want
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X-mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i])
                 for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    return data
 """
# import numpy as np


# def PCA(data):
#     "calculate PCA"
#     # 1 calculate mean value
#     mean_value = np.mean(data)
#     std_value = np.std(data, ddof=1)
#     data_mean = (data - mean_value) / std_value
#     # 2 calculate features
#     # np.linalg.eig()
#     np.linalg.svd()
#     related_coef = np.cov(data_mean, bias=0)
#     f_value, f_vector = np.linalg.eig(related_coef)
#     coeff = np.rot90(f_vector).transpose()
#     f_value = np.rot90(f_value, k=2, axes=0)
#     latent = np.diag(f_value)[:, np.newaxis]
#     # 3 calculate ratio
#     ratio = 0
#     for i in range(data.shape[1]):
#         r = latent[i] / np.sum(latent)
#         ratio += r
#         if ratio >= 0.90:
#             break
#     # output args
#     score = coeff * data_mean

#     return coeff, latent, score


# class PCA(object):

#     def __init__(self, data):
#         self.data = data
#         self.col = data.shape[1]
#         # self.ouput_var()

#     # first step: normalize the matrix
#     def cal_mean(self):
#         mean_value = np.mean(self.data)
#         # 要想正确使用std， 必须设置参数ddof=1
#         # 数据量较大时结果误差可忽略，但数据量小时必须注意
#         std_value = np.std(self.data, ddof=1)
#         self.data = (self.data - mean_value) / std_value

#     # second: calculate coefficients, feature vector and feature value
#     def cal_features(self):
#         Relate_coef_func = np.corrcoef(self.data)
#         f_value, f_vector = np.linalg.eig(Relate_coef_func)
#         coeff = np.rot90(f_vector).transpose()
#         f_value = np.rot90(np.rot90(f_value))
#         latent = np.diag(f_value)[:, np.newaxis]

#         return coeff, latent

#     # third: calculate ratio
#     def cal_ratio(self, latent):
#         ratio = 0
#         for i in range(len(self.col)):
#             r = latent[i] / np.sum(latent)
#             ratio += r
#             if ratio >= 0.95:
#                 break

#     # ouput values
#     def pca(self):

#         self.cal_mean()
#         coeff, latent = self.cal_features()
#         self.cal_ratio(latent)
#         score = coeff * self.data

#         return coeff, latent, score
