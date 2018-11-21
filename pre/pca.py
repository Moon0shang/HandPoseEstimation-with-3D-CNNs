'''
PCA module
'''
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

'''
function [coeff, score, latent, tsquare] = ppp(x,econFlag)
if nargin < 2, econFlag = 0; end
[n,p] = size(x);
# 矩阵减
x0 = bsxfun(@minus,x,mean(x,1));
r = min(n-1,p); % max possible rank of X0
[U,sigma,coeff] = svd(x0,econFlag); % put in 1/sqrt(n-1) later
if n == 1 % sigma might have only 1 row
    sigma = sigma(1);
else
    sigma = diag(sigma);
end

# 矩阵乘
score = bsxfun(@times,U,sigma转置); % == x0*coeff
sigma = sigma ./ sqrt(n-1);
if n <= p
    if isequal(econFlag, 'econ')
        sigma(n,:) = []; % make sure this shrinks as a column
        coeff(:,n) = [];
        score(:,n) = [];
    else
        sigma(n:p,1) = 0; % make sure this extends as a column
        score(:,n:p) = 0;
    end
end

latent = sigma.^2;
'''
