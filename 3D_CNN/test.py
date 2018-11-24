import numpy as np
# from tqdm import tqdm
# import time
# import torch

a = np.load('./PCA/0.npz')
coeff = a['coeff']
print(coeff.shape)
