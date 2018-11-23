import numpy as np
from tqdm import tqdm
import time
import torch


a = np.random.randint(0, 100, [21, 3])
b = np.random.randint(0, 100, [3])
c = a - b
print(c)
