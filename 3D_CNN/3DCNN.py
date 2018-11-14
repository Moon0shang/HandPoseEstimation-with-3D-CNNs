import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as fun
import matplotlib.pyplot as plt
import numpy as np

"""
conv3d
relu
BN
dropout
FC: full connected
regression
"""


class CNN_3D(nn.Module):
    """3D CNN"""
    super(CNN_3D, self).__init__()
    self.Conv1 = nn.Sequential(
        nn.Conv3d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
    )
