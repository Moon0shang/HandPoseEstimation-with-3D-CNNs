import os
import os.path
import time
import numpy as np

from tsdf_for import tsdf_f
from tsdf_numba import cal_tsdf_cuda

RAW_dir = ''
PC_dir = ''


def read_data():

    with open(RAW_dir) as f:
        header = np.fromfile(f, dtype=np.int32, count=6)
        depth = np.fromfile(f, dtype=np.float32)
    data = {
        'header': header,
        'depth': depth
    }
    pcs = np.load(PC_dir)
    pc = pcs[0]
    t1 = time.time()
    tsdf1, max_l1, mid_p1 = tsdf_f(data, pc)
    t2 = time.time()
    tsdf2, max_l2, mid_p2 = cal_tsdf_cuda(data)
    t3 = time.time()
    print('for loop time:%.4f' % (t2 - t1))
    print('numba time:%.4f' % (t3 - t2))
