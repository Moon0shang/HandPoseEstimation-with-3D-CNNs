import os
import os.path
import scipy.io as sio

from MSRA_pre import Read_MSRA
from visualize_3D import visualize


def gen_3d_points():

    read_data = Read_MSRA()
    read_data.read_all()


def show_points(file_dir):

    hand_points_raw = sio.loadmat(file_dir)
    dict_name = file_dir.split('/')[-1].split('.')[0]
    hand_points = hand_points_raw[dict_name]

    visualize(hand_points)


def run_TSDF():

    pass
