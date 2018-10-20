import os
import os.path
import scipy.io as sio

from MSRA_pre import Read_MSRA
from visualize_3D import visualize
from TSDF import tsdf


def initial_data():
    '''
    only need to run once!
    '''
    read_data = Read_MSRA()
    read_data.read_all()


def show_points(file_dir):

    hand_points_raw = sio.loadmat(file_dir)
    hand_points = hand_points_raw['points']

    visualize(hand_points)


def run_TSDF(file_dir):

    hand_points_raw = sio.loadmat(file_dir)
    hand_points = hand_points_raw['points']
    depth_ori = hand_points_raw['depth_ori']
    pic_info = hand_points_raw['pic_info'][0]

    tsdf(hand_points, depth_ori, pic_info)


if __name__ == '__main__':

    select_file_dir = './results/P0/1/points000.mat'
    # show_points(select_file_dir)
    run_TSDF(select_file_dir)
