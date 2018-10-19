# dataset params
DATA_EXT = '_depth.bin'
IMG_EXT = '_depth.jpg'
LABEL = 'joint.txt'
# GESTURES = ['1']
# GESTURES = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']
GESTURES = ['1', 'Y']
GESTURES_LEN = len(GESTURES)
MAX_SAMPLE_LEN = 500
JOINT_LEN = 21
BONE_LEN = 0
JOINT_POS_LEN = JOINT_LEN * 3

# tsdf params
VOXEL_RES = 32  # M
TRUC_DIS_T = 3
SURFACE_THICK = 0.1
MAXDIS = SURFACE_THICK + 20

# camera params
FOCAL = 241.42
CENTER_X = 160
CENTER_Y = 120

# train params. default value will be changed by passing args in console
DATA_DIR = '/home/hfy/data/msra15/'
# DATA_DIR = '/research/qxu2/alzeng/handpose/cvpr15_MSRAHandGestureDB/'
# DATA_DIR = '/home/alzeng/dataset/cvpr15_MSRAHandGestureDB/'
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
EPOCH_COUNT = 20
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 3
PRINT_FREQ = 10
WORKER = 0
DECAY_EPOCH = 5
DECAY_RATIO = 0.2
GPU_ID = '0'
# test params
ERROR_THRESH = 30.0  # 10mm


# heatmap params
IMG_SIZE = 128
HM_SIZE = 32
nSTACK = 2
nFEAT = 128
nModule = 1

# point cloud
PC_SIZE = 64
PC_GT_SIZE = 32

# data augmentation
ROTATE_ANGLE = 40
