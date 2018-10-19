import numpy as np
import numba as nb
from params import *
from timeit import default_timer as timer
from numba import cuda

threadsperblock = VOXEL_RES
mm_threadsperblock = 128


@cuda.jit
def tsdf_kernel(vox_ori, voxel_len, l, r, t, b, trunc_dis, i, o):
    b_w = r-l
    x = cuda.threadIdx.x
    y = cuda.blockIdx.x
    z = cuda.blockIdx.y
    v_index = x + y*cuda.blockDim.x + z*cuda.gridDim.x*cuda.blockDim.x
    volume_size = VOXEL_RES**3
    if v_index >= volume_size:
        return
    # voxel center
    v_x = vox_ori[0]+x * voxel_len
    v_y = vox_ori[1]+y * voxel_len
    v_z = vox_ori[2]+z * voxel_len
    # voxel center in image frame
    q = -FOCAL / v_z
    pix_x = int((v_x * q + CENTER_X))
    pix_y = int((-v_y * q + CENTER_Y))
    o[0, z, y, x] = 0
    o[1, z, y, x] = 0
    o[2, z, y, x] = 0
    if pix_x < l or pix_x >= r or pix_y < t or pix_y >= b:
        return
    idx = (pix_y - t) * b_w + pix_x - l
    pix_depth = i[idx]
    if abs(pix_depth) < 1:
        return
    # closest surface point in world frame
    q = pix_depth/FOCAL
    w_x = (pix_x-CENTER_X)*q
    w_y = -(pix_y-CENTER_Y)*q
    w_z = -pix_depth
    tsdf_x = abs(v_x-w_x)/trunc_dis
    tsdf_y = abs(v_y-w_y)/trunc_dis
    tsdf_z = abs(v_z-w_z)/trunc_dis

    disTosurfaceMin = pow(tsdf_x * tsdf_x + tsdf_y *
                          tsdf_y + tsdf_z * tsdf_z, 0.5)
    # disTosurfaceMin = min(disTosurfaceMin / SURFACE_THICK, 1.0)
    if disTosurfaceMin > 1:
        tsdf_x = 1
        tsdf_y = 1
        tsdf_z = 1
    tsdf_x = min(tsdf_x, 1)
    tsdf_y = min(tsdf_y, 1)
    tsdf_z = min(tsdf_z, 1)

    # tsdf_x = 0 if tsdf_x == 1 else tsdf_x
    # tsdf_y = 0 if tsdf_y == 1 else tsdf_y
    # tsdf_z = 0 if tsdf_z == 1 else tsdf_z
    if w_z > v_z:
        tsdf_x = - tsdf_x
        tsdf_y = - tsdf_y
        tsdf_z = - tsdf_z

    o[0, z, y, x] = tsdf_x
    o[1, z, y, x] = tsdf_y
    o[2, z, y, x] = tsdf_z


@cuda.jit
def min_max_kernel(l, r, t, b, a, o):
    b_w = r-l
    tid = cuda.threadIdx.x
    min_p = cuda.shared.array(
        shape=(mm_threadsperblock, 3), dtype=nb.types.float32)
    max_p = cuda.shared.array(
        shape=(mm_threadsperblock, 3), dtype=nb.types.float32)
    pos = cuda.grid(1)
    x = pos % b_w + l
    y = pos / b_w + t
    cam_z = a[pos]
    if abs(cam_z) < 1:
        min_p[tid][0], min_p[tid][1], min_p[tid][2] = 99999, 99999, 99999
        max_p[tid][0], max_p[tid][1], max_p[tid][2] = -99999, -99999, -99999
    else:
        q = cam_z / FOCAL
        cam_x = q * (x - CENTER_X)
        cam_y = -q * (y - CENTER_Y)  # change to camera frame
        cam_z = -cam_z
        min_p[tid][0], min_p[tid][1], min_p[tid][2] = cam_x, cam_y, cam_z
        max_p[tid][0], max_p[tid][1], max_p[tid][2] = cam_x, cam_y, cam_z
    cuda.syncthreads()
    s = 1
    while s < cuda.blockDim.x:
        index = 2 * s * tid
        if index+s < cuda.blockDim.x:
            min_p[index][0] = min(min_p[index + s][0], min_p[index][0])
            min_p[index][1] = min(min_p[index + s][1], min_p[index][1])
            min_p[index][2] = min(min_p[index + s][2], min_p[index][2])
            max_p[index][0] = max(max_p[index + s][0], max_p[index][0])
            max_p[index][1] = max(max_p[index + s][1], max_p[index][1])
            max_p[index][2] = max(max_p[index + s][2], max_p[index][2])
        s *= 2
        cuda.syncthreads()
    if tid == 0:
        o[cuda.blockIdx.x][0] = min_p[0][0]
        o[cuda.blockIdx.x][1] = min_p[0][1]
        o[cuda.blockIdx.x][2] = min_p[0][2]
        o[cuda.blockIdx.x][3] = max_p[0][0]
        o[cuda.blockIdx.x][4] = max_p[0][1]
        o[cuda.blockIdx.x][5] = max_p[0][2]


def cal_tsdf_cuda(s):
    try:
        t1 = timer()
        w = s['header'][0]
        h = s['header'][1]
        l = s['header'][2]
        t = s['header'][3]
        r = s['header'][4]
        b = s['header'][5]
        b_w = r - l
        b_h = b - t

        # min max calculation
        blockdim = (s['data'].size + mm_threadsperblock - 1)/mm_threadsperblock
        d_depth = cuda.to_device(s['data'])
        d_mm = cuda.device_array([blockdim, 6], dtype=np.float32)
        min_max_kernel[blockdim, mm_threadsperblock](l, r, t, b, d_depth, d_mm)
        mm_p = np.empty([blockdim, 6], dtype=np.float32)
        mm_p = d_mm.copy_to_host()
        if True in np.isnan(mm_p[-1]):
            mm_p = mm_p[:-1]
        min_p = np.min(mm_p[:, :3], axis=0)
        max_p = np.max(mm_p[:, 3:6], axis=0)
        mid_p = (min_p + max_p) / 2
        len_e = max_p - min_p
        max_l = np.max(len_e)
        voxel_len = max_l / VOXEL_RES
        trunc_dis = (voxel_len * TRUC_DIS_T)
        vox_ori = mid_p - max_l / 2 + voxel_len / 2
        t2 = timer()

        # tsdf calculation #
        d_depth = cuda.to_device(s['data'])
        d_tsdf = cuda.device_array(
            [3, VOXEL_RES, VOXEL_RES, VOXEL_RES], dtype=np.float32)
        blockspergrid = [VOXEL_RES, VOXEL_RES]
        tsdf_kernel[blockspergrid, threadsperblock](
            vox_ori, voxel_len, l, r, t, b, trunc_dis, d_depth, d_tsdf)
        # tsdf = np.empty([VOXEL_RES, VOXEL_RES, VOXEL_RES, 3], dtype=np.float32)
        tsdf = d_tsdf.copy_to_host()
        t3 = timer()
        # print("time 1: %.4f, time 1+2: %.4f" % (t2 - t1, t3 - t1)
        return tsdf, max_l, mid_p
    except RuntimeWarning as w:
        print("warning caught: ", w)
        print("min:", min_p)
        print("max:", max_p)
        print("minmax:", mm_p)
        print("mid:", mid_p)
        print("len_e", len_e)
        print("max_l", max_l)
        # print ""
        return None


# return point cloud in camera coordination and tsdf data
def cal_tsdf(s):
    w = s['header'][0]
    h = s['header'][1]
    l = s['header'][2]
    t = s['header'][3]
    r = s['header'][4]
    b = s['header'][5]
    b_w = r-l
    b_h = b-t
    p_clouds = []
    minx = miny = minz = 99999
    maxx = maxy = maxz = -99999
    for y in range(t, b):
        for x in range(l, r):
            idx = (y-t) * b_w + x-l
            cam_z = s['data'][idx]
            if cam_z == 0:
                continue
            # pixel coor -> camera coor
            q = cam_z/FOCAL
            cam_x = q * (x-CENTER_X)
            cam_y = -q * (y-CENTER_Y)  # change to right hand axis
            cam_z = -cam_z
            p_clouds.append([cam_x, cam_y, cam_z])
            minx = cam_x if cam_x < minx else minx
            miny = cam_y if cam_y < miny else miny
            minz = cam_z if cam_z < minz else minz

            maxx = cam_x if cam_x > maxx else maxx
            maxy = cam_y if cam_y > maxy else maxy
            maxz = cam_z if cam_z > maxz else maxz
    npa = np.asarray(p_clouds, dtype=np.float32)
    min_p = np.array([minx, miny, minz], dtype=np.float32)
    max_p = np.array([maxx, maxy, maxz], dtype=np.float32)
    print(min_p, max_p)
    return None
    mid_p = (min_p+max_p)/2
    len_e = max_p-min_p
    max_l = np.max(len_e)
    voxel_len = max_l/VOXEL_RES
    trunc_dis = voxel_len*TRUC_DIS_T
    vox_ori = mid_p-max_l/2+voxel_len/2
    tsdf = np.ones((VOXEL_RES, VOXEL_RES, VOXEL_RES))

    # print VOXEL_RES*VOXEL_RES*VOXEL_RES,"loop 2 iter"
    for z in range(VOXEL_RES):
        for y in range(VOXEL_RES):
            for x in range(VOXEL_RES):
                vox_center = vox_ori + \
                    np.array([x*voxel_len, y*voxel_len, z*voxel_len])
                vox_depth = -vox_center[2]
                q = FOCAL/vox_depth
                pix_x = int(round(vox_center[0]*q+CENTER_X))
                pix_y = int(round(-vox_center[1]*q+CENTER_Y))
                if pix_x < l or pix_x >= r or pix_y < t or pix_y >= b:
                    continue
                idx = (pix_y-t)*b_w+pix_x-l
                # print vox_center,x,y,z,pix_x,pix_y,idx
                pix_depth = s['data'][idx]
                if pix_depth == 0:
                    continue
                diff = (pix_depth - vox_depth)/trunc_dis
                if diff >= 1 or diff <= -1:
                    continue
                tsdf[z, y, x] = diff

    # label = (label.reshape(-1,3) - mid_p)/max_l+0.5
    return tsdf


# point cloud in image coordination
def get_project_data(s):
    w = s['header'][0]
    h = s['header'][1]
    l = s['header'][2]
    t = s['header'][3]
    r = s['header'][4]
    b = s['header'][5]
    b_w = r - l
    b_h = b - t
    p_clouds = []
    for y in range(t, b):
        for x in range(l, r):
            idx = (y - t) * b_w + x - l
            z = s['data'][idx]
            if z == 0:
                continue
            p_clouds.append([x, y, z])

    npa = np.asarray(p_clouds, dtype=np.float32)
    return npa


if __name__ == "__main__":
    pass
    # sample = {'header': np.array([320,240,100,100,200,200],dtype=np.int32),
    #           'data': np.random.rand(100*100)*100+100}
    # label = np.ones(63)
    # # pc, tsdf, labels = cal_tsdf((sample, label))
    # tsdf, labels = cal_tsdf_cuda((sample, label))
