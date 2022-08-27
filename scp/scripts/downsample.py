'''
Author: YunzhengSu && suyunzzzz1997@gmail.com
Date: 2022-07-28 22:09:18
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-27 15:59:58
FilePath: \pointcloud_ss_project\scp\scripts\downsample.py
Description: just for downsample sample to reduce memory cost

'''
import os
import sys
from matplotlib.pyplot import axis
import open3d as o3d
import numpy as np
import logging
import glob
from scipy.fft import dst

from tqdm import tqdm

logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)          # debug不显示

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

def voxel(filename):
    xyzrgbl = np.load(filename )

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyzrgbl[:, :3])
    pc.colors = o3d.utility.Vector3dVector(xyzrgbl[:, 3:6])
    normals = np.zeros((xyzrgbl.shape[0], 3))
    normals[:, 0] = xyzrgbl[:, -1]
    pc.normals = o3d.utility.Vector3dVector(normals)
    pc_down = pc.uniform_down_sample(10)
    # logger.info(pc)
    # logger.info(index)
    logger.info(np.asarray(pc.points).shape)
    logger.info(np.asarray(pc_down.points).shape)

    # o3d.visualization.draw_geometries(
    #     [pc_down])

    xyz = np.asarray(pc_down.points).reshape((-1,3))
    rgb = np.asarray(pc_down.colors).reshape((-1,3))
    label = np.asarray(pc_down.normals).reshape((-1,3))[:, 0].reshape((-1,1))
    logger.info(xyz.shape)
    logger.info(rgb.shape)
    logger.info(label.shape)

    xyzrgbl = np.concatenate((xyz, rgb, label), axis=1)

    logger.info('xyzrgbl:{}'.format(xyzrgbl))
    return xyzrgbl


    pass


if  __name__ == '__main__':
    root = os.path.join('/media/s/TOSHIBA/dataset-semantic-seg/scp', "test_pointclouds")
    dst_dir = os.path.join('/media/s/TOSHIBA/dataset-semantic-seg/scp', "test_pointclouds_downsample")
    if not os.path.exists(dst_dir):
        logger.info(f"mkdir {dst_dir}")
        os.makedirs(dst_dir)
    for file in tqdm(glob.glob(f'{root}/*.npy')):
        base = file.split('.')[0]
        base_name = base.split('/')[-1]         
        logger.info(f"base_name: {base_name}")

        logger.info(f'load: {file}')
        xyzrgbl = voxel(filename=file)
        np.save(os.path.join(dst_dir, base_name), xyzrgbl)
        logger.info(f"save to {os.path.join(dst_dir, base_name)}")
