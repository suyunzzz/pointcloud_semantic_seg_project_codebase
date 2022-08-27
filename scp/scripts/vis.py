'''
Author: YunzhengSu && suyunzzzz1997@gmail.com
Date: 2022-07-25 23:36:24
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-27 15:59:36
FilePath: \pointcloud_ss_project\scp\scripts\vis.py
Description: just for visualize

'''
from asyncio.log import logger
from operator import le, mod
import os
import sys
from turtle import left
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from glob import glob
import logging

# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(filename)s - lineno:%(lineno)d - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    13: (100., 85., 144.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

BLCOK_PATH = '/home/plusai/suyunz/dataset/part1/blocks'

def vis(data,label):
    '''
    :param data: n*3的矩阵
    :param label: n*1的矩阵
    :return: 可视化
    '''
    data=data[:,:3]
    
    assert data.shape[0] == label.shape[0]
    # labels=np.asarray(label)
    # max_label=labels.max()

    # # 颜色
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))[:,0,:]
    label = label.reshape((-1))
    logger.info(set(label))
    colors = [COLOR_MAP[i] for i in list(label)]
    colors = np.asarray(colors).reshape((-1,3))/255
    logger.info(colors.shape)

    pt1 = o3d.geometry.PointCloud()
    pt1.points = o3d.utility.Vector3dVector(data.reshape(-1, 3))
    pt1.colors=o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pt1],'label of cloud',width=1920,height=1080, left=0, top=0)

def visualize(arr, mode='label'):
    coord = arr[:,:3].reshape((-1,3))
    color = arr[:,3:-1].reshape(-1,3)
    label = arr[:,-1].reshape(-1,1)
    
    if mode=='label':
        vis(coord, label=label)
    elif mode=='color':
        pt1 = o3d.geometry.PointCloud()
        pt1.points = o3d.utility.Vector3dVector(coord.reshape(-1, 3))
        pt1.colors=o3d.utility.Vector3dVector(color[:, :3])

        o3d.visualization.draw_geometries([pt1],'color of cloud',width=1920,height=1080, left=0, top=0)




def main():
    
    # dir = '/home/plusai/suyunz/scp/dataset/part3/blocks'
    dir = 'E:\dataset-semantic-seg\scp\dataset\part3\\blocks'

    blocks = glob(f"{dir}/*.npy")
    logger.info(blocks)
    for block in blocks:
        logger.info(f"visualizing file:{block.split('/')[-1]}")
        arr = np.load(block)
        visualize(arr, mode='label')

    pass

if __name__ == "__main__":

    main()