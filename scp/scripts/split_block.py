'''
Author: YunzhengSu && suyunzzzz1997@gmail.com
Date: 2022-07-25 23:41:28
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-27 15:57:23
FilePath: \pointcloud_ss_project\scp\scripts\split_block.py
Description: deprecated, split to blocks

'''
import argparse
from ast import arg
import os
from plyfile import PlyData, PlyElement
import numpy as np
from sklearn.decomposition import PCA
import shutil
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

parser = argparse.ArgumentParser()
# parser.add_argument("--rootdir", type=str, required=True)
parser.add_argument("--destdir", type=str, required=True)
# parser.add_argument("--test", action="store_true")
args = parser.parse_args()

save_dir = args.destdir

if os.path.exists(save_dir):
    # os.rmdir(save_dir)
    shutil.rmtree(save_dir)    #递归删除文件夹
    logger.info(f"remove {save_dir}")
os.makedirs(save_dir, exist_ok=True)
logger.info(f"makedirs {save_dir}")

# create the directory
filenames = ["whole_part1.npy"]
# filenames = ["whole_part1.txt"]
for filename in filenames:
    # if args.test:
    #     fname = os.path.join(args.rootdir, "test_10_classes", filename)
    # else:
    #     fname = os.path.join(args.rootdir, "training_10_classes", filename)
    fname = filename
    logger.info(f"load file: {fname}")

    # plydata = PlyData.read(fname)
    if filename.split('.')[-1]=='txt':
        arr_xyzrgbl = np.loadtxt(fname=fname, dtype=np.float128).reshape((-1,7))
    elif filename.split('.')[-1]=='npy':
        arr_xyzrgbl = np.load(file=fname).reshape((-1,7))
    else:
        raise NotImplementedError
    logger.info(f"load {fname} succeed.")

    x_mean = np.mean(arr_xyzrgbl[:,0])
    arr_xyzrgbl[:,0]-=x_mean
    arr_xyzrgbl[:,1]-=np.mean(arr_xyzrgbl[:,1])
    arr_xyzrgbl[:,2]-=np.mean(arr_xyzrgbl[:,2])
    logger.info("coordinate shift")

    arr_xyzrgbl[:,3:-1] = arr_xyzrgbl[:, 3:-1] / 255
    logger.info("color scale")

    # arr_xyzrgbl = np.array(arr_xyzrgbl, dtype=np.float128)
    
    logger.info(f"label class: {set(arr_xyzrgbl[:,-1].reshape((-1)))}")

   
    pts = arr_xyzrgbl
    del arr_xyzrgbl
    logger.info("delete arr-xyzrgbl")
    logger.info(f'pts: {pts}')
    # import sys
    # sys.exit()
    pca = PCA(n_components=1)
    pca.fit(pts[::300,:2])
    pts_new = pca.transform(pts[:,:2])
    hist, edges = np.histogram(pts_new, pts_new.shape[0]// 2500000)

    count = 0

    for i in range(1,edges.shape[0]):
        mask = np.logical_and(pts_new<=edges[i], pts_new>edges[i-1])[:,0]
        np.save(os.path.join(save_dir, os.path.splitext(filename)[0]+f"_{count}"), pts[mask])
        count+=1


    hist, edges = np.histogram(pts_new, pts_new.shape[0]// 2500000 -2, range=[(edges[1]+edges[0])//2,(edges[-1]+edges[-2])//2])

    for i in range(1,edges.shape[0]):
        mask = np.logical_and(pts_new<=edges[i], pts_new>edges[i-1])[:,0]
        np.save(os.path.join(save_dir, os.path.splitext(filename)[0]+f"_{count}"), pts[mask])
        count+=1
