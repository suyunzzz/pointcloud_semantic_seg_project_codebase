'''
Author: YunzhengSu && suyunzzzz1997@gmail.com
Date: 2022-07-25 23:36:24
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-27 15:56:04
FilePath: \pointcloud_ss_project\scp\scripts\joint.py
Description: just for merge txt file to generate whole scene, and then split a scene to some blocks 

'''
from distutils import text_file
from fileinput import filename
from genericpath import isdir
import os
import sys
from unicodedata import name
import numpy as np
import argparse
import logging
from sklearn.decomposition import PCA
import shutil

# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

def get_label(file_name:str):
    file_name = file_name.split('.')[-2]
    logger.info(f'filename:{file_name}')
    name_list = file_name.split('_')
    logger.info(name_list)
    num_str = name_list[3]
    if num_str == 'Ground':
        return 19
    elif num_str == 'Unknow':
        return 20
    try:
        return int(num_str)
    except ValueError as e:
        logger.error(repr(e))
        return 19


'''
description: merge txt file to whole scene
param {*} dir_name
param {*} final_filename
return {*}
'''
def joint_file(dir_name, final_filename):
    txt_list = os.listdir(dir_name)
    logger.info(txt_list)
    count=1
    len_num = len(txt_list)
    arr_full_class = []
    for file in txt_list:
        logger.info(file)
        if file=='scripts':
            continue

        txtfile_path = os.path.join(dir_name, file)
        # if os.path.isdir(text_file):
        #     logger.info(f"===> {txtfile_path} is a directory.")
        #     continue
        logger.info(f"num:{count}/{len_num}, FILE:{txtfile_path}")
        count+=1

        arr = np.loadtxt(txtfile_path, dtype=np.float64)
        arr_xyzrgbl = np.asarray(arr, dtype=np.float64)[:,:-3].reshape((-1,6))

        # get label
        file_label = get_label(file)
        
        logger.info(f"file_label: {file_label}")
        current_label_list = [file_label for _ in range(arr.shape[0])]

        current_label = np.asarray(current_label_list).reshape((-1,1))
        logger.info(f"current label:{current_label}")

        # sys.exit()
        arr_xyzrgbl = np.concatenate((arr_xyzrgbl, current_label.reshape((-1,1))), axis=1)
        arr_xyzrgbl = np.asarray(arr_xyzrgbl, dtype=np.float64)
        logger.info(f"arr_xyzrgbl shape: {arr_xyzrgbl.shape, arr_xyzrgbl.dtype}")

        arr_full_class.append(arr_xyzrgbl)
    
    # arr_whole_scene_xyzrgbl = np.asarray(arr_full_class, dtype=np.float32)
    
    arr_whole_scene_xyzrgbl = np.concatenate(tuple(arr_full_class), axis=0, dtype=np.float64)
    arr_whole_scene_xyzrgbl = arr_whole_scene_xyzrgbl.reshape((-1, 7))
    logger.info(f"arr_whole_scene_xyzrgbl shape:{arr_whole_scene_xyzrgbl.shape, arr_whole_scene_xyzrgbl.dtype}")
    
    np.save(final_filename, arr_whole_scene_xyzrgbl)
    logger.info(f"save file:{final_filename}")

    return arr_whole_scene_xyzrgbl


'''
description: split current whole scene to blocks
param {*} arr_xyzrgbl
param {*} save_dir
param {*} filename
return {*}
'''
def split_blocks(arr_xyzrgbl, save_dir, filename):
    x_mean = np.mean(arr_xyzrgbl[:,0])
    arr_xyzrgbl[:,0]-=x_mean
    arr_xyzrgbl[:,1]-=np.mean(arr_xyzrgbl[:,1])
    arr_xyzrgbl[:,2]-=np.mean(arr_xyzrgbl[:,2])
    logger.info("coordinate shift")

    arr_xyzrgbl[:,3:-1] = arr_xyzrgbl[:, 3:-1] / 255
    logger.info("color scale")

    # arr_xyzrgbl = np.array(arr_xyzrgbl, dtype=np.float64)
    
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--joint_filename", type=str, required=True)
    parser.add_argument("--joint_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    
    args = parser.parse_args()
    # joint_file("/home/plusai/suyunz/dataset/part1/txt")
    arr = joint_file(args.joint_dir, args.joint_filename)
    logger.info("joint file done.")

    save_dir = args.save_dir
    if os.path.exists(save_dir):
        # os.rmdir(save_dir)
        shutil.rmtree(save_dir)    #递归删除文件夹
        logger.info(f"remove {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"makedirs {save_dir}")

    # split blocks
    split_blocks(arr, save_dir, filename=args.joint_filename)
    logger.info("split_blocks done.")

    

    pass


if __name__=='__main__':

    main()