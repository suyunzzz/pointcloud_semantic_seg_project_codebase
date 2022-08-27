'''
Author: YunzhengSu && suyunzzzz1997@gmail.com
Date: 2022-07-26 20:37:43
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-27 15:59:18
FilePath: \pointcloud_ss_project\scp\scripts\split_sample.py
Description: split sample to train and test

'''
from cgi import test
import shutil
import sys
import os
from matplotlib.cbook import ls_mapper
import numpy as np
from torch import group_norm, log, rand
import glob
import logging
import tqdm
import random

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

def split(src_dir, dst_dir):
    logger.info(f"src_dir:{src_dir}")
    logger.info(f"dst_dir:{dst_dir}")
    files = glob.glob(f'{src_dir}/*.npy')
    logger.info(files)

    train_dir = os.path.join(dst_dir, 'train_pointclouds')
    test_dir = os.path.join(dst_dir, 'test_pointclouds')
    if not os.path.exists(train_dir):
        logger.info(f"{train_dir} is not exist.")
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        logger.info(f"{test_dir} is not exist.")
        os.makedirs(test_dir)
    logger.info(f'train dir: {train_dir}')
    logger.info(f'test dir: {test_dir}')
    
    for file in tqdm.tqdm(files):
        logger.info(f'current file: {file}')
        rand_num = random.random()
        if rand_num > 0.8:
            logger.info(f'copy {file} to {test_dir}')
            shutil.copy(file, test_dir)
        else:
            shutil.copy(file, train_dir)
            logger.info(f'copy {file} to {train_dir}')


    pass

if __name__ == '__main__':
    root = '/media/s/TOSHIBA/dataset-semantic-seg/scp/dataset'
    list = ['part1', 'part2', 'part3']
    blocks_dir = 'blocks'
    suffix = '.npy'
    
    for dir in list:
        src_dir = root+'/'+dir+'/'+blocks_dir
        logger.info(src_dir)

        split(src_dir=src_dir, dst_dir='/media/s/TOSHIBA/dataset-semantic-seg/scp')


    