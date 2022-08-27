# NPM3D Example with ConvPoint

# add the parent folder to the python path to access convpoint library
from ast import arg
from fileinput import filename
import sys
sys.path.append('../../')
import open3d as o3d

import numpy as np
import argparse
from datetime import datetime
import os
import random
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
from PIL import Image

import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
import logging

from npm3d_seg_test import vis
# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


import utils.metrics as metrics
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors

from torch.utils.tensorboard import SummaryWriter
import argparse

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


def get_model(model_name, input_channels, output_channels, args):
    if model_name == "SegBig":
        from networks.network_seg import SegBig as Net
    return Net(input_channels, output_channels, args=args)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-s', help='Path to data folder')
    parser.add_argument('--nocolor', action="store_true")
    parser.add_argument("--model", default="SegBig", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--savedir", type=str, default="./results")
    parser.add_argument("--num_classes", default=20, type=int)          # 0~20, 20 can be ignored
    parser.add_argument('--block_size', help='Block size', type=float, default=8)
    parser.add_argument("--npoints", "-n", type=int, default=8192)
    parser.add_argument("--drop", default=0.5, type=float)

    args = parser.parse_args()

    logger.info(args)

    logger.info("get data...")
    pts, fts, lbs = get_data(args.filename, args)
    logger.info("get data done...")
    
    fts = fts.cuda()
    pts = pts.cuda()
    lbs = lbs.cuda()

    N_CLASSES = args.num_classes
    logger.debug(f"===> N_CLASSES: {N_CLASSES}")    # create model
    print("Creating the network...", end="", flush=True)
    input_c = 1
    if args.nocolor:
        input_c = 3
    else:
        input_c = 6
    net = get_model(args.model, input_channels=input_c, output_channels=N_CLASSES, args=args)
    if args.test:
        net.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
    net.cuda()

    if args.test:
        net.eval()

    print("Done")
    with torch.no_grad():

        outputs = net(fts, pts)
        output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
        target_np = lbs.cpu().numpy().copy()

    # vis
    vis(pts=fts.cpu().detach().numpy(), label=target_np)
    vis(pts=fts.cpu().detach().numpy(), label=output_np)

    cm = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
    logger.info("output_np:{}".format(output_np))
    logger.info("target_np:{}".format(target_np))

    logger.info("oa:{}, aa:{}, iou:{}".format(metrics.stats_overall_accuracy(cm), \
        metrics.stats_accuracy_per_class(cm), metrics.stats_iou_per_class(cm)[0]))
    

def get_data(filename, args):
    logger.info("loading file:{}".format(filename))
    pts = np.load(filename)
    pts = np.asarray(pts, dtype=np.float64).reshape((-1, 7))

    selected_points = pts[:, :-1]  # num_point * 6
    current_points = np.zeros((pts.shape[0], 9))  # num_point * 9
    current_points[:, 6] = selected_points[:, 0] / np.max(selected_points[:, 0])
    current_points[:, 7] = selected_points[:, 1] / np.max(selected_points[:, 1])
    current_points[:, 8] = selected_points[:, 2] / np.max(selected_points[:, 2])
    selected_points[:, 0] = selected_points[:, 0] - np.mean(selected_points[:, 0])
    selected_points[:, 1] = selected_points[:, 1] - np.mean(selected_points[:, 1])
    selected_points[:, 3:6] /= 255.0
    selected_points[:, 3:6] -= 0.5
    current_points[:, 0:6] = selected_points

    if args.nocolor:
        fts = current_points[:, 6:]            # n*3
    else:
        fts = current_points[:, 3:]            # n*6

    # get the labels
    lbs = pts[:,-1].astype(int)
    pts = pts[:, :3]

    pt_id = random.randint(0, pts.shape[0]-1)
    pt = pts[pt_id]
    logger.debug('lbs:{}'.format(lbs.shape))
    logger.debug('pts:{}'.format(pts.shape))
    logger.debug('fts:{}'.format(fts.shape))
    print("\n=================\n")

    # create the mask
    mask_x = np.logical_and(pts[:,0]<pt[0]+args.block_size/2, pts[:,0]>pt[0]-args.block_size/2)
    mask_y = np.logical_and(pts[:,1]<pt[1]+args.block_size/2, pts[:,1]>pt[1]-args.block_size/2)
    mask = np.logical_and(mask_x, mask_y)
    pts = pts[mask]
    lbs = lbs[mask]
    fts = fts[mask]
    logger.info('lbs:{}'.format(lbs.shape))
    logger.info('pts:{}'.format(pts.shape))
    logger.info('fts:{}'.format(fts.shape))
    # vis(pts, colors=None, label=lbs)


    # random selection
    choice = np.random.choice(pts.shape[0], args.npoints, replace=True)
    pts = pts[choice]
    lbs = lbs[choice]
    fts = fts[choice]

    fts = fts.astype(np.float64)

    vis(pts, colors=None, label=lbs)


    pts = torch.from_numpy(pts).float()
    fts = torch.from_numpy(fts).float()
    lbs = torch.from_numpy(lbs).long()

    logger.debug("===> return tensor")
    return pts, fts, lbs

    pass


if __name__ == '__main__':

    main()