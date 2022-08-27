# NPM3D Example with ConvPoint

# add the parent folder to the python path to access convpoint library
from cProfile import label
import io
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

# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


import utils.metrics as metrics
import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors

from torch.utils.tensorboard import SummaryWriter

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




def visualize_with_label(cloud, label, window_name="label"):
    assert cloud.shape[0] == label.shape[0]

    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(cloud)
    label = label.reshape((-1))
    logger.info(label)

    colors = np.asarray([COLOR_MAP[i] for i in list(label)]).reshape((-1, 3)) / 255
    pt.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pt], window_name ,width=500, height=500)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC



def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    print(pts_dest.shape)
    indices = nearest_neighbors.knn(pts_src, pts_dest, K, omp=True)
    print(indices.shape)
    if K==1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1],])
    return np.dot(batch_data, rotation_matrix)

# Part dataset only for training / validation
class PartDataset():

    def __init__ (self, filelist, folder,
                    training=False, 
                    iteration_number = None,
                    block_size=8,
                    npoints = 8192,
                    nocolor=False):

        self.folder = folder
        self.training = training
        self.filelist = filelist
        self.bs = block_size
        self.nocolor = nocolor

        self.npoints = npoints
        self.iterations = iteration_number
        self.verbose = False


        self.transform = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4)

    def __getitem__(self, index):
        
        # load the data
        index = random.randint(0, len(self.filelist)-1)
        logger.debug("file name: {}".format(os.path.join(self.folder, self.filelist[index])))
        pts = np.load(os.path.join(self.folder, self.filelist[index]))
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

        # fts = np.expand_dims(pts[:,3], 1).astype(np.float32)
        # logger.debug(f'no color:{self.nocolor}')
        if self.nocolor:
            fts = current_points[:, 6:]            # n*3
        else:
            fts = current_points[:, 3:]            # n*6
        
        # logger.debug(f'fts shape:{fts.shape}')

        # get the labels
        lbs = pts[:,-1].astype(int)

        # get the point coordinates
        pts = pts[:, :3]
        # vis(pts, colors=None, label=lbs)


        # pick a random point
        pt_id = random.randint(0, pts.shape[0]-1)
        pt = pts[pt_id]

        # create the mask
        mask_x = np.logical_and(pts[:,0]<pt[0]+self.bs/2, pts[:,0]>pt[0]-self.bs/2)
        mask_y = np.logical_and(pts[:,1]<pt[1]+self.bs/2, pts[:,1]>pt[1]-self.bs/2)
        mask = np.logical_and(mask_x, mask_y)
        pts = pts[mask]
        pts2 = selected_points[:, :3][mask]
        lbs = lbs[mask]
        fts = fts[mask]
        # vis(pts, colors=None, label=lbs)
        
        # random selection
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]
        pts2 = pts2[choice]
        lbs = lbs[choice]
        fts = fts[choice]

        # data augmentation
        if self.training:
            # random rotation
            pts = rotate_point_cloud_z(pts)
            pts2 = rotate_point_cloud_z(pts2)
        
        fts = fts.astype(np.float64)
        # fts = fts/255 - 0.5
        

        logger.debug('lbs:{}'.format(lbs.shape))
        logger.debug('pts:{}'.format(pts.shape))
        logger.debug('pts2:{}'.format(pts2.shape))
        logger.debug('fts:{}'.format(fts.shape))

        # vis(pts, colors=None, label=lbs)

        pts = torch.from_numpy(pts).float()
        pts2 = torch.from_numpy(pts2).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        logger.debug("===> return tensor")
        return pts, pts2, fts, lbs

    def __len__(self):
        return self.iterations

class PartDatasetTest():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:,0]<pt[0]+bs/2, self.xyzrgb[:,0]>pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:,1]<pt[1]+bs/2, self.xyzrgb[:,1]>pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__ (self, filename, folder,
                    block_size=8,
                    npoints = 8192,
                    test_step=0.8, nocolor=False):

        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.nocolor = nocolor
        self.filename = filename

        # load the points
        self.xyzrgb = np.load(os.path.join(self.folder, self.filename))

        step = test_step
        discretized = ((self.xyzrgb[:,:2]).astype(float)/step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float)*step

    def __getitem__(self, index):
        
        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzrgb[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # labels will contain indices in the original point cloud
        lbs = np.where(mask)[0][choice]

        # separate between features and points
        logger.debug(f'no color: {self.nocolor}')
        if self.nocolor:
            fts = np.ones((pts.shape[0], 1))
        else:
            fts = np.expand_dims(pts[:,3], 1).astype(np.float32)
            fts = fts/255 - 0.5

        pts = pts[:, :3].copy()

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs


        # ==============================
        # index = random.randint(0, len(self.filelist)-1)
        # logger.debug("file name: {}".format(os.path.join(self.folder, self.filelist[index])))
        # pts = np.load(os.path.join(self.folder, self.filelist[index]))
        # pts = np.asarray(pts, dtype=np.float64).reshape((-1, 6))
        

        # selected_points = pts[:, :-1]  # num_point * 6
        # current_points = np.zeros((pts.shape[0], 9))  # num_point * 9
        # current_points[:, 6] = selected_points[:, 0] / np.max(selected_points[:, 0])
        # current_points[:, 7] = selected_points[:, 1] / np.max(selected_points[:, 1])
        # current_points[:, 8] = selected_points[:, 2] / np.max(selected_points[:, 2])
        # selected_points[:, 0] = selected_points[:, 0] - np.mean(selected_points[:, 0])
        # selected_points[:, 1] = selected_points[:, 1] - np.mean(selected_points[:, 1])
        # selected_points[:, 3:6] /= 255.0
        # selected_points[:, 3:6] -= 0.5
        # current_points[:, 0:6] = selected_points

        # # fts = np.expand_dims(pts[:,3], 1).astype(np.float32)
        # logger.debug(f'no color:{self.nocolor}')
        # if self.nocolor:
        #     fts = current_points[:, 6:]            # n*3
        # else:
        #     fts = current_points[:, 3:]            # n*6
        
        # logger.debug(f'fts shape:{fts.shape}')
     
        # # get the point coordinates
        # pts = pts[:, :3]
        # # vis(pts, colors=None, label=lbs)


        # # pick a random point
        # pt_id = random.randint(0, pts.shape[0]-1)
        # pt = pts[pt_id]

        # # create the mask
        # mask_x = np.logical_and(pts[:,0]<pt[0]+self.bs/2, pts[:,0]>pt[0]-self.bs/2)
        # mask_y = np.logical_and(pts[:,1]<pt[1]+self.bs/2, pts[:,1]>pt[1]-self.bs/2)
        # mask = np.logical_and(mask_x, mask_y)
        # pts = pts[mask]
        # pts2 = selected_points[:, :3][mask]
        # lbs = lbs[mask]
        # fts = fts[mask]
        # # vis(pts, colors=None, label=lbs)
        
        # # random selection
        # choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        # pts = pts[choice]
        # pts2 = pts2[choice]
        # lbs = np.where(mask)[0][choice]         # index
        # # lbs = lbs[choice]
        # fts = fts[choice]

        # # data augmentation
        # if self.training:
        #     # random rotation
        #     pts = rotate_point_cloud_z(pts)
        #     pts2 = rotate_point_cloud_z(pts2)
        
        # fts = fts.astype(np.float64)
        # # fts = fts/255 - 0.5
        

        # logger.debug('lbs:{}'.format(lbs))
        # logger.debug('pts:{}'.format(pts))
        # logger.debug('pts2:{}'.format(pts2))
        # logger.debug('fts:{}'.format(fts))

        # vis(pts, colors=None, label=lbs)

        # pts = torch.from_numpy(pts).float()
        # pts2 = torch.from_numpy(pts2).float()
        # fts = torch.from_numpy(fts).float()
        # lbs = torch.from_numpy(lbs).long()

        # return pts, pts2, fts, lbs

    def __len__(self):
        return len(self.pts)

def get_model(model_name, input_channels, output_channels, args):
    if model_name == "SegBig":
        from networks.network_seg import SegBig as Net
    return Net(input_channels, output_channels, args=args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', '-s', help='Path to data folder')
    parser.add_argument("--savedir", type=str, default="./results")
    parser.add_argument('--block_size', help='Block size', type=float, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--iter", "-i", type=int, default=1000)
    parser.add_argument("--npoints", "-n", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nocolor", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--test_step", default=0.5, type=float)
    parser.add_argument("--model", default="SegBig", type=str)
    parser.add_argument("--drop", default=0.5, type=float)
    parser.add_argument("--num_classes", default=10, type=int)          # 0~20, 20 can be ignored
    parser.add_argument("--ignore_index", default=-100, type=int)          # 0~20, 21 can be ignored

    args = parser.parse_args()

    # makedirs
    if not os.path.exists(args.savedir):
        logger.debug(f'{args.savedir} is not exist, makedirs.')
        os.makedirs(args.savedir)

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    root_folder = os.path.join(args.savedir, "{}_{}_nocolor{}_drop{}_{}".format(
        args.model, args.npoints, args.nocolor, args.drop, time_string))

    from torch.utils.tensorboard import SummaryWriter
    
    # 将 train accuracy 保存到 "tensorboard/train" 文件夹
    log_dir = os.path.join(root_folder, 'train')
    train_writer = SummaryWriter(log_dir=log_dir)
    # 将 test accuracy 保存到 "tensorboard/test" 文件夹
    log_dir = os.path.join(root_folder, 'test')
    test_writer = SummaryWriter(log_dir=log_dir)
    

     # create the filelits (train / val) according to area
    print("Create filelist...", end="")
    train_dir = os.path.join(args.rootdir, "train_pointclouds_downsample")
    filelist_train = [dataset for dataset in os.listdir(train_dir)]
    test_dir = os.path.join(args.rootdir, "test_pointclouds_downsample")
    filelist_test = [dataset for dataset in os.listdir(test_dir)]
    print(f"done, {len(filelist_train)} train files, {len(filelist_test)} test files")

    N_CLASSES = args.num_classes
    logger.debug(f"===> N_CLASSES: {N_CLASSES}")

    # create model
    print("Creating the network...", end="", flush=True)
    input_c = 1
    if args.nocolor:
        input_c = 3
    else:
        input_c = 6
    net = get_model(args.model, input_channels=input_c, output_channels=N_CLASSES, args=args)
    if args.test:
        time_dir = "SegBig_8192_nocolorFalse_drop0.5_2022-07-27-08-47-23"
        net.load_state_dict(torch.load(os.path.join(args.savedir, time_dir, "state_dict.pth")))
    net.cuda()

    print("Done")


    if not args.test:

        print("Create the datasets...", end="", flush=True)

        ds = PartDataset(filelist_train, train_dir,
                                training=True, block_size=args.block_size,
                                iteration_number=args.batch_size*args.iter,
                                npoints=args.npoints,
                                nocolor=args.nocolor)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.threads
                                            )
        print("Done")


        print("Create optimizer...", end="", flush=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        print("Done")
        
        # create the root folder
        os.makedirs(root_folder, exist_ok=True)
        
        # create the log file
        logs = open(os.path.join(root_folder, "log.txt"), "w")

        # iterate over epochs
        batch_count = 0

        for epoch in range(args.epochs):

            #######
            # training
            net.train()

            train_loss = 0
            cm = np.zeros((N_CLASSES, N_CLASSES))
            t = tqdm(train_loader, ncols=100, desc="Epoch {}".format(epoch))
            for pts, pts2, features, seg in t:
                
                logger.debug(f"feats: {features}")
                features = features.cuda()

                logger.debug("pts:{}".format(pts.shape))
                pts = pts.cuda()
                seg = seg.cuda()
                
                optimizer.zero_grad()
                outputs = net(features, pts)
                loss =  F.cross_entropy(outputs.view(-1, N_CLASSES), seg.view(-1), ignore_index=args.ignore_index)
                loss.backward()
                optimizer.step()

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
                target_np = seg.cpu().numpy().copy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
                cm += cm_

                oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
                aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.5f}"
                iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"

                train_loss += loss.detach().cpu().item()

                t.set_postfix(OA=wblue(oa), AA=wblue(aa), IOU=wblue(iou), LOSS=wblue(f"{train_loss/cm.sum():.4e}"))

                # log
                train_writer.add_scalar('OA', metrics.stats_overall_accuracy(cm), batch_count)
                train_writer.add_scalar('AA', metrics.stats_accuracy_per_class(cm)[0], batch_count)
                train_writer.add_scalar('IoU', metrics.stats_iou_per_class(cm)[0], batch_count)
                
                batch_count+=1
            # save the model
            torch.save(net.state_dict(), os.path.join(root_folder, "state_dict.pth"))

            # write the logs
            logs.write(f"epoch:{epoch} oa:{oa} aa:{aa} iou:{iou}\n")
            logs.flush()

            # log
            train_writer.add_scalar('epoch IoU', metrics.stats_iou_per_class(cm)[0], epoch)
            train_writer.add_scalar('epoch IoU', metrics.stats_overall_accuracy(cm), epoch)


        logs.close()

    else:
        print("Testing on NPM3D")
        net.eval()
        
        ds = PartDataset(filelist_test, test_dir,
                        training=False, block_size=args.block_size,
                        iteration_number=args.batch_size*args.iter,
                        npoints=args.npoints,
                        nocolor=args.nocolor)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.threads
                                        )
        
        whole_cm = 0
        with torch.no_grad():
            t = tqdm(loader, ncols=80)
            for pts, pts2, features, lbs in t:
                
                features = features.cuda()
                pts = pts.cuda()
                outputs = net(features, pts)

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()[0,:]
                target_np = lbs.cpu().numpy().copy()[0,:]

                cm = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
                whole_cm+=cm

                oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
                aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.5f}"
                iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"
                print(oa, aa, iou)
                logger.info("oa:{}, aa:{}, iou:{}".format(metrics.stats_overall_accuracy(cm),\
                    metrics.stats_accuracy_per_class(cm)[0], metrics.stats_iou_per_class(cm)[0]))

                logger.info("output_np:{}".format(output_np.shape))
                logger.info("target_np:{}".format(target_np.shape))

                pts = pts.cpu().detach().numpy()[0, :, :]
                logger.info('pts shape:{}'.format(pts.shape))

                logger.info("pred")
                # vis(pts, colors=None, label=output_np, window_name='pred')
                logger.info("target")
                # vis(pts, colors=None, label=target_np, window_name='target')

                # sys.exit()
        
        oa = f"{metrics.stats_overall_accuracy(whole_cm):.5f}"
        aa = f"{metrics.stats_accuracy_per_class(whole_cm)[0]:.5f}"
        iou = f"{metrics.stats_iou_per_class(whole_cm)[0]:.5f}"
        print("=====\n\n")
        print(oa, aa, iou)
        print("class iou:{}".format( metrics.stats_iou_per_class(whole_cm)[1]))


def vis(pts, colors, label, window_name='label'):
    pts = np.asarray(pts).reshape((-1,3))
    label = np.asarray(label).reshape((-1,1))
    assert isinstance(pts, np.ndarray),f'pt is not np.array'
    
    logger.debug(pts.shape)
    if colors is not None:
        logger.debug("visualize with color...")
        colors = np.asarray(colors).reshape((-1,3))
        assert pts.shape[0] == colors.shape[0]
        pt = o3d.geometry.PointCloud()
        pt.points = o3d.utility.Vector3dVector(pts)
        pt.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pt], 'color' ,width=500, height=500)
    elif label is not None:
        logger.debug("visualize with label...")
        visualize_with_label(pts, label, window_name)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))
