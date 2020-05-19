"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.utils.data as data
import os.path
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_pointcloud_loader(path):
    n_vec = 4
    dtype = np.float32
    lidar_pc_raw = np.fromfile(path, dtype)
    return lidar_pc_raw.reshape((-1, n_vec))


def pointcloud_loader_kitti(path):
    n_vec = 4
    dtype = np.float32
    lidar_pc_raw = np.fromfile(path, dtype)
    return lidar_pc_raw.reshape((-1, n_vec))


def pointcloud_loader_kitti_fov(path):
    n_vec = 4
    dtype = np.float32
    lidar_pc_raw = np.fromfile(path, dtype)
    lidar_pc_raw_reshaped = lidar_pc_raw.reshape((-1, n_vec))
    lidar_pc_raw_reshaped_fov = remove_fov_points(lidar_pc_raw_reshaped)
    return lidar_pc_raw_reshaped_fov

def pointcloud_loader_lyftkitti(path):
    n_vec = 4
    dtype = np.float32
    lidar_pc_raw = np.fromfile(path, dtype)
    lidar_pc_raw = lidar_pc_raw.reshape((-1, n_vec))
    lidar_pc_raw[:, 3] = 0.0  # lyft doesn't provide intensity values, so they are always set to 100.0
    return lidar_pc_raw


def pointcloud_loader_lyft(path):
    n_vec = 5
    dtype = np.float32
    lidar_pc_raw = np.fromfile(path, dtype)
    lidar_pc = lidar_pc_raw.reshape((-1, n_vec))
    return lidar_pc


def pointcloud_loader_audi(path):
    lidar_pc_raw = np.load(path)
    lidar_pc = np.zeros([lidar_pc_raw['points'].shape[0], 4])
    lidar_pc[:, :3] = lidar_pc_raw['points']
    lidar_pc[:, 3] = lidar_pc_raw['reflectance'] / 255.
    return lidar_pc


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where(
            (PointCloud[:, 0] >= minX) &
            (PointCloud[:, 0] <= maxX) &
            (PointCloud[:, 1] >= minY) &
            (PointCloud[:, 1] <= maxY) &
            (PointCloud[:, 2] >= minZ) &
            (PointCloud[:, 2] <= maxZ)
            )
    PointCloud = PointCloud[mask]
    PointCloud[:, 2] = PointCloud[:, 2] - minZ
    return PointCloud


def remove_fov_points(lidar_data):
    b = lidar_data.copy()
    pts_3d_velo = b[:, 0:3]

    V2C = np.array([
            [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
            [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
            [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01]
    ])
    R0 = np.array([
            [0.99992475, 0.00975976, -0.00734152],
            [-0.0097913, 0.99994262, -0.00430371],
            [0.00729911, 0.0043753, 0.99996319]
    ])
    P = np.array([[719.787081, 0., 608.463003, 44.9538775],
                      [0., 719.787081, 174.545111, 0.1066855],
                      [0., 0., 1., 3.0106472e-03]
    ])

    # project 3D lidar to 2D camera coordinates
    pts_3d_velo = np.hstack((pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1), dtype=np.float32)))
    pts_3d_ref = np.dot(pts_3d_velo, np.transpose(V2C))
    pts_3d_rect = np.transpose(np.dot(R0, np.transpose(pts_3d_ref)))
    pts_3d_rect = np.hstack((pts_3d_rect, np.ones((pts_3d_rect.shape[0], 1), dtype=np.float32)))
    pts_2d_ = np.dot(pts_3d_rect, np.transpose(P))  # nx3
    pts_2d_[:, 0] /= pts_2d_[:, 2]
    pts_2d_[:, 1] /= pts_2d_[:, 2]
    pts_2d = pts_2d_[:, 0:2]

    # crop FOV
    fov_inds = (
            (pts_2d[:, 0] < 1000)
            & (pts_2d[:, 0] >= 200)
    )
    fov_inds = fov_inds & (b[:, 0] > 6.0)
    b2 = b[fov_inds, :]
    return b2


def makeBVFeature(PointCloud_, BoundaryCond, img_height, img_width, Discretization):
    Height = img_height + 1
    Width = img_width + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    # because of the rounding of points, there are many identical points
    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]

    # Some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(BoundaryCond['maxZ'] - BoundaryCond['minZ']))
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height
    # heightMap_normalized = (heightMap - BoundaryCond['minZ'])/abs(BoundaryCond['maxZ']-BoundaryCond['minZ'])  # Normalize to [0, 1]

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # 'counts': The number of times each of the unique values comes up in the original array
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((Height - 1, Width - 1, 3))
    RGB_Map[:, :, 0] = densityMap[0:img_height, 0:img_width]  # r_map
    # RGB_Map[:, :, 1] = heightMap_normalized[0:img_height, 0:img_width]  # g_map
    RGB_Map[:, :, 1] = heightMap[0:img_height, 0:img_width]  # g_map
    RGB_Map[:, :, 2] = intensityMap[0:img_height, 0:img_width]  # / 255  # b_map
    return RGB_Map


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

PC_EXTENSIONS = [
    '.bin', '.npz'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_binary_file(filename):
    return any(filename.endswith(extension) for extension in PC_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def make_dataset_pc(dir):
    pointclouds = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_binary_file(fname):
                path = os.path.join(root, fname)
                pointclouds.append(path)

    return pointclouds


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class BevImageFolder(data.Dataset):

    def __init__(self, root, seed, transform=None, img_height=256, img_width=512, return_paths=False,
                 loader=default_pointcloud_loader):
        pointclouds = sorted(make_dataset_pc(root))
        if len(pointclouds) == 0:
            raise(RuntimeError("Found 0 pointclouds in: " + root + "\n"
                               "Supported pointcloud extensions are: " +
                               ",".join(PC_EXTENSIONS)))

        np.random.seed(seed)  # set same seed as in torch
        self.root = root
        self.pointclouds = pointclouds
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.bev_boundary = {
            "minX": 0,
            "maxX": 50,
            "minY": -25,
            "maxY": 25,
            "minZ": -2.73,
            "maxZ": 1.27
        }

    def __getitem__(self, index):
        path = self.pointclouds[index]
        pointcloud = self.loader(path)
        lidar_pc_filtered = removePoints(pointcloud, self.bev_boundary)  # filter point cloud points inside fov
        discretization = (self.bev_boundary["maxX"] - self.bev_boundary["minX"])/self.img_height
        lidar_bev = makeBVFeature(lidar_pc_filtered, self.bev_boundary, self.img_height, self.img_width, discretization)  # create Bird's Eye View





        ################################################################################################################################################
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        # ALWAYS CHANGE:
        # - LOADER IN UTILS:
        #   => FÜR LYFT:
        #      - LYFT_KITTI LOADER FÜR A
        #      - KITTI_LOADER FÜR B
        #   => FÜR AUDI:
        #      - AUDI_LOADER FÜR A
        #      - KITTI_FOV_LOADER FÜR B
        # - NUMBER AND ORDER OF CHANNELS (SIEHE UNTEN)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        ################################################################################################################################################




        ########################
        # set intensity to zero since lyft doesn't provide intensity values
        #lidar_bev[:, :, 2] = 0.0

        # remove intensity channel fully, since lyft doesn't provide intensity values
        #lidar_bev = lidar_bev[:, :, :2]

        # use only height channel
        lidar_bev = lidar_bev[:, :, 1]
        lidar_bev.shape += (1,)  # add dimension to shape (which is lost, due to only 1 channel), because UNIT code needs it
        ########################

        # map all float values from [0, 1] to integers [0, 255] (as in summer2winter)
        #lidar_bev_int = np.zeros(lidar_bev.shape, dtype=np.uint8)
        #lidar_bev_int[:, :, 0] = lidar_bev[:, :, 0] * 255
        #lidar_bev_int[:, :, 1] = lidar_bev[:, :, 1] * 255
        #lidar_bev_int[:, :, 2] = lidar_bev[:, :, 2] * 255

        #print("1 MIN: ", np.amin(lidar_bev[:, :, 0]))
        #print("1 MAX: ", np.amax(lidar_bev[:, :, 0]))
        #print("2 MIN: ", np.amin(lidar_bev[:, :, 1]))
        #print("2 MAX: ", np.amax(lidar_bev[:, :, 1]))
        #print("3 MIN: ", np.amin(lidar_bev[:, :, 2]))
        #print("3 MAX: ", np.amax(lidar_bev[:, :, 2]))

        if self.transform is not None:
            pointcloud = self.transform(lidar_bev)
            pointcloud = pointcloud.to(dtype=torch.float)
        if self.return_paths:
            return pointcloud, path
        else:
            return pointcloud

    def __len__(self):
        return len(self.pointclouds)
