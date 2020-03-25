"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.utils.data as data
import os.path
import numpy as np
import cv2

from PIL import Image
from dataset_calibration import KittiCalibration


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


def get_mask(rect_pts, points_2d, imgsize):
    mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < imgsize[0]) & \
           (points_2d[:, 1] >= 0) & (points_2d[:, 1] < imgsize[1])
    mask = mask & (rect_pts[:, 2] >= 2)

    # pts_on_image_with_depth = np.append(points_2d[mask, 0:2], rect_pts[mask, 2], axis=1)

    pts_on_image_with_depth = np.zeros([mask.sum(), 3])
    pts_on_image_with_depth[:, 0:2] = points_2d[mask, 0:2]
    pts_on_image_with_depth[:, 2] = rect_pts[mask, 2]

    # return points_2d[mask, 0:2], rect_pts[mask, ]

    return pts_on_image_with_depth, rect_pts[mask,]


def lidarimg2grid(pts_image, img_shape):
    size_0 = img_shape[0]
    size_1 = img_shape[1]
    grid = np.zeros(img_shape[:2])
    for p in pts_image:
        i = int(p[0]) - 1
        j = int(p[1]) - 1

        value = p[2]   # representation of depth, i.e. p[2], 1/p[2], log(p[2])

        grid[i, j] = value

        if i + 1 < size_0 and j + 1 < size_1:
            grid[i+1, j+1] = value
        if i < size_0 and j + 1 < size_1:
            grid[i, j+1] = value
        if i + 1 < size_0 and j < size_1:
            grid[i+1, j] = value
        if i - 1 >= 0 and j - 1 >= 0:
            grid[i-1, j-1] = value
        if i - 1 >= 0 and j < size_1:
            grid[i-1, j] = value
        if i < size_0 and j - 1 >= 0:
            grid[i, j-1] = value
        if i - 1 >= 0 and j + 1 < size_1:
            grid[i-1, j+1] = value
        if i + 1 < size_0 and j - 1 >= 0:
            grid[i+1, j-1] = value

        if i + 2 < size_0 and j < size_1:
            grid[i+2, j] = value
        if i < size_0 and j + 2 < size_1:
            grid[i, j+2] = value
        if i - 2 >= 0 and j + 2 < size_1:
            grid[i-2, j] = value
        if i < size_0 and j - 2 >= 0:
            grid[i, j-2] = value
        if i + 2 < size_0 and j + 1 < size_1:
            grid[i+2, j+1] = value
        if i + 2 < size_0 and j + 1 >= 0:
            grid[i+2, j-1] = value
        if i - 2 >= 0 and j + 1 < size_1:
            grid[i-2, j+1] = value
        if i - 2 >= 0 and j - 1 >= 0:
            grid[i-2, j-1] = value
        if i + 1 < size_0 and j + 2 < size_1:
            grid[i+1, j+2] = value
        if i + 1 < size_0 and j - 2 >= 0:
            grid[i+1, j-2] = value
        if i - 1 >= 0 and j + 2 < size_1:
            grid[i-1, j+2] = value
        if i - 1 >= 0 and j - 2 >= 0:
            grid[i-1, j-2] = value
    return grid.T


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


class FovImageFolder(data.Dataset):

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

    def __getitem__(self, index):
        path = self.pointclouds[index]
        lidar = self.loader(path)
        calib = KittiCalibration()
        fov_img_shape = (1242, 375)  # image resolution of KITTI rgb camera, on which lidars get mapped
        rect_pts = calib.project_velo_to_rect(lidar[:, 0:3])
        points_2d = calib.project_rect_to_image(rect_pts)
        pts_image, pts_xyz_mask = get_mask(rect_pts, points_2d, imgsize=fov_img_shape)
        fov_image = lidarimg2grid(pts_image, fov_img_shape)
        #print("min: ", np.amin(fov_image))
        #print("max: ", np.amax(fov_image))
        fov_image = Image.fromarray(fov_image)
        #fov_image = fov_image.resize((416, 128))
        #print(fov_image.size)

        #fov_image.shape += (1,)  # add dimension to shape (which is lost, due to only 1 channel), because UNIT code needs it

        if self.transform is not None:
            pointcloud = self.transform(fov_image)
            pointcloud = pointcloud.to(dtype=torch.float)
        if self.return_paths:
            return pointcloud, path
        else:
            return pointcloud

    def __len__(self):
        return len(self.pointclouds)
