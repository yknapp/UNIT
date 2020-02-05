"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import argparse
import sys
import os
from PIL import Image
from data import pointcloud_loader_kitti, removePoints, makeBVFeature
import imageio
import numpy as np
from lyft2kitti_converter import Lyft2KittiConverter

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input directory path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
opts = parser.parse_args()

lyft2kitti_conv = Lyft2KittiConverter(opts.config, opts.checkpoint, a2b=opts.a2b, seed=opts.seed)

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

img_height = 480
img_width = 480

bev_boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

input_dir_path = opts.input
pc_filename_list = os.listdir(input_dir_path)
for pc_filename in pc_filename_list:
    print("processing ", pc_filename)
    input_file_path = os.path.join(input_dir_path, pc_filename)
    pointcloud = pointcloud_loader_kitti(input_file_path)
    lidar_pc_filtered = removePoints(pointcloud, bev_boundary)  # filter point cloud points inside fov
    discretization = (bev_boundary["maxX"] - bev_boundary["minX"]) / img_height
    lidar_bev = makeBVFeature(lidar_pc_filtered, bev_boundary, img_height, img_width,
                              discretization)  # create Bird's Eye View
    lidar_bev[:, :, 2] = 0.0
    imageio.imwrite('%s_before.png' % pc_filename.replace('.bin', ''), lidar_bev)

    # remove intensity channel fully, since lyft doesn't provide intensity values
    lidar_bev_2channel = lidar_bev[:, :, :2]

    lidar_bev_2channel_transformed_raw = lyft2kitti_conv.transform(lidar_bev_2channel)
    lidar_rgb = np.zeros((img_height, img_width, 3))
    lidar_rgb[:, :, 0] = lidar_bev_2channel_transformed_raw[0, :, :]
    lidar_rgb[:, :, 1] = lidar_bev_2channel_transformed_raw[1, :, :]
    imageio.imwrite('%s_after.png' % pc_filename.replace('.bin', ''), lidar_rgb)

    sys.exit()
