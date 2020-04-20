"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function

import argparse
import sys
import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from data import pointcloud_loader_kitti, pointcloud_loader_lyftkitti, pointcloud_loader_audi, get_mask, lidarimg2grid
from fov_converter import FovConverter
from dataset_calibration import KittiCalibration
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='kitti', help="dataset, which should be transformed to KITTI")
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input directory path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
opts = parser.parse_args()

fov_conv = FovConverter(opts.config, opts.checkpoint, a2b=opts.a2b, seed=opts.seed)

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

input_dir_path = opts.input
output_dir_path = opts.output_folder
pc_filename_list = os.listdir(input_dir_path)
for pc_filename in pc_filename_list:
    print("processing ", pc_filename)
    # load processed fov image file
    input_file_path = os.path.join(input_dir_path, pc_filename)
    fov_image = np.load(input_file_path)

    # convert torch.Tensor to PIL image
    fov_image_pil = Image.fromarray(fov_image)

    # remove file extension
    pc_filename = pc_filename.replace('.bin', '')
    # save as image
    #output_filename = os.path.join(output_dir_path, '%s.png' % pc_filename)

    # save untransformed image
    #if opts.dataset in ('kitti', 'lyft'):
    #    fov_image_pil_resized = fov_image_pil.resize((844, 256), Image.BILINEAR)  # lyft2kitti: same method as torch resizing
    #elif opts.dataset == 'audi':
    #    fov_image_pil_resized = fov_image_pil.resize((548, 256), Image.BILINEAR)  # audi2kitti: same method as torch resizing
    #else:
    #    print("Error: Unknown datasets '%s'" % opts.dataset)
    #fov_image_np = np.asarray(fov_image_pil_resized)
    #imageio.imwrite(output_filename, fov_image_np)
    #plt.imsave(output_filename, fov_image_np)
    #plt.close()

    # save as numpy array
    #np.save(os.path.join(output_dir_path, '%s_original' % pc_filename), fov_image)

    # UNIT transformation
    fov_image_transformed = fov_conv.transform(fov_image_pil)

    #print("MIN 1: ", np.amin(fov_image_pil_resized))
    #print("MAX 1: ", np.amax(fov_image_pil_resized))
    #print("MIN 2: ", np.amin(fov_image_transformed))
    #print("MAX 2: ", np.amax(fov_image_transformed))

    # save as numpy array
    #np.save(os.path.join(output_dir_path, '%s_transformed' % pc_filename), fov_image_transformed)
    # save as image
    #output_filename = os.path.join(output_dir_path, '%s_transformed.png' % pc_filename)
    output_filename = os.path.join(output_dir_path, '%s.png' % pc_filename)
    imageio.imwrite(output_filename, fov_image_transformed)
    #plt.imsave(output_filename, fov_image_transformed)
    #plt.close()
