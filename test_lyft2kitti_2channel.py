"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
from data import pointcloud_loader_kitti, removePoints, makeBVFeature
import imageio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input directory path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style', type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()



torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint))
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode  # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode  # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode  # decode function

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b == 1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

if 'img_height' in config and 'img_width' in config:
    img_height = config['img_height']
    img_width = config['img_width']
else:
    img_height = 480
    img_width = 480

with torch.no_grad():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ####################################################################################################################

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
        #pc_filename = '7f469f522e4862386992055bdcbf8beaf391428681d87a439f2fa1381e9e3ad0.bin'
        input_file_path = os.path.join(input_dir_path, pc_filename)
        pointcloud = pointcloud_loader_kitti(input_file_path)
        lidar_pc_filtered = removePoints(pointcloud, bev_boundary)  # filter point cloud points inside fov
        discretization = (bev_boundary["maxX"] - bev_boundary["minX"]) / img_height
        lidar_bev = makeBVFeature(lidar_pc_filtered, bev_boundary, img_height, img_width,
                                  discretization)  # create Bird's Eye View

        # remove intensity channel fully, since lyft doesn't provide intensity values
        lidar_bev_2channel = lidar_bev[:, :, :2]

        pointcloud = transform(lidar_bev_2channel).unsqueeze(0).cuda()
        pointcloud = pointcloud.to(dtype=torch.float)
        pointcloud = Variable(pointcloud)

        ####################################################################################################################


        # Start testing
        content, _ = encode(pointcloud)

        if opts.trainer == 'UNIT':
            outputs = decode(content)
            outputs = (outputs + 1) / 2.

            # export as .png image
            pc_filename_ = pc_filename.replace('.bin', '')
            path = os.path.join(opts.output_folder, pc_filename_ + '_output.png')
            n, c, h, w = outputs.data.shape
            padding = Variable(torch.zeros(n, 1, h, w))
            padding = padding.cuda()
            original = outputs.data
            padded_inp = torch.cat((original, padding), 1)
            vutils.save_image(padded_inp, path, padding=0, normalize=True)

            # export as numpy .npy file
            npimg = outputs.data.detach().cpu().numpy()[-1, :, :, :]  # to numpy array

            path = os.path.join(opts.output_folder, pc_filename_ + '.npy')
            np.save(path, npimg)
        else:
            pass

        if not opts.output_only:
            # also save input images
            output_filepath = os.path.join(opts.output_folder, pc_filename + '_input.png')
            imageio.imwrite(output_filepath, lidar_bev)
