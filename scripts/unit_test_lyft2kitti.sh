#!/bin/bash
python test.py --config configs/unit_bev_lyft2kitti_folder.yaml --trainer UNIT --checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_folder_copy/checkpoints/gen_01000000.pt --input /home/user/datasets/bev_images/lyft/host-a004_lidar1_1232815260900990726.png --output_folder /home/user/work/master_thesis/code/UNIT/results/lyft2kitti --a2b 1
