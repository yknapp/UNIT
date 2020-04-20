#!/bin/bash

UNIT_FOV_DIR="/home/user/work/master_thesis/code/UNIT/outputs/unit_fov_lyft2kitti"
UNIT_CHECKPOINT="gen_00024000.pt"
#UNIT_CHECKPOINT="gen_00200000.pt"
python test_fov_converter.py --dataset="lyftkitti" --config $UNIT_FOV_DIR/config.yaml --checkpoint $UNIT_FOV_DIR/checkpoints/$UNIT_CHECKPOINT --input /home/user/work/master_thesis/datasets/lyft_kitti/object/training/fov --output_folder /home/user/work/master_thesis/datasets/lyft_kitti/object/training/fov_arrays --a2b 1
