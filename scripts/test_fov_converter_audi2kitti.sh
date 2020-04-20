#!/bin/bash

UNIT_FOV_DIR="/home/user/work/master_thesis/code/UNIT/outputs/unit_fov_audi2kitti"
UNIT_CHECKPOINT="gen_00078000.pt"
#UNIT_CHECKPOINT="gen_00200000.pt"
python test_fov_converter.py --dataset="audi" --config $UNIT_FOV_DIR/config.yaml --checkpoint $UNIT_FOV_DIR/checkpoints/$UNIT_CHECKPOINT --input /home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/fov --output_folder /home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/fov_arrays --a2b 1
