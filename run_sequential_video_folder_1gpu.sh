#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main_pl.py \
    -d SequentialVideoFolder \
    -td /mnt/NAS-TVS872XT/dataset/Kinetics400/train/ \
    -vd /mnt/NAS-TVS872XT/dataset/Kinetics400/val/ \
    -m memvit \
    -w 8 \
    -b 6 \
    -e 20 \
    --optimizer_name Adam \
    --log_interval_steps 10 \
    --devices 1 \
    --cfg_file configs/MeMViT_16_K400.yaml \
    --loop_mode train \
    # --disable_comet \
    --val_interval_steps 30 \
