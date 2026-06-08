#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python3 main_pl.py \
    -d Salads50SequentialDataset \
    -r /mnt/HDD10TB-2/ishikawa/2026_04_ishikawa_simple-MeMViT/data/50Salads \
    -td rgb/train1 \
    -vd rgb/val1 \
    --train_annotation_root framelabels_custom/train1 \
    --val_annotation_root framelabels_custom/val1 \
    --train_split_root framelabels_custom/train1 \
    --val_split_root framelabels_custom/val1 \
    --label_granularity fine \
    -m memvit \
    --frames_per_clip 16 \
    -b 2 \
    --grad_accum 1 \
    -w 4 \
    -e 10 \
    -vi 1 \
    --optimizer_name OrthogonalAdamW \
    -lr 0.0001 \
    --use_scheduler \
    --log_interval_steps 1 \
    --devices 0
