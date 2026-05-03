#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1

args=(
    -d EpicKitchenSequentialDataset
    -td /mnt/NAS-TVS872XT/dataset/EPIC_KITCHENS/videos
    -vd /mnt/NAS-TVS872XT/dataset/EPIC_KITCHENS/videos
    --sequential_label_mode frame
    --train_annotation_path /mnt/NAS-TVS872XT/dataset/EPIC_KITCHENS/epic-kitchens-100-annotations/EPIC_100_train.csv
    --val_annotation_path /mnt/NAS-TVS872XT/dataset/EPIC_KITCHENS/epic-kitchens-100-annotations/EPIC_100_validation.csv
    --epic_label_type verb
    --epic_task action_recognition
    -m memvit
    --frames_per_clip 16
    -b 1
    -w 4
    -e 10
    --optimizer_name Adam
    --log_interval_steps 10
    --disable_comet
    --devices -1
    --cfg_file configs/MeMViT_16_K400.yaml
)

python main_pl.py "${args[@]}" "$@"
