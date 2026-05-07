from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader

from .ava_sequential_dataset import AvaSequentialDataset


@dataclass
class AvaSequentialDataFolderInfo:
    cfg: Any
    batch_size: int
    num_workers: int
    train_transform: Any
    val_transform: Any
    drop_last: bool = True
    pin_memory: bool = False
    persistent_workers: bool = False


def ava_sequential_collate_fn(batch):
    xs = torch.stack([sample.x for sample in batch], dim=0)
    ys = [sample.y for sample in batch]
    frame_indices = [sample.frame_indices for sample in batch]
    infos = [sample.info for sample in batch]
    return xs, ys, frame_indices, infos


def ava_sequential_data_folder(
    ava_info: AvaSequentialDataFolderInfo,
) -> Tuple[DataLoader, DataLoader, int]:
    cfg = ava_info.cfg
    frame_dir = Path(cfg.AVA.FRAME_DIR)
    frame_list_dir = Path(cfg.AVA.FRAME_LIST_DIR)
    annotation_dir = Path(cfg.AVA.ANNOTATION_DIR)

    train_frame_list_paths = [frame_list_dir / path for path in cfg.AVA.TRAIN_LISTS]
    val_frame_list_paths = [frame_list_dir / path for path in cfg.AVA.TEST_LISTS]
    train_box_list_paths = [
        annotation_dir / path for path in cfg.AVA.TRAIN_GT_BOX_LISTS
    ] + [
        annotation_dir / path for path in cfg.AVA.TRAIN_PREDICT_BOX_LISTS
    ]
    val_box_list_paths = [
        annotation_dir / path for path in cfg.AVA.TEST_PREDICT_BOX_LISTS
    ]
    label_map_path = annotation_dir / cfg.AVA.LABEL_MAP_FILE
    exclusion_path = annotation_dir / cfg.AVA.EXCLUSION_FILE
    groundtruth_path = annotation_dir / cfg.AVA.GROUNDTRUTH_FILE

    clip_duration = (
        ((cfg.DATA.NUM_FRAMES - 1) * cfg.DATA.SAMPLING_RATE) + 1
    ) / AvaSequentialDataset.AVA_FPS

    train_dataset = AvaSequentialDataset(
        video_path=frame_dir,
        clip_duration=clip_duration,
        video_edge_time=0.0,
        ext="*.jpg",
        is_train=True,
        transform=ava_info.train_transform,
        frame_list_paths=train_frame_list_paths,
        box_list_paths=train_box_list_paths,
        label_map_path=label_map_path,
        frames_per_clip=cfg.DATA.NUM_FRAMES,
        batch_size=ava_info.batch_size,
        sampling_rate=cfg.DATA.SAMPLING_RATE,
        detection_score_thresh=cfg.AVA.DETECTION_SCORE_THRESH,
    )
    val_dataset = AvaSequentialDataset(
        video_path=frame_dir,
        clip_duration=clip_duration,
        video_edge_time=0.0,
        ext="*.jpg",
        is_train=False,
        transform=ava_info.val_transform,
        frame_list_paths=val_frame_list_paths,
        box_list_paths=val_box_list_paths,
        label_map_path=label_map_path,
        frames_per_clip=cfg.DATA.NUM_FRAMES,
        batch_size=ava_info.batch_size,
        sampling_rate=cfg.DATA.SAMPLING_RATE,
        detection_score_thresh=cfg.AVA.DETECTION_SCORE_THRESH,
        exclusion_path=exclusion_path,
        groundtruth_path=groundtruth_path,
        shuffle=False,
    )

    n_classes = len(train_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=ava_info.batch_size,
        drop_last=ava_info.drop_last,
        num_workers=ava_info.num_workers,
        collate_fn=ava_sequential_collate_fn,
        pin_memory=ava_info.pin_memory,
        persistent_workers=(
            ava_info.persistent_workers
            if ava_info.num_workers > 0 else False
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=ava_info.batch_size,
        drop_last=False,
        num_workers=ava_info.num_workers,
        collate_fn=ava_sequential_collate_fn,
        pin_memory=ava_info.pin_memory,
        persistent_workers=(
            ava_info.persistent_workers
            if ava_info.num_workers > 0 else False
        ),
    )

    return train_loader, val_loader, n_classes
