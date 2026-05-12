import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transforms

from .salads50_sequential_dataset import FiftySaladsSequentialDataset


@dataclass
class FiftySaladsSequentialDataFolderInfo():
    root: str
    train_dir: str
    val_dir: str
    batch_size: int
    num_workers: int
    train_transform: transforms
    val_transform: transforms
    clip_duration: float
    video_edge_time: float
    ext: str
    annotation_root: str
    split_root: str
    split_id: int = 1
    label_granularity: str = "fine"
    label_map_path: str | None = None
    frames_per_clip: int = 16
    drop_last: bool = True
    pin_memory: bool = False
    persistent_workers: bool = False


def fiftysalads_sequential_collate_fn(batch):
    xs = torch.stack([sample.x for sample in batch], dim=0)
    ys = torch.stack([sample.y for sample in batch], dim=0)
    frame_indices = [sample.frame_indices for sample in batch]
    infos = [sample.info for sample in batch]
    return xs, ys, frame_indices, infos


def salads50_sequential_data_folder(
    fiftysalads_info: FiftySaladsSequentialDataFolderInfo,
) -> Tuple[DataLoader, DataLoader, int]:
    root_train_dir = os.path.join(
        fiftysalads_info.root,
        fiftysalads_info.train_dir,
    )
    root_val_dir = os.path.join(
        fiftysalads_info.root,
        fiftysalads_info.val_dir,
    )
    assert os.path.exists(root_train_dir) and os.path.isdir(root_train_dir)
    assert os.path.exists(root_val_dir) and os.path.isdir(root_val_dir)

    train_dataset = FiftySaladsSequentialDataset(
        video_path=root_train_dir,
        clip_duration=fiftysalads_info.clip_duration,
        video_edge_time=fiftysalads_info.video_edge_time,
        ext=fiftysalads_info.ext,
        is_train=True,
        transform=fiftysalads_info.train_transform,
        annotation_root=fiftysalads_info.annotation_root,
        split_root=fiftysalads_info.split_root,
        split_id=fiftysalads_info.split_id,
        label_granularity=fiftysalads_info.label_granularity,
        label_map_path=fiftysalads_info.label_map_path,
        frames_per_clip=fiftysalads_info.frames_per_clip,
        batch_size=fiftysalads_info.batch_size,
        split_name="train",
    )
    val_dataset = FiftySaladsSequentialDataset(
        video_path=root_val_dir,
        clip_duration=fiftysalads_info.clip_duration,
        video_edge_time=fiftysalads_info.video_edge_time,
        ext=fiftysalads_info.ext,
        is_train=False,
        transform=fiftysalads_info.val_transform,
        annotation_root=fiftysalads_info.annotation_root,
        split_root=fiftysalads_info.split_root,
        split_id=fiftysalads_info.split_id,
        label_granularity=fiftysalads_info.label_granularity,
        label_map_path=fiftysalads_info.label_map_path,
        frames_per_clip=fiftysalads_info.frames_per_clip,
        batch_size=fiftysalads_info.batch_size,
        shuffle=False,
        split_name="val",
    )

    shared_labels = sorted({
        *train_dataset.class_to_idx.keys(),
        *val_dataset.class_to_idx.keys(),
    })
    shared_class_to_idx = {
        label_name: idx for idx, label_name in enumerate(shared_labels)
    }
    train_dataset.class_to_idx = shared_class_to_idx
    val_dataset.class_to_idx = shared_class_to_idx
    train_dataset.num_classes = len(shared_class_to_idx)
    val_dataset.num_classes = len(shared_class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=fiftysalads_info.batch_size,
        drop_last=fiftysalads_info.drop_last,
        num_workers=fiftysalads_info.num_workers,
        collate_fn=fiftysalads_sequential_collate_fn,
        pin_memory=fiftysalads_info.pin_memory,
        persistent_workers=(
            fiftysalads_info.persistent_workers
            if fiftysalads_info.num_workers > 0 else False
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=fiftysalads_info.batch_size,
        drop_last=False,
        num_workers=fiftysalads_info.num_workers,
        collate_fn=fiftysalads_sequential_collate_fn,
        pin_memory=fiftysalads_info.pin_memory,
        persistent_workers=(
            fiftysalads_info.persistent_workers
            if fiftysalads_info.num_workers > 0 else False
        ),
    )

    return train_loader, val_loader, len(shared_class_to_idx)
