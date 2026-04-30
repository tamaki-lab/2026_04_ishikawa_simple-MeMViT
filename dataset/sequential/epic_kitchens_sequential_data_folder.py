import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transforms

from .epic_kitchens_sequential_dataset import EpicKitchenSequentialDataset


@dataclass
class EpicKitchensSequentialDataFolderInfo():
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
    frames_per_clip: int = 16
    task: str = "action_recognition"
    train_annotation_path: str | None = None
    val_annotation_path: str | None = None
    label_type: str = "verb"
    background_label: str = "background"
    anticipation_time: float = 1.0
    recognition_label_strategy: str = "center"
    drop_last: bool = True
    pin_memory: bool = False
    persistent_workers: bool = False


def epic_kitchens_sequential_collate_fn(batch):
    xs = torch.stack([sample.x for sample in batch], dim=0)
    ys = torch.tensor([sample.y for sample in batch], dtype=torch.long)
    frame_indices = [sample.frame_indices for sample in batch]
    infos = [sample.info for sample in batch]
    return xs, ys, frame_indices, infos


def epic_kitchens_sequential_data_folder(
        epic_kitchens_info: EpicKitchensSequentialDataFolderInfo
) -> Tuple[DataLoader, DataLoader, int]:
    root_train_dir = os.path.join(
        epic_kitchens_info.root,
        epic_kitchens_info.train_dir
    )
    root_val_dir = os.path.join(
        epic_kitchens_info.root,
        epic_kitchens_info.val_dir
    )
    assert os.path.exists(root_train_dir) and os.path.isdir(root_train_dir)
    assert os.path.exists(root_val_dir) and os.path.isdir(root_val_dir)

    train_dataset = EpicKitchenSequentialDataset(
        video_path=root_train_dir,
        clip_duration=epic_kitchens_info.clip_duration,
        video_edge_time=epic_kitchens_info.video_edge_time,
        ext=epic_kitchens_info.ext,
        is_train=True,
        transform=epic_kitchens_info.train_transform,
        annotation_path=epic_kitchens_info.train_annotation_path,
        frames_per_clip=epic_kitchens_info.frames_per_clip,
        batch_size=epic_kitchens_info.batch_size,
        task=epic_kitchens_info.task,
        label_type=epic_kitchens_info.label_type,
        background_label=epic_kitchens_info.background_label,
        anticipation_time=epic_kitchens_info.anticipation_time,
        recognition_label_strategy=epic_kitchens_info.recognition_label_strategy,
    )
    val_dataset = EpicKitchenSequentialDataset(
        video_path=root_val_dir,
        clip_duration=epic_kitchens_info.clip_duration,
        video_edge_time=epic_kitchens_info.video_edge_time,
        ext=epic_kitchens_info.ext,
        is_train=False,
        transform=epic_kitchens_info.val_transform,
        annotation_path=epic_kitchens_info.val_annotation_path,
        frames_per_clip=epic_kitchens_info.frames_per_clip,
        batch_size=epic_kitchens_info.batch_size,
        task=epic_kitchens_info.task,
        label_type=epic_kitchens_info.label_type,
        background_label=epic_kitchens_info.background_label,
        anticipation_time=epic_kitchens_info.anticipation_time,
        recognition_label_strategy=epic_kitchens_info.recognition_label_strategy,
        shuffle=False,
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

    n_classes = len(shared_class_to_idx)
    train_dataset.num_classes = n_classes
    val_dataset.num_classes = n_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=epic_kitchens_info.batch_size,
        drop_last=epic_kitchens_info.drop_last,
        num_workers=epic_kitchens_info.num_workers,
        collate_fn=epic_kitchens_sequential_collate_fn,
        pin_memory=epic_kitchens_info.pin_memory,
        persistent_workers=(
            epic_kitchens_info.persistent_workers
            if epic_kitchens_info.num_workers > 0 else False
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=epic_kitchens_info.batch_size,
        drop_last=epic_kitchens_info.drop_last,
        num_workers=epic_kitchens_info.num_workers,
        collate_fn=epic_kitchens_sequential_collate_fn,
        pin_memory=epic_kitchens_info.pin_memory,
        persistent_workers=(
            epic_kitchens_info.persistent_workers
            if epic_kitchens_info.num_workers > 0 else False
        ),
    )

    return train_loader, val_loader, n_classes
