from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as transforms

from .salads50_sequential_dataset import Salads50SequentialDataset


@dataclass
class Salads50SequentialDataFolderInfo():
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
    train_annotation_root: str | None = None
    val_annotation_root: str | None = None
    train_split_root: str | None = None
    val_split_root: str | None = None
    split_id: int = 1
    label_granularity: str = "fine"
    label_map_path: str | None = None
    background_label: str = "background"
    max_train_clips_per_video: int | None = None
    frames_per_clip: int = 16
    drop_last: bool = True
    pin_memory: bool = False
    persistent_workers: bool = False


def _resolve_existing_path(
    root: str,
    path_value: str | None,
    *,
    expect_dir: bool,
    fallback_to_root: bool = False,
) -> str | None:
    if path_value is None:
        return None

    root_path = Path(root)
    raw_path = Path(path_value)

    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(root_path / raw_path)
        candidates.append(raw_path)
        if fallback_to_root:
            candidates.append(root_path)

    seen: set[Path] = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if not candidate.exists():
            continue
        if expect_dir and not candidate.is_dir():
            continue
        return str(candidate)

    expected_kind = "directory" if expect_dir else "path"
    raise FileNotFoundError(
        f"Could not resolve {expected_kind} for {path_value!r} with root={root!r}. "
        f"Tried: {[str(candidate) for candidate in unique_candidates]}"
    )


def salads50_sequential_collate_fn(batch):
    xs = torch.stack([sample.x for sample in batch], dim=0)
    ys = torch.stack([sample.y for sample in batch], dim=0)
    frame_indices = [sample.frame_indices for sample in batch]
    infos = [sample.info for sample in batch]
    return xs, ys, frame_indices, infos


def salads50_sequential_data_folder(
    salads50_info: Salads50SequentialDataFolderInfo,
) -> Tuple[DataLoader, DataLoader, int]:
    root_train_dir = _resolve_existing_path(
        salads50_info.root,
        salads50_info.train_dir,
        expect_dir=True,
        fallback_to_root=True,
    )
    root_val_dir = _resolve_existing_path(
        salads50_info.root,
        salads50_info.val_dir,
        expect_dir=True,
        fallback_to_root=True,
    )
    train_annotation_root = _resolve_existing_path(
        salads50_info.root,
        salads50_info.train_annotation_root or salads50_info.annotation_root,
        expect_dir=True,
    )
    val_annotation_root = _resolve_existing_path(
        salads50_info.root,
        salads50_info.val_annotation_root or salads50_info.annotation_root,
        expect_dir=True,
    )
    train_split_root = _resolve_existing_path(
        salads50_info.root,
        salads50_info.train_split_root or salads50_info.split_root,
        expect_dir=True,
    )
    val_split_root = _resolve_existing_path(
        salads50_info.root,
        salads50_info.val_split_root or salads50_info.split_root,
        expect_dir=True,
    )
    label_map_path = _resolve_existing_path(
        salads50_info.root,
        salads50_info.label_map_path,
        expect_dir=False,
    )

    train_dataset = Salads50SequentialDataset(
        video_path=root_train_dir,
        clip_duration=salads50_info.clip_duration,
        video_edge_time=salads50_info.video_edge_time,
        ext=salads50_info.ext,
        is_train=True,
        transform=salads50_info.train_transform,
        annotation_root=train_annotation_root,
        split_root=train_split_root,
        split_id=salads50_info.split_id,
        label_granularity=salads50_info.label_granularity,
        label_map_path=label_map_path,
        background_label=salads50_info.background_label,
        max_train_clips_per_video=salads50_info.max_train_clips_per_video,
        frames_per_clip=salads50_info.frames_per_clip,
        batch_size=salads50_info.batch_size,
        split_name="train",
    )
    val_dataset = Salads50SequentialDataset(
        video_path=root_val_dir,
        clip_duration=salads50_info.clip_duration,
        video_edge_time=salads50_info.video_edge_time,
        ext=salads50_info.ext,
        is_train=False,
        transform=salads50_info.val_transform,
        annotation_root=val_annotation_root,
        split_root=val_split_root,
        split_id=salads50_info.split_id,
        label_granularity=salads50_info.label_granularity,
        label_map_path=label_map_path,
        background_label=salads50_info.background_label,
        frames_per_clip=salads50_info.frames_per_clip,
        batch_size=salads50_info.batch_size,
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
        batch_size=salads50_info.batch_size,
        drop_last=salads50_info.drop_last,
        num_workers=salads50_info.num_workers,
        collate_fn=salads50_sequential_collate_fn,
        pin_memory=salads50_info.pin_memory,
        persistent_workers=(
            salads50_info.persistent_workers
            if salads50_info.num_workers > 0 else False
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=salads50_info.batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=salads50_sequential_collate_fn,
        pin_memory=salads50_info.pin_memory,
        persistent_workers=False,
    )

    return train_loader, val_loader, len(shared_class_to_idx)
