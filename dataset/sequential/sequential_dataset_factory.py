from __future__ import annotations

from typing import Optional

import torch

from .epic_kitchens_sequential_dataset import EpicKitchenSequentialDataset

from .sequential_transforms import build_sequential_video_transform
from .sequential_collate import (
    sequential_collate_fn,
    classification_collate_fn,
)


DATASET_REGISTRY = {
    "epic_kitchens": EpicKitchenSequentialDataset,
}


CLASSIFICATION_LIKE_DATASETS = {
    "video_folder",
}


CLASSIFICATION_LIKE_EPIC_TASKS = {
    "action_recognition",
    "action_anticipation",
}


def build_sequential_dataset(
    dataset_name: str,
    video_path: str,
    clip_duration: float,
    video_edge_time: float,
    ext: str,
    is_train: bool,
    frames_per_clip: int = 16,
    batch_size: int = 1,
    task: Optional[str] = None,
    annotation_path: Optional[str] = None,
    label_type: str = "verb",
    background_label: str = "background",
    anticipation_time: float = 1.0,
    recognition_label_strategy: str = "center",
    transform=None,
    shuffle: Optional[bool] = None,
):
    """
    dataset_name に応じて Sequential Dataset を作成する．

    Parameters
    ----------
    dataset_name：現状"epic_kitchens"のみ対応
    video_path：動画ファイルが置かれているディレクトリ
    clip_duration：1 clip の秒数
    video_edge_time：動画末尾を避けるための秒数
    ext：動画拡張子
    is_train：train dataset or validation dataset
    frames_per_clip：transform 後のフレーム数
    batch_size：DataLoader の batch_size
    task：データセットごとのタスク名
        epic_kitchens：
            "action_recognition"
            "action_anticipation"

    annotation_path：EPIC-KITCHENS や AVA の annotation path
    label_type：
        EPIC-KITCHENS 用
        "verb", "noun", "action"
    background_label：
        EPIC-KITCHENS 用
        対応する行動がない場合のラベル
    anticipation_time：
        EPIC-KITCHENS の action_anticipation 用
    recognition_label_strategy：
        EPIC-KITCHENS の action_recognition 用
        "center" or "majority"
    transform：None の場合は build_sequential_video_transform で作成
    shuffle：None の場合，train なら True，val なら False

    Returns
    -------
    dataset:
        BaseSequentialVideoDataset を継承した Dataset．
    """

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset_name: {dataset_name}. "
            f"Available datasets: {sorted(DATASET_REGISTRY.keys())}"
        )

    if transform is None:
        transform = build_sequential_video_transform(
            is_train=is_train,
            frames_per_clip=frames_per_clip,
        )

    if shuffle is None:
        shuffle = is_train

    dataset_cls = DATASET_REGISTRY[dataset_name]

    common_kwargs = dict(
        video_path=video_path,
        clip_duration=clip_duration,
        video_edge_time=video_edge_time,
        ext=ext,
        is_train=is_train,
        transform=transform,
        frames_per_clip=frames_per_clip,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    if dataset_name == "epic_kitchens":
        if annotation_path is None:
            raise ValueError(
                "annotation_path is required when dataset_name='epic_kitchens'"
            )

        return dataset_cls(
            **common_kwargs,
            annotation_path=annotation_path,
            task=task or "action_recognition",
            label_type=label_type,
            background_label=background_label,
            anticipation_time=anticipation_time,
            recognition_label_strategy=recognition_label_strategy,
        )

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def choose_collate_fn(
    dataset_name: str,
    task: Optional[str] = None,
):
    """
    dataset_name と task に応じて collate_fn を選択する

    基本方針:
        y が単純な class index の場合:
            classification_collate_fn

        y が dict/list など複雑な場合:
            sequential_collate_fn
    """

    if dataset_name in CLASSIFICATION_LIKE_DATASETS:
        return classification_collate_fn

    if dataset_name == "epic_kitchens":
        if task in CLASSIFICATION_LIKE_EPIC_TASKS or task is None:
            return classification_collate_fn

    return sequential_collate_fn


def build_sequential_dataloader(
    dataset_name: str,
    train_dir: str,
    val_dir: str,
    clip_duration: float,
    video_edge_time: float,
    batch_size: int,
    num_workers: int,
    ext: str,
    frames_per_clip: int = 16,
    task: Optional[str] = None,
    train_annotation_path: Optional[str] = None,
    val_annotation_path: Optional[str] = None,
    label_type: str = "verb",
    background_label: str = "background",
    anticipation_time: float = 1.0,
    recognition_label_strategy: str = "center",
    drop_last: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = False,
):
    """
    train_loader, val_loader, n_classes をまとめて作成する

    Returns
    -------
    train_loader：torch.utils.data.DataLoader
    val_loader：torch.utils.data.DataLoader
    n_classes：train_dataset.num_classes
    """

    train_dataset = build_sequential_dataset(
        dataset_name=dataset_name,
        video_path=train_dir,
        clip_duration=clip_duration,
        video_edge_time=video_edge_time,
        ext=ext,
        is_train=True,
        frames_per_clip=frames_per_clip,
        batch_size=batch_size,
        task=task,
        annotation_path=train_annotation_path,
        label_type=label_type,
        background_label=background_label,
        anticipation_time=anticipation_time,
        recognition_label_strategy=recognition_label_strategy,
    )

    val_dataset = build_sequential_dataset(
        dataset_name=dataset_name,
        video_path=val_dir,
        clip_duration=clip_duration,
        video_edge_time=video_edge_time,
        ext=ext,
        is_train=False,
        frames_per_clip=frames_per_clip,
        batch_size=batch_size,
        task=task,
        annotation_path=val_annotation_path,
        label_type=label_type,
        background_label=background_label,
        anticipation_time=anticipation_time,
        recognition_label_strategy=recognition_label_strategy,
    )

    collate_fn = choose_collate_fn(
        dataset_name=dataset_name,
        task=task,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    n_classes = train_dataset.num_classes

    return train_loader, val_loader, n_classes
