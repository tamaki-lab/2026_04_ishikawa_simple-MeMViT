from typing import Literal
from dataclasses import dataclass
import argparse

from torch.utils.data import DataLoader

from dataset import (
    image_folder,
    ImageFolderInfo,
    video_folder,
    VideoFolderInfo,
    sequential_video_folder,
    epic_kitchens_sequential_data_folder,
    EpicKitchensSequentialDataFolderInfo,
    salads50_sequential_data_folder,
    Salads50SequentialDataFolderInfo,
    transform_image,
    TransformImageInfo,
    transform_video,
    TransformVideoInfo,
    build_sequential_video_transform,
)
from model.memvit.config.defaults import get_cfg


@dataclass
class DataloadersInfo:
    """DataloadersInfo

        train_loader (torch.utils.data.DataLoader): training set loader
        val_loader (torch.utils.data.DataLoader): validation set loader
        n_classes (int | tuple[int, ...]): number of classes
    """
    train_loader: DataLoader
    val_loader: DataLoader
    n_classes: int | tuple[int, ...]


SupportedDatasets = Literal["ImageFolder", "VideoFolder",
                            "SequentialVideoFolder", "EpicKitchenSequentialDataset",
                            "Salads50SequentialDataset"]


def configure_dataloader(
    command_line_args: argparse.Namespace,
    dataset_name: SupportedDatasets,
    cfg=None,
):
    """dataloader factory

    Args:
        command_line_args (argparse.Namespace): command line args
        dataset_name (SupportedDatasets): dataset name (str).
            ["ImageFolder", "VideoFolder", "SequentialVideoFolder", "EpicKitchenSequentialDataset"]

    Raises:
        ValueError: invalid dataset_name is given

    Returns:
        (DataloadersInfo): dataset information
    """

    args = command_line_args

    if dataset_name == "ImageFolder":
        train_transform, val_transform = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            image_folder(ImageFolderInfo(
                root=args.root,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataset_name == "VideoFolder":
        train_transform, val_transform = \
            transform_video(TransformVideoInfo(
                frames_per_clip=args.frames_per_clip
            ))
        train_loader, val_loader, n_classes = \
            video_folder(VideoFolderInfo(
                root=args.root,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform,
                clip_duration=args.clip_duration,
                clips_per_video=args.clips_per_video
            ))

    elif dataset_name == "SequentialVideoFolder":
        train_loader, val_loader, n_classes = sequential_video_folder(
            clip_duration=args.clip_duration,
            video_edge_time=args.video_edge_time,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            ext=args.ext,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            frames_per_clip=args.frames_per_clip,
            label_mode=args.sequential_label_mode,
            train_annotation_path=args.train_annotation_path,
            val_annotation_path=args.val_annotation_path,
            background_label=args.background_label,
            epic_label_type=args.epic_label_type,
        )

    elif dataset_name == "EpicKitchenSequentialDataset":
        train_transform, val_transform = \
            build_sequential_video_transform(TransformVideoInfo(
                frames_per_clip=args.frames_per_clip
            ))
        train_loader, val_loader, n_classes = \
            epic_kitchens_sequential_data_folder(
                EpicKitchensSequentialDataFolderInfo(
                    root=args.root,
                    train_dir=args.train_dir,
                    val_dir=args.val_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    train_transform=train_transform,
                    val_transform=val_transform,
                    clip_duration=args.clip_duration,
                    video_edge_time=args.video_edge_time,
                    ext=args.ext,
                    frames_per_clip=args.frames_per_clip,
                    task=args.epic_task,
                    train_annotation_path=args.train_annotation_path,
                    val_annotation_path=args.val_annotation_path,
                    label_type=args.epic_label_type,
                    background_label=args.background_label,
                    anticipation_time=args.epic_anticipation_time,
                ))

    elif dataset_name == "Salads50SequentialDataset":
        train_transform, val_transform = \
            build_sequential_video_transform(TransformVideoInfo(
                frames_per_clip=args.frames_per_clip
            ))
        train_loader, val_loader, n_classes = \
            salads50_sequential_data_folder(
                Salads50SequentialDataFolderInfo(
                    root=args.root,
                    train_dir=args.train_dir,
                    val_dir=args.val_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    train_transform=train_transform,
                    val_transform=val_transform,
                    clip_duration=args.clip_duration,
                    video_edge_time=args.video_edge_time,
                    ext=args.ext,
                    annotation_root=args.annotation_root,
                    split_root=args.split_root,
                    split_id=args.split_id,
                    label_granularity=args.label_granularity,
                    label_map_path=args.label_map_path,
                    frames_per_clip=args.frames_per_clip,
                ))

    else:
        raise ValueError("invalid dataset_name")

    return DataloadersInfo(
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=n_classes
    )
