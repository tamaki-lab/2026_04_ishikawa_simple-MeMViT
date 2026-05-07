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
    ava_sequential_data_folder,
    AvaSequentialDataFolderInfo,
    transform_image,
    TransformImageInfo,
    transform_video,
    TransformVideoInfo,
    build_epic_kitchens_sequential_transform,
    build_ava_sequential_transform,
)
from model.memvit.config.defaults import get_cfg


@dataclass
class DataloadersInfo:
    """DataloadersInfo

        train_loader (torch.utils.data.DataLoader): training set loader
        val_loader (torch.utils.data.DataLoader): validation set loader
        n_classes (int): number of classes
    """
    train_loader: DataLoader
    val_loader: DataLoader
    n_classes: int


SupportedDatasets = Literal["ImageFolder", "VideoFolder",
                            "SequentialVideoFolder", "EpicKitchenSequentialDataset",
                            "AVADetectionDataset"]


def configure_dataloader(
    command_line_args: argparse.Namespace,
    dataset_name: SupportedDatasets,
    cfg=None,
):
    """dataloader factory

    Args:
        command_line_args (argparse.Namespace): command line args
        dataset_name (SupportedDatasets): dataset name (str).
            ["ImageFolder", "VideoFolder", "SequentialVideoFolder", "EpicKitchenSequentialDataset", "AVADetectionDataset"]

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
            build_epic_kitchens_sequential_transform(TransformVideoInfo(
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

    elif dataset_name == "AVADetectionDataset":
        if cfg is None:
            cfg = get_cfg()
            if args.cfg_file is not None:
                cfg.merge_from_file(args.cfg_file)
            if args.opts is not None:
                cfg.merge_from_list(args.opts)

        train_transform, val_transform = \
            build_ava_sequential_transform(TransformVideoInfo(
                frames_per_clip=cfg.DATA.NUM_FRAMES,
                val_shorter_side_size=cfg.DATA.TEST_CROP_SIZE,
                crop_size=cfg.DATA.TRAIN_CROP_SIZE,
            ))
        train_loader, val_loader, n_classes = \
            ava_sequential_data_folder(
                AvaSequentialDataFolderInfo(
                    cfg=cfg,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    train_transform=train_transform,
                    val_transform=val_transform,
                ))

    else:
        raise ValueError("invalid dataset_name")

    return DataloadersInfo(
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=n_classes
    )
