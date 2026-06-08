from typing import Literal
from dataclasses import dataclass
import argparse
from pathlib import Path

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


def _infer_salads50_root(args: argparse.Namespace) -> Path:
    candidate_roots: list[Path] = []
    for path_value in (args.train_dir, args.val_dir):
        path = Path(path_value)
        if path.is_absolute():
            candidate_roots.append(path.parent)
            candidate_roots.append(path.parent.parent)

    candidate_roots.append(Path(args.root))

    wanted_markers = (
        "framelabels_custom",
        "framelabels",
        "activityAnnotations",
        "rgb",
    )

    for candidate_root in candidate_roots:
        if any((candidate_root / marker).exists() for marker in wanted_markers):
            return candidate_root

    for candidate_root in candidate_roots:
        if candidate_root != Path(args.root):
            return candidate_root

    return Path(args.root)


def _first_existing_child(root: Path, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if (root / candidate).exists():
            return candidate
    return candidates[0]


def _infer_salads50_ext(video_root: Path, requested_ext: str) -> str:
    if requested_ext != "*.MP4,*.mp4":
        return requested_ext

    for pattern in ("*.avi", "*.AVI"):
        if any(video_root.glob(f"**/{pattern}")):
            return pattern

    return requested_ext


def _maybe_assign_default_memvit_cfg(
    args: argparse.Namespace,
    dataset_name: SupportedDatasets,
) -> None:
    if getattr(args, "cfg_file", None) is not None:
        return

    model_name = getattr(args, "model_name", None)
    if model_name not in {"memvit", "vit_b"}:
        return

    config_name: str | None = None
    if dataset_name == "Salads50SequentialDataset":
        config_name = "MeMViT_16_50Salads_frame.yaml"
    elif dataset_name == "EpicKitchenSequentialDataset":
        if getattr(args, "epic_label_type", None) == "verb_noun":
            config_name = "MeMViT_16_K400_multi_classes.yaml"
        else:
            config_name = "MeMViT_16_K400.yaml"
    elif dataset_name in {"SequentialVideoFolder", "VideoFolder"}:
        config_name = "MeMViT_16_K400.yaml"

    if config_name is None:
        return

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / config_name
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Default MemViT config file not found: {cfg_path}"
        )

    args.cfg_file = str(cfg_path)


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
    _maybe_assign_default_memvit_cfg(args, dataset_name)

    if dataset_name == "ImageFolder":
        train_transform, val_transform =             transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes =             image_folder(ImageFolderInfo(
                root=args.root,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataset_name == "VideoFolder":
        train_transform, val_transform =             transform_video(TransformVideoInfo(
                frames_per_clip=args.frames_per_clip
            ))
        train_loader, val_loader, n_classes =             video_folder(VideoFolderInfo(
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
        train_transform, val_transform =             build_sequential_video_transform(TransformVideoInfo(
                frames_per_clip=args.frames_per_clip
            ))
        train_loader, val_loader, n_classes =             epic_kitchens_sequential_data_folder(
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
        salads_root = _infer_salads50_root(args)
        annotation_root = args.annotation_root
        if annotation_root is None:
            annotation_root = _first_existing_child(
                salads_root,
                ("framelabels_custom", "framelabels", "activityAnnotations"),
            )

        split_root = args.split_root
        if split_root is None:
            split_root = _first_existing_child(
                salads_root,
                ("framelabels_custom", "activityAnnotations"),
            )

        def _resolve_salads50_path(path_value: str | None) -> str | None:
            if path_value is None:
                return None
            path = Path(path_value)
            if not path.is_absolute():
                path = salads_root / path
            return str(path)

        annotation_root_path = _resolve_salads50_path(annotation_root)
        train_annotation_root_path = _resolve_salads50_path(args.train_annotation_root)
        val_annotation_root_path = _resolve_salads50_path(args.val_annotation_root)

        train_dir = args.train_dir
        val_dir = args.val_dir
        if train_dir == "train" and val_dir == "val":
            train_dir = "rgb/train"
            val_dir = "rgb/val"

        split_root_path = _resolve_salads50_path(split_root)
        train_split_root_path = _resolve_salads50_path(args.train_split_root)
        val_split_root_path = _resolve_salads50_path(args.val_split_root)

        ext = _infer_salads50_ext(salads_root, args.ext)

        train_transform, val_transform =             build_sequential_video_transform(TransformVideoInfo(
                frames_per_clip=args.frames_per_clip
            ))
        train_loader, val_loader, n_classes =             salads50_sequential_data_folder(
                Salads50SequentialDataFolderInfo(
                    root=args.root,
                    train_dir=train_dir,
                    val_dir=val_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    train_transform=train_transform,
                    val_transform=val_transform,
                    clip_duration=args.clip_duration,
                    video_edge_time=args.video_edge_time,
                    ext=ext,
                    annotation_root=str(annotation_root_path),
                    split_root=str(split_root_path),
                    train_annotation_root=train_annotation_root_path,
                    val_annotation_root=val_annotation_root_path,
                    train_split_root=train_split_root_path,
                    val_split_root=val_split_root_path,
                    split_id=args.split_id,
                    label_granularity=args.label_granularity,
                    label_map_path=args.label_map_path,
                    background_label=args.background_label,
                    max_train_clips_per_video=args.max_train_clips_per_video,
                    frames_per_clip=args.frames_per_clip,
                ))

    else:
        raise ValueError("invalid dataset_name")

    return DataloadersInfo(
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=n_classes
    )
