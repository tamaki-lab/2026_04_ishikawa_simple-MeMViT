
import torch
from .sequential_video_dataset import SequentialVideoDataset

from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
    Compose,
)
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)


def sequential_video_folder(
        clip_duration,
        video_edge_time,
        batch_size,
        num_workers,
        ext,
        train_dir,
        val_dir,
        frames_per_clip=16,
        label_mode="video",
        train_annotation_path=None,
        val_annotation_path=None,
        background_label="background",
        epic_label_type="verb",
):

    train_dataset = SequentialVideoDataset(
        clip_duration,
        video_edge_time,
        num_workers,
        ext,
        train_dir,
        is_train=True,
        transform=transform_video(is_train=True, frames_per_clip=frames_per_clip),
        label_mode=label_mode,
        annotation_path=train_annotation_path,
        background_label=background_label,
        frames_per_clip=frames_per_clip,
        batch_size=batch_size,
        epic_label_type=epic_label_type,
    )

    val_dataset = SequentialVideoDataset(
        clip_duration,
        video_edge_time,
        num_workers,
        ext,
        val_dir,
        is_train=False,
        transform=transform_video(is_train=False, frames_per_clip=frames_per_clip),
        label_mode=label_mode,
        annotation_path=val_annotation_path,
        background_label=background_label,
        frames_per_clip=frames_per_clip,
        batch_size=batch_size,
        epic_label_type=epic_label_type,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    n_classes = train_dataset.num_classes
    return train_loader, val_loader, n_classes


def transform_video(is_train, frames_per_clip=16):
    transform_list = [
        UniformTemporalSubsample(frames_per_clip),
        transforms.Lambda(lambda x: x / 255.),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]),
    ]

    if is_train:
        transform_list.extend([
            RandomShortSideScale(min_size=256, max_size=320,),
            RandomCrop(224),
            RandomHorizontalFlip(),
        ])
    else:
        transform_list.extend([
            ShortSideScale(256),
            CenterCrop(224),
        ])

    transform = Compose(transform_list)
    return transform


def collate_fn(batch):
    subclips = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    frame_indices = [item[2] for item in batch]
    infos = [item[3] for item in batch]
    return subclips, labels, frame_indices, infos
