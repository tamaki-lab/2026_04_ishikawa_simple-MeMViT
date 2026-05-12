from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..base_sequential_video_dataset import (
    BaseSequentialVideoDataset,
    LogicalSampleSpec,
    SubSampleSpec,
    VideoItem,
)


@dataclass(frozen=True)
class FiftySaladsSplitSpec:
    split_name: str
    candidate_names: tuple[str, ...]


class FiftySaladsSequentialDataset(BaseSequentialVideoDataset):
    VALID_LABEL_GRANULARITIES = {"fine"}
    SPLIT_SPECS = {
        "train": FiftySaladsSplitSpec(
            split_name="train",
            candidate_names=(
                "train.split{split_id}.bundle",
                "train.split{split_id}.txt",
                "train.split{split_id}",
            ),
        ),
        "val": FiftySaladsSplitSpec(
            split_name="val",
            candidate_names=(
                "test.split{split_id}.bundle",
                "test.split{split_id}.txt",
                "test.split{split_id}",
                "val.split{split_id}.bundle",
                "val.split{split_id}.txt",
                "val.split{split_id}",
            ),
        ),
    }

    def __init__(
        self,
        video_path,
        clip_duration,
        video_edge_time,
        ext,
        is_train,
        transform,
        annotation_root,
        split_root,
        split_id,
        label_granularity="fine",
        label_map_path=None,
        frames_per_clip=16,
        batch_size=1,
        task="online_action_recognition",
        shuffle=True,
        split_name=None,
    ):
        if label_granularity not in self.VALID_LABEL_GRANULARITIES:
            raise ValueError(
                f"label_granularity must be one of {sorted(self.VALID_LABEL_GRANULARITIES)}, "
                f"but got {label_granularity}"
            )
        if annotation_root is None:
            raise ValueError("annotation_root is required for FiftySaladsSequentialDataset")
        if split_root is None:
            raise ValueError("split_root is required for FiftySaladsSequentialDataset")

        self.annotation_root = Path(annotation_root)
        self.split_root = Path(split_root)
        self.split_id = split_id
        self.label_granularity = label_granularity
        self.label_map_path = Path(label_map_path) if label_map_path is not None else None
        self.split_name = split_name or ("train" if is_train else "val")

        self.frame_labels_by_video: dict[str, list[str]] = {}
        self.annotation_path_by_video: dict[str, Path] = {}
        self.allowed_video_ids: set[str] = set()

        super().__init__(
            video_path=video_path,
            clip_duration=clip_duration,
            video_edge_time=video_edge_time,
            ext=ext,
            is_train=is_train,
            transform=transform,
            frames_per_clip=frames_per_clip,
            batch_size=batch_size,
            task=task,
            shuffle=shuffle,
        )

    def load_video_items(self) -> list[VideoItem]:
        split_video_ids = self.parse_split_video_ids()
        self.allowed_video_ids = set(split_video_ids)
        self.frame_labels_by_video = {
            video_id: self.parse_frame_labels(video_id)
            for video_id in split_video_ids
        }

        video_paths = self.list_video_paths()
        items: list[VideoItem] = []
        for path in video_paths:
            video_id = self.resolve_video_id(path)
            if video_id not in self.allowed_video_ids:
                continue
            items.append(
                VideoItem(
                    path=path,
                    video_id=video_id,
                    meta={
                        "dataset": "50salads",
                        "annotation_path": self.annotation_path_by_video[video_id],
                    },
                )
            )
        return items

    def build_class_to_idx(self) -> dict[str, int]:
        if self.label_map_path is not None:
            labels = self.parse_label_map(self.label_map_path)
        else:
            labels = sorted({
                label_name
                for frame_labels in self.frame_labels_by_video.values()
                for label_name in frame_labels
            })

        return {
            label_name: idx for idx, label_name in enumerate(labels)
        }

    def iter_logical_samples(
        self,
        video_item: VideoItem,
        fps: float,
    ) -> list[LogicalSampleSpec]:
        del fps
        frame_labels = self.frame_labels_by_video.get(video_item.video_id)
        if frame_labels is None:
            raise KeyError(f"Missing frame labels for video_id={video_item.video_id}")
        if not frame_labels:
            return []

        return [
            LogicalSampleSpec(
                sample_id=video_item.video_id,
                video_id=video_item.video_id,
                start_frame=0,
                end_frame=len(frame_labels),
            )
        ]

    def make_target(self, logical_sample: LogicalSampleSpec) -> torch.Tensor:
        frame_labels = self.frame_labels_by_video.get(logical_sample.video_id)
        if frame_labels is None:
            raise KeyError(f"Missing frame labels for video_id={logical_sample.video_id}")
        return torch.tensor(
            [self.class_to_idx[label_name] for label_name in frame_labels],
            dtype=torch.long,
        )

    def make_target_for_subsample(
        self,
        logical_sample: LogicalSampleSpec,
        sub_sample: SubSampleSpec,
        logical_target: torch.Tensor,
    ) -> torch.Tensor:
        del logical_sample
        frame_index_tensor = torch.tensor(sub_sample.frame_indices, dtype=torch.long)
        return logical_target[frame_index_tensor].clone()

    def build_evaluation_info(
        self,
        video_item: VideoItem,
        logical_sample: LogicalSampleSpec,
        sub_sample: SubSampleSpec,
        fps: float,
        worker_id: int,
    ) -> dict[str, str]:
        del video_item, logical_sample, sub_sample, fps, worker_id
        return {
            "evaluation_mode": "framewise",
        }

    def list_video_paths(self) -> list[Path]:
        patterns = [pattern.strip() for pattern in str(self.ext).split(",")]
        patterns = [pattern for pattern in patterns if pattern]
        if not patterns:
            patterns = [str(self.ext)]

        video_paths = {
            path
            for pattern in patterns
            for path in Path(self.video_path).glob(f"**/{pattern}")
            if not path.is_dir()
        }
        return sorted(video_paths)

    def parse_split_video_ids(self) -> list[str]:
        split_spec = self.SPLIT_SPECS.get(self.split_name)
        if split_spec is None:
            raise ValueError(f"Unsupported split_name: {self.split_name}")

        split_path = self.resolve_split_path(split_spec)
        video_ids: list[str] = []
        for line in split_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            video_ids.append(Path(stripped).stem)

        if not video_ids:
            raise ValueError(f"Split file is empty: {split_path}")
        return video_ids

    def resolve_split_path(self, split_spec: FiftySaladsSplitSpec) -> Path:
        for candidate_name in split_spec.candidate_names:
            candidate_path = self.split_root / candidate_name.format(split_id=self.split_id)
            if candidate_path.exists():
                return candidate_path
        raise FileNotFoundError(
            "Could not find a 50Salads split file for "
            f"{split_spec.split_name} split_id={self.split_id} under {self.split_root}"
        )

    def parse_frame_labels(self, video_id: str) -> list[str]:
        annotation_path = self.resolve_annotation_path(video_id)
        labels = [
            line.strip()
            for line in annotation_path.read_text().splitlines()
            if line.strip()
        ]
        if not labels:
            raise ValueError(f"Annotation file is empty: {annotation_path}")
        self.annotation_path_by_video[video_id] = annotation_path
        return labels

    def resolve_annotation_path(self, video_id: str) -> Path:
        direct_path = self.annotation_root / f"{video_id}.txt"
        if direct_path.exists():
            return direct_path

        matches = sorted(self.annotation_root.glob(f"**/{video_id}.txt"))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Could not find 50Salads annotation for video_id={video_id} under {self.annotation_root}"
        )

    def parse_label_map(self, label_map_path: Path) -> list[str]:
        labels_by_id: dict[int, str] = {}
        ordered_labels: list[str] = []

        for line in label_map_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) == 1:
                ordered_labels.append(parts[0])
                continue

            first, second = parts[0], parts[1]
            if first.isdigit():
                labels_by_id[int(first)] = second
                continue
            if second.isdigit():
                labels_by_id[int(second)] = first
                continue

            ordered_labels.append(first)

        if labels_by_id:
            return [label_name for _, label_name in sorted(labels_by_id.items())]
        return ordered_labels

    def resolve_video_id(self, video_path: Path) -> str:
        return video_path.stem
