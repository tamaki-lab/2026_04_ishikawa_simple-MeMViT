from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import torch

from ..base_sequential_video_dataset import (
    BaseSequentialVideoDataset,
    LogicalSampleSpec,
    SubSampleSpec,
    VideoItem,
)


@dataclass(frozen=True)
class Salads50SplitSpec:
    split_name: str
    candidate_names: tuple[str, ...]


class Salads50SequentialDataset(BaseSequentialVideoDataset):
    VALID_LABEL_GRANULARITIES = {"coarse", "fine"}
    SPLIT_SPECS = {
        "train": Salads50SplitSpec(
            split_name="train",
            candidate_names=(
                "train.split{split_id}.bundle",
                "train.split{split_id}.txt",
                "train.split{split_id}",
            ),
        ),
        "val": Salads50SplitSpec(
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
    VIDEO_ID_PREFIX = "rgb-"
    VIDEO_ID_SUFFIXES = (
        "-activityAnnotation",
        "-framelabels",
    )

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
        background_label="background",
        max_train_clips_per_video=None,
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
            raise ValueError("annotation_root is required for Salads50SequentialDataset")
        if split_root is None:
            raise ValueError("split_root is required for Salads50SequentialDataset")

        self.annotation_root = Path(annotation_root)
        self.split_root = Path(split_root)
        self.split_id = split_id
        self.label_granularity = label_granularity
        self.label_map_path = Path(label_map_path) if label_map_path is not None else None
        self.background_label = background_label
        self.max_train_clips_per_video = max_train_clips_per_video
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

        video_path_by_id = {
            self.resolve_video_id(path): path
            for path in self.list_video_paths()
        }
        missing_video_ids = [
            video_id
            for video_id in split_video_ids
            if video_id not in video_path_by_id
        ]
        if missing_video_ids:
            warnings.warn(
                "Skipping missing 50Salads videos under "
                f"{self.video_path}: {missing_video_ids[:5]}",
                stacklevel=2,
            )

        split_video_ids = [
            video_id for video_id in split_video_ids
            if video_id in video_path_by_id
        ]
        self.allowed_video_ids = set(split_video_ids)
        self.frame_labels_by_video = {
            video_id: self.parse_frame_labels(
                video_id,
                video_path_by_id[video_id],
            )
            for video_id in split_video_ids
        }

        return [
            VideoItem(
                path=video_path_by_id[video_id],
                video_id=video_id,
                meta={
                    "dataset": "50salads",
                    "annotation_path": self.annotation_path_by_video[video_id],
                },
            )
            for video_id in split_video_ids
        ]

    def build_class_to_idx(self) -> dict[str, int]:
        if self.label_map_path is not None:
            labels = self.parse_label_map(self.label_map_path)
        else:
            labels = sorted({
                label_name
                for frame_labels in self.frame_labels_by_video.values()
                for label_name in frame_labels
            })

        if self.background_label not in labels:
            labels.append(self.background_label)

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

    def make_subsample_specs(
        self,
        logical_sample: LogicalSampleSpec,
    ) -> list[SubSampleSpec]:
        sub_samples = super().make_subsample_specs(logical_sample)
        if not self.is_train:
            return sub_samples

        if logical_sample.video_id not in self.frame_labels_by_video:
            raise KeyError(f"Missing frame labels for video_id={logical_sample.video_id}")

        kept_sub_samples = []
        for sub_sample in sub_samples:
            if any(sub_sample.valid_mask):
                kept_sub_samples.append(sub_sample)

        if self.max_train_clips_per_video is not None:
            if self.max_train_clips_per_video <= 0:
                raise ValueError(
                    "max_train_clips_per_video must be a positive integer when provided"
                )
            kept_sub_samples = kept_sub_samples[:self.max_train_clips_per_video]

        num_subsamples = len(kept_sub_samples)
        return [
            SubSampleSpec(
                sample_id=logical_sample.sample_id,
                sub_id=sub_id,
                num_subsamples=num_subsamples,
                frame_indices=sub_sample.frame_indices,
                valid_mask=sub_sample.valid_mask,
                is_first=(sub_id == 0),
                is_last=(sub_id == num_subsamples - 1),
            )
            for sub_id, sub_sample in enumerate(kept_sub_samples)
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
        if split_path.is_dir():
            video_ids = [
                self.normalize_video_id(path.stem)
                for path in sorted(split_path.glob("*.txt"))
                if path.is_file()
            ]
            if not video_ids:
                raise ValueError(f"Split directory is empty: {split_path}")
            return video_ids

        video_ids: list[str] = []
        seen_video_ids: set[str] = set()
        for line in split_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            video_id = self.normalize_video_id(Path(stripped).stem)
            if video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)
            video_ids.append(video_id)

        if not video_ids:
            raise ValueError(f"Split file is empty: {split_path}")
        return video_ids

    def resolve_split_path(self, split_spec: Salads50SplitSpec) -> Path:
        split_dir_candidates = [
            self.split_root / split_spec.split_name,
            self.split_root / f"{split_spec.split_name}{self.split_id}",
            self.split_root,
        ]
        for candidate_path in split_dir_candidates:
            if candidate_path.exists() and candidate_path.is_dir():
                return candidate_path

        for candidate_name in split_spec.candidate_names:
            candidate_path = self.split_root / candidate_name.format(split_id=self.split_id)
            if candidate_path.exists():
                return candidate_path
        raise FileNotFoundError(
            "Could not find a 50Salads split file for "
            f"{split_spec.split_name} split_id={self.split_id} under {self.split_root}"
        )

    def parse_frame_labels(self, video_id: str, video_path: Path) -> list[str]:
        annotation_path = self.resolve_annotation_path(video_id)
        raw_lines = annotation_path.read_text().splitlines()
        total_frames = self.resolve_total_frames(video_path)

        indexed_entries: list[tuple[int, str]] = []
        interval_entries: list[tuple[int, int, str]] = []
        sequential_labels: list[str] = []

        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if not parts:
                continue

            parsed_label = self.parse_annotation_line(raw_line)
            if parsed_label is None:
                continue

            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                interval_entries.append((int(parts[0]), int(parts[1]), parsed_label))
                continue

            if parts[0].isdigit():
                indexed_entries.append((int(parts[0]), parsed_label))
                continue

            sequential_labels.append(parsed_label)

        labels: list[str]
        if interval_entries:
            labels = self.expand_interval_labels(
                interval_entries=interval_entries,
                total_frames=total_frames,
            )
        elif indexed_entries:
            labels = self.expand_indexed_labels(
                indexed_entries=indexed_entries,
                total_frames=total_frames,
            )
        else:
            labels = sequential_labels

        if not labels:
            raise ValueError(f"Annotation file is empty: {annotation_path}")
        self.annotation_path_by_video[video_id] = annotation_path
        return labels

    def parse_annotation_line(self, raw_line: str) -> str | None:
        stripped = raw_line.strip()
        if not stripped:
            return None

        parts = stripped.split()
        if not parts:
            return None

        if len(parts) == 1:
            return self.background_label if parts[0].isdigit() else parts[0]

        coarse_label = parts[1]
        fine_label = parts[2] if len(parts) >= 3 else coarse_label

        if self.label_granularity == "coarse":
            return coarse_label
        return fine_label

    def resolve_annotation_path(self, video_id: str) -> Path:
        candidate_names = [
            f"{video_id}.txt",
            f"{video_id}-activityAnnotation.txt",
            f"{video_id}-framelabels.txt",
        ]
        for candidate_name in candidate_names:
            direct_path = self.annotation_root / candidate_name
            if direct_path.exists():
                return direct_path

            matches = sorted(self.annotation_root.glob(f"**/{candidate_name}"))
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Could not find 50Salads annotation for video_id={video_id} under {self.annotation_root}"
        )

    def resolve_total_frames(self, video_path: Path) -> int | None:
        container, stream, fps = self.open_video(video_path)
        try:
            if stream.frames and stream.frames > 0:
                return int(stream.frames)

            if stream.duration is not None and stream.time_base is not None:
                duration_seconds = float(stream.duration * stream.time_base)
                estimated_frames = round(duration_seconds * fps)
                if estimated_frames > 0:
                    return estimated_frames
        finally:
            container.close()

        return None

    def expand_interval_labels(
        self,
        interval_entries: list[tuple[int, int, str]],
        total_frames: int | None,
    ) -> list[str]:
        max_frame_id = max(end_frame for _, end_frame, _ in interval_entries)
        resolved_total_frames = max(total_frames or 0, max_frame_id)
        labels = [self.background_label] * resolved_total_frames

        for start_frame, end_frame, label_name in interval_entries:
            start_idx = max(start_frame - 1, 0)
            end_idx = min(max(end_frame, 0), resolved_total_frames)
            if start_idx >= end_idx:
                continue
            labels[start_idx:end_idx] = [label_name] * (end_idx - start_idx)

        return labels

    def expand_indexed_labels(
        self,
        indexed_entries: list[tuple[int, str]],
        total_frames: int | None,
    ) -> list[str]:
        max_frame_id = max(frame_id for frame_id, _ in indexed_entries)
        resolved_total_frames = max(total_frames or 0, max_frame_id)
        labels = [self.background_label] * resolved_total_frames

        for frame_id, label_name in indexed_entries:
            frame_idx = frame_id - 1
            if 0 <= frame_idx < resolved_total_frames:
                labels[frame_idx] = label_name

        return labels

    @classmethod
    def normalize_video_id(cls, value: str) -> str:
        normalized = value.removeprefix(cls.VIDEO_ID_PREFIX)
        for suffix in cls.VIDEO_ID_SUFFIXES:
            normalized = normalized.removesuffix(suffix)
        return normalized

    def __len__(self):
        total_subsamples = 0
        for video_item in self.video_items:
            logical_samples = self.iter_logical_samples(video_item, fps=0.0)
            for logical_sample in logical_samples:
                total_subsamples += len(self.make_subsample_specs(logical_sample))
        return total_subsamples

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
        return self.normalize_video_id(video_path.stem)
