from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pandas as pd

from ..base_sequential_video_dataset import (
    BaseSequentialVideoDataset,
    LogicalSampleSpec,
    SubSampleSpec,
    VideoItem,
)


@dataclass
class EpicKitchenActionInterval:
    start_frame: int
    stop_frame: int
    verb_label: str
    noun_label: str

    @property
    def action_label(self) -> str:
        return f"{self.verb_label}:{self.noun_label}"


EpicKitchenLabel: TypeAlias = str | tuple[str, str]


class EpicKitchenSequentialDataset(BaseSequentialVideoDataset):
    VALID_TASKS = {
        "action_recognition",
        "action_anticipation",
    }

    VALID_LABEL_TYPES = {
        "verb",
        "noun",
        "action",
        "verb_noun",
    }

    def __init__(
        self,
        video_path,
        clip_duration,
        video_edge_time,
        ext,
        is_train,
        transform,
        annotation_path,
        frames_per_clip=16,
        batch_size=1,
        task="action_recognition",
        label_type="verb",
        background_label="background",
        anticipation_time=1.0,
        recognition_label_strategy="center",
        shuffle=True,
    ):
        if task not in self.VALID_TASKS:
            raise ValueError(
                f"task must be one of {sorted(self.VALID_TASKS)}, but got {task}"
            )

        if label_type not in self.VALID_LABEL_TYPES:
            raise ValueError(
                f"label_type must be one of {sorted(self.VALID_LABEL_TYPES)}, "
                f"but got {label_type}"
            )

        if annotation_path is None:
            raise ValueError("annotation_path is required for EpicKitchenSequentialDataset")

        self.annotation_path = Path(annotation_path)
        self.label_type = label_type
        self.background_label = background_label
        self.anticipation_time = anticipation_time
        self.recognition_label_strategy = recognition_label_strategy

        self.annotations: dict[str, list[EpicKitchenActionInterval]] = {}
        self.sample_id_to_label_name: dict[str, EpicKitchenLabel] = {}
        self.verb_class_to_idx: dict[str, int] = {}
        self.noun_class_to_idx: dict[str, int] = {}

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
        self.annotations = self.parse_annotations()

        video_paths = self.list_video_paths()

        return [
            VideoItem(
                path=path,
                video_id=self.resolve_video_id(path),
                meta={"dataset": "epic_kitchens"},
            )
            for path in sorted(video_paths)
        ]

    def build_class_to_idx(self) -> dict[str, int]:
        if self.label_type == "verb_noun":
            verb_labels = {
                interval.verb_label
                for intervals in self.annotations.values()
                for interval in intervals
            }
            noun_labels = {
                interval.noun_label
                for intervals in self.annotations.values()
                for interval in intervals
            }
            self.verb_class_to_idx = {
                label_name: idx
                for idx, label_name in enumerate(sorted(verb_labels))
            }
            self.noun_class_to_idx = {
                label_name: idx
                for idx, label_name in enumerate(sorted(noun_labels))
            }
            # BaseSequentialVideoDataset expects a single class_to_idx mapping.
            # For paired verb/noun targets, the per-head mappings above are the
            # authoritative ones used in make_target.
            return dict(self.verb_class_to_idx)

        labels = {
            self.get_label_name_from_interval(interval)
            for intervals in self.annotations.values()
            for interval in intervals
        }

        return {
            label_name: idx
            for idx, label_name in enumerate(sorted(labels))
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

    def iter_logical_samples(
        self,
        video_item: VideoItem,
        fps: float,
    ) -> list[LogicalSampleSpec]:
        intervals = self.annotations.get(video_item.video_id, [])
        logical_samples: list[LogicalSampleSpec] = []

        for interval_index, interval in enumerate(intervals):
            sample_id = f"{video_item.video_id}_{self.task}_{interval_index:06d}"
            label_name = self.get_label_name_from_interval(interval)

            if self.task == "action_recognition":
                start_frame = interval.start_frame
                end_frame = interval.stop_frame + 1
            elif self.task == "action_anticipation":
                context_end_frame = interval.start_frame - int(
                    round(self.anticipation_time * fps)
                )
                context_length = int(round(self.clip_duration * fps))
                start_frame = max(0, context_end_frame - context_length)
                end_frame = context_end_frame
            else:
                raise ValueError(f"Unsupported task: {self.task}")

            if end_frame <= start_frame:
                continue

            self.sample_id_to_label_name[sample_id] = label_name
            logical_samples.append(
                LogicalSampleSpec(
                    sample_id=sample_id,
                    video_id=video_item.video_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

        return logical_samples

    def make_target(self, logical_sample: LogicalSampleSpec) -> int | tuple[int, int]:
        label_name = self.sample_id_to_label_name.get(logical_sample.sample_id)
        if label_name is None:
            raise KeyError(
                f"Missing label for logical sample: {logical_sample.sample_id}"
            )

        if self.label_type == "verb_noun":
            if not isinstance(label_name, tuple):
                raise TypeError(
                    "Expected a (verb_label, noun_label) target for verb_noun mode."
                )
            return (
                self.verb_class_to_idx[label_name[0]],
                self.noun_class_to_idx[label_name[1]],
            )

        return self.class_to_idx[label_name]

    def build_evaluation_info(
        self,
        video_item: VideoItem,
        logical_sample: LogicalSampleSpec,
        sub_sample: SubSampleSpec,
        fps: float,
        worker_id: int,
    ) -> dict[str, str]:
        if self.task == "action_recognition":
            aggregation_strategy = "mean"
        elif self.task == "action_anticipation":
            aggregation_strategy = "last"
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        return {
            "evaluation_mode": "grouped_topk",
            "group_id": logical_sample.sample_id,
            "aggregation_strategy": aggregation_strategy,
        }

    def parse_annotations(self) -> dict[str, list[EpicKitchenActionInterval]]:
        df = pd.read_csv(self.annotation_path)

        required_columns = {
            "video_id",
            "start_frame",
            "stop_frame",
            "verb_class",
            "noun_class",
        }
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                "EPIC-KITCHENS annotation CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )

        annotations: dict[str, list[EpicKitchenActionInterval]] = {}
        for _, row in df.iterrows():
            video_id = str(row["video_id"])
            interval = EpicKitchenActionInterval(
                start_frame=int(row["start_frame"]),
                stop_frame=int(row["stop_frame"]),
                verb_label=str(row["verb_class"]),
                noun_label=str(row["noun_class"]),
            )
            annotations.setdefault(video_id, []).append(interval)

        for intervals in annotations.values():
            intervals.sort(key=lambda item: item.start_frame)

        return annotations

    def resolve_video_id(self, video_path: Path) -> str:
        stem = video_path.stem

        if stem in self.annotations:
            return stem

        split_stem = stem.split("-", 1)
        if len(split_stem) == 2 and split_stem[1] in self.annotations:
            return split_stem[1]

        annotation_id = stem.rsplit("-", 1)[0]
        if annotation_id in self.annotations:
            return annotation_id

        return stem

    def get_label_name_from_interval(
        self,
        interval: EpicKitchenActionInterval,
    ) -> EpicKitchenLabel:
        if self.label_type == "verb":
            return interval.verb_label

        if self.label_type == "noun":
            return interval.noun_label

        if self.label_type == "action":
            return interval.action_label

        if self.label_type == "verb_noun":
            return (interval.verb_label, interval.noun_label)

        raise ValueError(f"Unsupported label_type: {self.label_type}")
