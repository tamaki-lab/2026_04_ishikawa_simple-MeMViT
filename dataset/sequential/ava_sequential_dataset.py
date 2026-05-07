from __future__ import annotations

from bisect import bisect_left
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image
import torch

from .base_sequential_video_dataset import (
    BaseSequentialVideoDataset,
    LogicalSampleSpec,
    SubSampleSpec,
    VideoItem,
)


@dataclass(frozen=True)
class FrameDirectoryContainer:
    frame_dir: Path
    video_id: str
    frame_numbers: tuple[int, ...]

    def close(self):
        return None


@dataclass
class AvaBoxRecord:
    normalized_box: tuple[float, float, float, float]
    score: float
    label_ids: set[int]


class AvaSequentialDataset(BaseSequentialVideoDataset):
    AVA_FPS = 30.0
    FRAME_NAME_TEMPLATE = "{frame_number:06d}.jpg"
    BOX_KEY_PRECISION = 6

    def __init__(
        self,
        video_path,
        clip_duration,
        video_edge_time,
        ext,
        is_train,
        transform,
        frame_list_paths,
        box_list_paths,
        label_map_path,
        frames_per_clip=16,
        batch_size=1,
        sampling_rate=4,
        detection_score_thresh=0.9,
        exclusion_path=None,
        groundtruth_path=None,
        shuffle=True,
    ):
        self.frame_list_paths = [Path(path) for path in frame_list_paths]
        self.box_list_paths = [Path(path) for path in box_list_paths]
        self.label_map_path = Path(label_map_path)
        self.sampling_rate = sampling_rate
        self.detection_score_thresh = detection_score_thresh
        self.exclusion_path = (
            None if exclusion_path is None else Path(exclusion_path)
        )
        self.groundtruth_path = (
            None if groundtruth_path is None else Path(groundtruth_path)
        )

        self.frame_numbers_by_video: dict[str, tuple[int, ...]] = {}
        self.keyframes_by_video: dict[str, dict[int, list[AvaBoxRecord]]] = {}
        self.label_map: dict[int, int] = {}
        self.sample_id_to_timestamp: dict[str, int] = {}
        self.sample_id_to_frame_indices: dict[str, list[int]] = {}
        self.sample_id_to_target: dict[str, dict[str, Any]] = {}
        self._active_target: dict[str, Any] | None = None

        super().__init__(
            video_path=video_path,
            clip_duration=clip_duration,
            video_edge_time=video_edge_time,
            ext=ext,
            is_train=is_train,
            transform=transform,
            frames_per_clip=frames_per_clip,
            batch_size=batch_size,
            task="ava_detection",
            shuffle=shuffle,
        )

    def load_video_items(self) -> list[VideoItem]:
        self.label_map = self.parse_label_map()
        self.frame_numbers_by_video = self.parse_frame_lists()
        self.keyframes_by_video = self.parse_keyframes()

        frame_video_ids = set(self.frame_numbers_by_video.keys())
        annotation_video_ids = set(self.keyframes_by_video.keys())
        video_ids = sorted(frame_video_ids & annotation_video_ids)

        return [
            VideoItem(
                path=self.video_path / video_id,
                video_id=video_id,
                meta={
                    "dataset": "ava",
                    "frame_numbers": self.frame_numbers_by_video[video_id],
                },
            )
            for video_id in video_ids
            if (self.video_path / video_id).is_dir()
        ]

    def build_class_to_idx(self) -> dict[str, int]:
        return {
            str(action_id): class_idx
            for action_id, class_idx in sorted(self.label_map.items())
        }

    def iter_logical_samples(
        self,
        video_item: VideoItem,
        fps: float,
    ) -> list[LogicalSampleSpec]:
        del fps

        available_frames = self.frame_numbers_by_video.get(video_item.video_id, ())
        if not available_frames:
            return []

        logical_samples: list[LogicalSampleSpec] = []
        timestamps = sorted(self.keyframes_by_video.get(video_item.video_id, {}).keys())
        for timestamp in timestamps:
            sample_id = f"{video_item.video_id}_{timestamp:04d}"
            frame_indices = self.build_sample_frame_indices(
                video_id=video_item.video_id,
                timestamp=timestamp,
            )
            if not frame_indices:
                continue

            self.sample_id_to_timestamp[sample_id] = timestamp
            self.sample_id_to_frame_indices[sample_id] = frame_indices
            logical_samples.append(
                LogicalSampleSpec(
                    sample_id=sample_id,
                    video_id=video_item.video_id,
                    start_frame=min(frame_indices),
                    end_frame=max(frame_indices) + 1,
                )
            )

        return logical_samples

    def make_subsample_specs(
        self,
        logical_sample: LogicalSampleSpec,
    ) -> list[SubSampleSpec]:
        frame_indices = self.sample_id_to_frame_indices.get(logical_sample.sample_id)
        if frame_indices is None:
            return []

        return [
            SubSampleSpec(
                sample_id=logical_sample.sample_id,
                sub_id=0,
                num_subsamples=1,
                frame_indices=frame_indices,
                valid_mask=[True] * len(frame_indices),
                is_first=True,
                is_last=True,
            )
        ]

    def open_video(self, path):
        video_id = Path(path).name
        frame_numbers = self.frame_numbers_by_video.get(video_id)
        if frame_numbers is None:
            raise KeyError(f"Unknown AVA video_id: {video_id}")

        container = FrameDirectoryContainer(
            frame_dir=Path(path),
            video_id=video_id,
            frame_numbers=frame_numbers,
        )
        return container, None, self.AVA_FPS

    def read_subsample_frames(
        self,
        container,
        stream,
        fps: float,
        sub_sample: SubSampleSpec,
    ) -> list[np.ndarray]:
        del stream, fps

        if not isinstance(container, FrameDirectoryContainer):
            raise TypeError("AvaSequentialDataset requires a FrameDirectoryContainer")

        return [
            self.read_frame_image(
                container.frame_dir,
                self.clamp_to_available_frame(container.frame_numbers, frame_index),
            )
            for frame_index in sub_sample.frame_indices
        ]

    def make_target(self, logical_sample: LogicalSampleSpec) -> dict[str, Any]:
        timestamp = self.sample_id_to_timestamp[logical_sample.sample_id]
        box_records = self.keyframes_by_video[logical_sample.video_id][timestamp]

        normalized_boxes = torch.tensor(
            [record.normalized_box for record in box_records],
            dtype=torch.float32,
        )
        labels = torch.zeros(
            (len(box_records), len(self.label_map)),
            dtype=torch.float32,
        )
        scores = torch.tensor(
            [record.score for record in box_records],
            dtype=torch.float32,
        )

        for box_index, record in enumerate(box_records):
            for action_id in record.label_ids:
                class_idx = self.label_map.get(action_id)
                if class_idx is not None:
                    labels[box_index, class_idx] = 1.0

        target: dict[str, Any] = {
            "boxes": torch.zeros((len(box_records), 4), dtype=torch.float32),
            "labels": labels,
            "orig_boxes": torch.zeros((len(box_records), 4), dtype=torch.float32),
            "scores": scores,
            "video_id": logical_sample.video_id,
            "timestamp": timestamp,
            "keyframe_index": int(round(timestamp * self.AVA_FPS)),
            "orig_size": None,
            "image_size": None,
            "_normalized_boxes": normalized_boxes,
        }

        self.sample_id_to_target[logical_sample.sample_id] = target
        self._active_target = target
        return target

    def build_x(self, clip_frames: list[np.ndarray]):
        clip = np.stack(clip_frames, axis=0)
        clip = np.transpose(clip, (3, 0, 1, 2))
        clip_tensor = torch.from_numpy(clip)

        target = self._active_target
        if target is None:
            return clip_tensor

        normalized_boxes = target.pop("_normalized_boxes")
        height, width = clip_frames[0].shape[:2]
        size_tensor = normalized_boxes.new_tensor([width, height, width, height])
        orig_boxes = normalized_boxes * size_tensor
        target["orig_boxes"] = orig_boxes
        target["orig_size"] = (height, width)

        if self.transform is None:
            target["boxes"] = orig_boxes.clone()
            target["image_size"] = (height, width)
            return clip_tensor

        clip_tensor, resized_boxes, image_size = self.transform(clip_tensor, orig_boxes)
        target["boxes"] = resized_boxes
        target["image_size"] = image_size
        return clip_tensor

    def build_info(
        self,
        video_item: VideoItem,
        logical_sample: LogicalSampleSpec,
        sub_sample: SubSampleSpec,
        fps: float,
        worker_id: int,
    ) -> dict[str, Any]:
        info = super().build_info(
            video_item=video_item,
            logical_sample=logical_sample,
            sub_sample=sub_sample,
            fps=fps,
            worker_id=worker_id,
        )

        target = self.sample_id_to_target.get(logical_sample.sample_id, {})
        info.update(
            {
                "timestamp": self.sample_id_to_timestamp.get(logical_sample.sample_id),
                "keyframe_index": target.get("keyframe_index"),
                "num_boxes": int(target.get("boxes", torch.empty((0, 4))).shape[0]),
                "orig_size": target.get("orig_size"),
                "image_size": target.get("image_size"),
                "detection_task": "ava_detection",
            }
        )
        return info

    def build_evaluation_info(
        self,
        video_item: VideoItem,
        logical_sample: LogicalSampleSpec,
        sub_sample: SubSampleSpec,
        fps: float,
        worker_id: int,
    ) -> dict[str, Any]:
        del video_item, sub_sample, fps, worker_id
        return {
            "evaluation_mode": "ava_detection",
            "sequence_id": logical_sample.sample_id,
            "timestamp": self.sample_id_to_timestamp.get(logical_sample.sample_id),
            "keyframe_index": int(
                round(
                    self.sample_id_to_timestamp.get(logical_sample.sample_id, 0)
                    * self.AVA_FPS
                )
            ),
        }

    def make_sample(self, x, y, frame_indices, info):
        sample = super().make_sample(x=x, y=y, frame_indices=frame_indices, info=info)
        self._active_target = None
        self.sample_id_to_target.pop(info["sample_id"], None)
        return sample

    def parse_label_map(self) -> dict[int, int]:
        label_ids: list[int] = []
        id_pattern = re.compile(r"(?:label_id|id)\s*:\s*(\d+)")

        with self.label_map_path.open() as handle:
            for line in handle:
                match = id_pattern.search(line)
                if match:
                    label_ids.append(int(match.group(1)))

        unique_label_ids = sorted(set(label_ids))
        if not unique_label_ids:
            raise ValueError(f"No AVA labels found in {self.label_map_path}")

        return {
            action_id: class_idx
            for class_idx, action_id in enumerate(unique_label_ids)
        }

    def parse_frame_lists(self) -> dict[str, tuple[int, ...]]:
        frame_numbers_by_video: dict[str, set[int]] = {}

        for frame_list_path in self.frame_list_paths:
            with frame_list_path.open(newline="") as handle:
                reader = csv.reader(handle)
                for row_index, row in enumerate(reader):
                    values = [value.strip() for value in row if value.strip()]
                    if not values:
                        continue
                    if values[0].startswith("#"):
                        continue
                    if row_index == 0 and self.looks_like_frame_list_header(values):
                        continue

                    video_id = self.extract_video_id_from_frame_list(values)
                    frame_number = self.extract_frame_number_from_frame_list(values)
                    if video_id is None or frame_number is None:
                        continue

                    frame_numbers_by_video.setdefault(video_id, set()).add(frame_number)

        return {
            video_id: tuple(sorted(frame_numbers))
            for video_id, frame_numbers in frame_numbers_by_video.items()
        }

    def parse_keyframes(self) -> dict[str, dict[int, list[AvaBoxRecord]]]:
        if self.is_train:
            return self.parse_train_keyframes()

        return self.parse_val_keyframes()

    def parse_train_keyframes(self) -> dict[str, dict[int, list[AvaBoxRecord]]]:
        grouped: dict[str, dict[int, dict[tuple[Any, ...], AvaBoxRecord]]] = {}

        for box_list_path in self.box_list_paths:
            has_labels = True
            is_predicted = "predicted" in box_list_path.name or "detection" in box_list_path.name
            for row in self.iter_ava_rows(
                box_list_path,
                has_labels=has_labels,
                is_predicted=is_predicted,
            ):
                self.merge_box_record(grouped, row)

        return self.finalize_grouped_keyframes(grouped)

    def parse_val_keyframes(self) -> dict[str, dict[int, list[AvaBoxRecord]]]:
        grouped: dict[str, dict[int, dict[tuple[Any, ...], AvaBoxRecord]]] = {}

        for box_list_path in self.box_list_paths:
            for row in self.iter_ava_rows(
                box_list_path,
                has_labels=False,
                is_predicted=True,
            ):
                self.merge_box_record(grouped, row)

        if self.groundtruth_path is not None:
            for row in self.iter_ava_rows(
                self.groundtruth_path,
                has_labels=True,
                is_predicted=False,
            ):
                video_group = grouped.get(row["video_id"], {})
                timestamp_group = video_group.get(row["timestamp"], {})
                box_record = timestamp_group.get(row["box_key"])
                if box_record is None:
                    continue
                label_id = row["label_id"]
                if label_id is not None and label_id > 0:
                    box_record.label_ids.add(label_id)

        exclusions = self.parse_exclusions()
        finalized = self.finalize_grouped_keyframes(grouped)
        if not exclusions:
            return finalized

        filtered: dict[str, dict[int, list[AvaBoxRecord]]] = {}
        for video_id, timestamp_records in finalized.items():
            kept_records = {
                timestamp: records
                for timestamp, records in timestamp_records.items()
                if (video_id, timestamp) not in exclusions
            }
            if kept_records:
                filtered[video_id] = kept_records

        return filtered

    def parse_exclusions(self) -> set[tuple[str, int]]:
        if self.exclusion_path is None or not self.exclusion_path.exists():
            return set()

        exclusions: set[tuple[str, int]] = set()
        with self.exclusion_path.open(newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                values = [value.strip() for value in row if value.strip()]
                if not values:
                    continue
                if values[0].startswith("#") or values[0] == "video_id":
                    continue
                exclusions.add((values[0], int(float(values[1]))))
        return exclusions

    def iter_ava_rows(self, csv_path: Path, has_labels: bool, is_predicted: bool):
        with csv_path.open(newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                values = [value.strip() for value in row if value.strip()]
                if len(values) < 6:
                    continue
                if values[0].startswith("#") or values[0] == "video_id":
                    continue

                video_id = values[0]
                timestamp = int(float(values[1]))
                x1, y1, x2, y2 = (float(values[index]) for index in range(2, 6))
                label_id = int(float(values[6])) if has_labels and len(values) >= 7 else None

                if is_predicted:
                    score_index = 7 if has_labels else 6
                    score = (
                        float(values[score_index])
                        if len(values) > score_index
                        else 1.0
                    )
                else:
                    score = 1.0

                if is_predicted and score < self.detection_score_thresh:
                    continue

                normalized_box = tuple(
                    max(0.0, min(1.0, coord))
                    for coord in (x1, y1, x2, y2)
                )
                yield {
                    "video_id": video_id,
                    "timestamp": timestamp,
                    "label_id": label_id,
                    "score": score,
                    "normalized_box": normalized_box,
                    "box_key": self.make_box_key(video_id, timestamp, normalized_box),
                }

    def merge_box_record(self, grouped, row):
        video_group = grouped.setdefault(row["video_id"], {})
        timestamp_group = video_group.setdefault(row["timestamp"], {})
        box_record = timestamp_group.get(row["box_key"])
        if box_record is None:
            box_record = AvaBoxRecord(
                normalized_box=row["normalized_box"],
                score=row["score"],
                label_ids=set(),
            )
            timestamp_group[row["box_key"]] = box_record

        box_record.score = max(box_record.score, row["score"])
        label_id = row["label_id"]
        if label_id is not None and label_id > 0:
            box_record.label_ids.add(label_id)

    def finalize_grouped_keyframes(self, grouped):
        return {
            video_id: {
                timestamp: list(timestamp_group.values())
                for timestamp, timestamp_group in sorted(timestamp_groups.items())
            }
            for video_id, timestamp_groups in grouped.items()
        }

    def looks_like_frame_list_header(self, values: list[str]) -> bool:
        header_tokens = {"video_id", "original_video_id", "frame_id", "path"}
        return any(value.lower() in header_tokens for value in values)

    def extract_video_id_from_frame_list(self, values: list[str]) -> str | None:
        if len(values) >= 2 and values[1]:
            return values[1]
        if len(values) >= 1 and "/" not in values[0]:
            return values[0]
        if values:
            return Path(values[-1]).parent.name or None
        return None

    def extract_frame_number_from_frame_list(self, values: list[str]) -> int | None:
        if len(values) >= 3:
            try:
                return int(float(values[2]))
            except ValueError:
                pass

        if values:
            return self.extract_frame_number_from_path(values[-1])

        return None

    def extract_frame_number_from_path(self, value: str) -> int | None:
        stem = Path(value).stem
        match = re.search(r"(\d+)$", stem)
        if match is None:
            return None
        return int(match.group(1))

    def build_sample_frame_indices(self, video_id: str, timestamp: int) -> list[int]:
        available_frames = self.frame_numbers_by_video.get(video_id, ())
        if not available_frames:
            return []

        center_frame = int(round(timestamp * self.AVA_FPS))
        half_window = self.frames_per_clip // 2
        frame_indices = [
            center_frame + (index - half_window) * self.sampling_rate
            for index in range(self.frames_per_clip)
        ]

        return [
            self.clamp_to_available_frame(available_frames, frame_index)
            for frame_index in frame_indices
        ]

    def clamp_to_available_frame(
        self,
        available_frames: tuple[int, ...],
        frame_index: int,
    ) -> int:
        if frame_index <= available_frames[0]:
            return available_frames[0]
        if frame_index >= available_frames[-1]:
            return available_frames[-1]

        insert_index = bisect_left(available_frames, frame_index)
        if insert_index < len(available_frames) and available_frames[insert_index] == frame_index:
            return frame_index

        previous_frame = available_frames[insert_index - 1]
        next_frame = available_frames[insert_index]
        if (frame_index - previous_frame) <= (next_frame - frame_index):
            return previous_frame
        return next_frame

    def read_frame_image(self, frame_dir: Path, frame_number: int) -> np.ndarray:
        frame_path = frame_dir / self.FRAME_NAME_TEMPLATE.format(frame_number=frame_number)
        with Image.open(frame_path) as image:
            return np.asarray(image.convert("RGB"))

    def make_box_key(
        self,
        video_id: str,
        timestamp: int,
        normalized_box: tuple[float, float, float, float],
    ) -> tuple[Any, ...]:
        rounded_box = tuple(
            round(coord, self.BOX_KEY_PRECISION) for coord in normalized_box
        )
        return (video_id, timestamp, *rounded_box)
