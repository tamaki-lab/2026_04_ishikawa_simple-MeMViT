from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random

import av
import numpy as np
import torch
from torch.utils.data import IterableDataset


@dataclass
class VideoItem:
    path: Path
    video_id: str
    meta: dict


@dataclass
class SequentialSample:
    x: torch.Tensor
    y: Any
    frame_indices: list[int]
    info: dict


@dataclass
class LogicalSampleSpec:
    sample_id: str
    video_id: str
    start_frame: int
    end_frame: int


@dataclass
class SubSampleSpec:
    sample_id: str
    sub_id: int
    num_subsamples: int
    frame_indices: list[int]
    valid_mask: list[bool]
    is_first: bool
    is_last: bool


class BaseSequentialVideoDataset(IterableDataset, ABC):
    def __init__(
        self,
        video_path,
        clip_duration,
        video_edge_time,
        ext,
        is_train,
        transform,
        frames_per_clip=16,
        batch_size=1,
        task=None,
        shuffle=True,
    ):
        super().__init__()

        self.video_path = Path(video_path)
        self.clip_duration = clip_duration
        self.video_edge_time = video_edge_time
        self.ext = ext
        self.is_train = is_train
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.batch_size = batch_size
        self.task = task
        self.shuffle = shuffle

        if self.frames_per_clip <= 0:
            raise ValueError("frames_per_clip must be a positive integer")

        self.video_items = self.load_video_items()
        self.class_to_idx = self.build_class_to_idx()
        self.num_classes = len(self.class_to_idx)

    @abstractmethod
    def load_video_items(self) -> list[VideoItem]:
        pass

    @abstractmethod
    def build_class_to_idx(self) -> dict:
        pass

    @abstractmethod
    def iter_logical_samples(
        self,
        video_item: VideoItem,
        fps: float,
    ) -> list[LogicalSampleSpec] | tuple[LogicalSampleSpec, ...]:
        pass

    @abstractmethod
    def make_target(self, logical_sample: LogicalSampleSpec) -> Any:
        pass

    def make_target_for_subsample(
        self,
        logical_sample: LogicalSampleSpec,
        sub_sample: SubSampleSpec,
        logical_target: Any,
    ) -> Any:
        return logical_target

    def __iter__(self):
        worker_id, num_workers = self.get_worker_info()

        video_items = list(self.video_items)
        if self.shuffle and self.is_train:
            random.shuffle(video_items)

        for video_item in self.split_by_worker(video_items, worker_id, num_workers):
            try:
                container, stream, fps = self.open_video(video_item.path)
            except Exception as error:
                print(f"Failed to open {video_item.path}: {error}")
                continue

            try:
                logical_samples = self.iter_logical_samples(
                    video_item=video_item,
                    fps=fps,
                )

                for logical_sample in logical_samples:
                    sub_samples = self.make_subsample_specs(logical_sample)
                    if not sub_samples:
                        continue

                    logical_target = self.make_target(logical_sample)

                    for sub_sample in sub_samples:
                        try:
                            clip_frames = self.read_subsample_frames(
                                container=container,
                                stream=stream,
                                fps=fps,
                                sub_sample=sub_sample,
                            )
                        except Exception as error:
                            print(
                                "Failed to read subsample "
                                f"{sub_sample.sample_id}:{sub_sample.sub_id} "
                                f"from {video_item.path}: {error}"
                            )
                            break

                        target = self.make_target_for_subsample(
                            logical_sample=logical_sample,
                            sub_sample=sub_sample,
                            logical_target=logical_target,
                        )
                        x = self.build_x(clip_frames)
                        info = self.build_info(
                            video_item=video_item,
                            logical_sample=logical_sample,
                            sub_sample=sub_sample,
                            fps=fps,
                            worker_id=worker_id,
                        )

                        yield self.make_sample(
                            x=x,
                            y=target,
                            frame_indices=sub_sample.frame_indices,
                            info=info,
                        )
            finally:
                container.close()

    def get_worker_info(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            return 0, 1

        return worker_info.id, worker_info.num_workers

    def split_by_worker(self, items, worker_id, num_workers):
        items_per_worker = (len(items) + num_workers - 1) // num_workers
        start_idx = worker_id * items_per_worker
        end_idx = min(start_idx + items_per_worker, len(items))
        return items[start_idx:end_idx]

    def open_video(self, path):
        container = av.open(str(path))

        if len(container.streams.video) == 0:
            raise RuntimeError(f"{path} has no video stream")

        stream = container.streams.video[0]
        fps = float(stream.base_rate)

        return container, stream, fps

    def make_subsample_specs(
        self,
        logical_sample: LogicalSampleSpec,
    ) -> list[SubSampleSpec]:
        if logical_sample.end_frame <= logical_sample.start_frame:
            return []

        all_frame_indices = list(
            range(logical_sample.start_frame, logical_sample.end_frame)
        )
        chunks: list[list[int]] = []
        masks: list[list[bool]] = []

        for start_idx in range(0, len(all_frame_indices), self.frames_per_clip):
            chunk = all_frame_indices[start_idx:start_idx + self.frames_per_clip]
            valid_len = len(chunk)
            if valid_len == 0:
                continue

            if valid_len < self.frames_per_clip:
                chunk = chunk + [chunk[-1]] * (self.frames_per_clip - valid_len)

            valid_mask = [True] * valid_len + [False] * (self.frames_per_clip - valid_len)
            chunks.append(chunk)
            masks.append(valid_mask)

        num_subsamples = len(chunks)
        return [
            SubSampleSpec(
                sample_id=logical_sample.sample_id,
                sub_id=sub_id,
                num_subsamples=num_subsamples,
                frame_indices=chunks[sub_id],
                valid_mask=masks[sub_id],
                is_first=(sub_id == 0),
                is_last=(sub_id == num_subsamples - 1),
            )
            for sub_id in range(num_subsamples)
        ]

    def read_subsample_frames(
        self,
        container,
        stream,
        fps: float,
        sub_sample: SubSampleSpec,
    ) -> list[np.ndarray]:
        frame_map = self.read_frame_map(
            container=container,
            stream=stream,
            fps=fps,
            frame_indices=sub_sample.frame_indices,
        )

        return [frame_map[frame_index] for frame_index in sub_sample.frame_indices]

    def read_frame_map(
        self,
        container,
        stream,
        fps: float,
        frame_indices: list[int],
    ) -> dict[int, np.ndarray]:
        unique_frame_indices = sorted(set(frame_indices))
        if not unique_frame_indices:
            return {}

        frame_map = self._read_frame_map_with_seek(
            container=container,
            stream=stream,
            fps=fps,
            frame_indices=unique_frame_indices,
        )

        if len(frame_map) == len(unique_frame_indices):
            return frame_map

        frame_map = self._read_frame_map_from_start(
            container=container,
            stream=stream,
            frame_indices=unique_frame_indices,
        )

        missing_indices = [
            frame_index
            for frame_index in unique_frame_indices
            if frame_index not in frame_map
        ]
        if missing_indices:
            raise RuntimeError(f"Missing frames: {missing_indices}")

        return frame_map

    def _read_frame_map_with_seek(
        self,
        container,
        stream,
        fps: float,
        frame_indices: list[int],
    ) -> dict[int, np.ndarray]:
        if not frame_indices or fps <= 0 or stream.time_base is None:
            return {}

        start_frame = frame_indices[0]
        end_frame = frame_indices[-1]

        try:
            pts = int((start_frame / fps) / float(stream.time_base))
            container.seek(
                offset=max(pts, 0),
                any_frame=False,
                backward=True,
                stream=stream,
            )
        except Exception:
            return {}

        needed = set(frame_indices)
        frame_map: dict[int, np.ndarray] = {}

        for frame in container.decode(stream):
            frame_number = self.estimate_frame_number(frame, fps)
            if frame_number is None:
                continue

            if frame_number < start_frame:
                continue

            if frame_number in needed and frame_number not in frame_map:
                frame_map[frame_number] = frame.to_ndarray(format="rgb24")

                if len(frame_map) == len(needed):
                    break

            if frame_number > end_frame:
                break

        return frame_map

    def _read_frame_map_from_start(
        self,
        container,
        stream,
        frame_indices: list[int],
    ) -> dict[int, np.ndarray]:
        if not frame_indices:
            return {}

        container.seek(0, backward=True, stream=stream)

        needed = set(frame_indices)
        end_frame = frame_indices[-1]
        frame_map: dict[int, np.ndarray] = {}

        for frame_number, frame in enumerate(container.decode(stream)):
            if frame_number > end_frame:
                break

            if frame_number in needed:
                frame_map[frame_number] = frame.to_ndarray(format="rgb24")

                if len(frame_map) == len(needed):
                    break

        return frame_map

    def estimate_frame_number(self, frame, fps: float) -> int | None:
        if frame.time is None or fps <= 0:
            return None

        return int(round(frame.time * fps))

    def build_x(self, clip_frames: list[np.ndarray]):
        clip = np.stack(clip_frames, axis=0)
        clip = np.transpose(clip, (3, 0, 1, 2))
        clip = torch.from_numpy(clip)

        if self.transform is None:
            return clip

        return self.transform(clip)

    def build_info(
        self,
        video_item: VideoItem,
        logical_sample: LogicalSampleSpec,
        sub_sample: SubSampleSpec,
        fps: float,
        worker_id: int,
    ) -> dict[str, Any]:
        info = {
            "video_id": video_item.video_id,
            "task": self.task,
            "sample_id": logical_sample.sample_id,
            "sub_id": sub_sample.sub_id,
            "num_subsamples": sub_sample.num_subsamples,
            "is_first": sub_sample.is_first,
            "is_last": sub_sample.is_last,
            "sequence_id": logical_sample.sample_id,
            "sequence_index": sub_sample.sub_id,
            "clip_index": sub_sample.sub_id,
            "sequence_length": sub_sample.num_subsamples,
            "sequence_start": sub_sample.is_first,
            "sequence_end": sub_sample.is_last,
            "logical_start_frame": logical_sample.start_frame,
            "logical_end_frame": logical_sample.end_frame,
            "valid_mask": sub_sample.valid_mask,
            "source_video_path": video_item.path,
            "fps": fps,
            "worker": worker_id,
        }
        info.update(
            self.build_evaluation_info(
                video_item=video_item,
                logical_sample=logical_sample,
                sub_sample=sub_sample,
                fps=fps,
                worker_id=worker_id,
            )
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
        return {}

    def make_sample(self, x, y, frame_indices, info):
        return SequentialSample(
            x=x,
            y=y,
            frame_indices=frame_indices,
            info=info,
        )

    # ファイル数を示す
    def __len__(self):
        return len(self.video_items)
