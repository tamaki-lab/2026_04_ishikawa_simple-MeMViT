import av
import os
import time
import torch
import random
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import IterableDataset


class SequentialVideoDataset(IterableDataset):
    def __init__(
        self,
        clip_duration,
        video_edge_time,
        num_workers,
        ext,
        video_path,
        is_train,
        transform,
        label_mode="video",
        annotation_path=None,
        background_label="background",
        frames_per_clip=16,
        batch_size=1,
        epic_label_type="verb",
    ):
        super().__init__()
        if label_mode not in ["video", "frame"]:
            raise ValueError("label_mode must be 'video' or 'frame'")
        if epic_label_type not in ["verb", "noun"]:
            raise ValueError("epic_label_type must be 'verb' or 'noun'")

        self.video_path = video_path
        self.num_workers = num_workers
        self.ext = ext
        self.clip_duration = clip_duration
        self.video_edge_time = video_edge_time

        self.is_train = is_train
        self.transform = transform
        self.label_mode = label_mode
        self.annotation_path = annotation_path
        self.background_label = background_label
        self.frames_per_clip = frames_per_clip
        self.batch_size = batch_size
        self.epic_label_type = epic_label_type

        if self.label_mode == "frame":
            self.annotations = self.parse_epic_annotations()
            self.video_file_path, self.class_to_idx = self.get_frame_video_file_paths_and_labels()
        else:
            self.annotations = {}
            self.video_file_path, self.class_to_idx = self.get_video_file_paths_and_labels()

        self.num_elements = len(self.video_file_path)

        self.num_classes = len(self.class_to_idx)

        self.current_time = time.time()

    def get_video_file_paths_and_labels(self):
        ext = self.ext
        video_file_paths = [
            path
            for path in Path(self.video_path).glob(f"**/{ext}")
            if not path.is_dir()
        ]
        video_file_paths = sorted(video_file_paths)

        class_list = sorted(
            entry.name for entry in os.scandir(self.video_path) if entry.is_dir()
        )
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_list)}

        return video_file_paths, class_to_idx

    def parse_epic_annotations(self):
        if self.annotation_path is None:
            raise ValueError("annotation_path is required when label_mode='frame'")

        label_column = f"{self.epic_label_type}_class"
        annotations = pd.read_csv(self.annotation_path)
        required_columns = {"video_id", "start_frame", "stop_frame", label_column}
        missing_columns = required_columns - set(annotations.columns)
        if missing_columns:
            raise ValueError(
                f"EPIC annotation CSV is missing columns: {sorted(missing_columns)}"
            )

        video_annotations = {}
        for _, row in annotations.iterrows():
            video_id = str(row["video_id"])
            video_annotations.setdefault(video_id, []).append({
                "start_frame": int(row["start_frame"]),
                "stop_frame": int(row["stop_frame"]),
                "label": str(row[label_column]),
            })

        for intervals in video_annotations.values():
            intervals.sort(key=lambda item: item["start_frame"])

        return video_annotations

    def get_frame_video_file_paths_and_labels(self):
        video_file_paths = [
            path
            for path in Path(self.video_path).glob(f"**/{self.ext}")
            if not path.is_dir()
        ]
        video_file_paths = sorted(video_file_paths)

        class_list = sorted({
            interval["label"]
            for intervals in self.annotations.values()
            for interval in intervals
            if interval["label"] is not None
        })

        if self.background_label not in class_list:
            class_list.append(self.background_label)

        class_to_idx = {label: i for i, label in enumerate(class_list)}
        return video_file_paths, class_to_idx

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # シードをworker IDと時間で初期化
        random.seed(worker_id + int(self.current_time))

        video_file_paths = list(self.video_file_path)
        random.shuffle(video_file_paths)

        videos_per_worker = (len(video_file_paths) + num_workers - 1) // num_workers
        start_idx = worker_id * videos_per_worker
        end_idx = min(start_idx + videos_per_worker, len(video_file_paths))

        for one_video_file_path in video_file_paths[start_idx:end_idx]:
            try:
                container = av.open(str(one_video_file_path))
            except Exception as e:
                print(e)
                continue
            if len(container.streams.video) == 0:
                print(f'{one_video_file_path.name} has no video streams. Skipping.')
                continue

            container, stream, duration, fps = self.get_trimmed_info(one_video_file_path)
            if duration is None or fps is None:
                print(f"Skipping file {one_video_file_path} due to missing duration or fps.")
                continue

            if self.label_mode == "frame":
                yield from self.generate_frame_label_clips(
                    one_video_file_path,
                    container,
                    stream,
                    fps,
                    worker_id,
                )
                continue

            category_name = one_video_file_path.parent.name
            if category_name not in self.class_to_idx:
                print(
                    f"Skipping file {one_video_file_path}: category '{category_name}' not in class_to_idx. Available classes: {list(self.class_to_idx.keys())}")
                continue
            label = self.class_to_idx[category_name]
            clip_len = int(duration) * int(fps)
            clip_num = 1

            clip = []
            frame_idx = []

            for i, frame in enumerate(container.decode(stream)):
                frame_idx.append(i)
                img = frame.to_ndarray(format="rgb24")
                clip.append(img)
                if i > clip_len:
                    break

                if frame.time < self.clip_duration * clip_num:
                    continue
                else:
                    clip = np.stack(clip, 0)  # THWC
                    clip = np.transpose(clip, (3, 0, 1, 2))  # --> CTHW
                    clip = torch.from_numpy(clip)
                    subclip = self.transform(clip)

                    yield subclip, label, frame_idx, {
                        "video_id": one_video_file_path,
                        "duration": duration,
                        "fps": fps,
                        "worker": worker_id,
                    }

                    clip = []
                    clip_num += 1
                    frame_idx = []
                if self.video_edge_time > (float(duration) - frame.time):
                    break

            if len(clip) > 0:
                clip = np.stack(clip, 0)  # THWC
                clip = np.transpose(clip, (3, 0, 1, 2))  # --> CTHW
                clip = torch.from_numpy(clip)

                while clip.shape[1] < 16:
                    clip = torch.cat((clip, clip[:, :16 - clip.shape[1], :, :]), dim=1)

                subclip = self.transform(clip)

                yield subclip, label, frame_idx, {
                    "video_id": one_video_file_path,
                    "duration": duration,
                    "fps": fps,
                    "worker": worker_id,
                }

    def get_annotation_video_id(self, one_video_file_path):
        stem = one_video_file_path.stem
        if stem in self.annotations:
            return stem

        split_stem = stem.split("-", 1)
        if len(split_stem) == 2 and split_stem[1] in self.annotations:
            return split_stem[1]

        annotation_id = stem.rsplit("-", 1)[0]
        if annotation_id in self.annotations:
            return annotation_id

        return stem

    def get_frame_label(self, video_id, frame_number):
        label_name = self.background_label
        for interval in self.annotations.get(video_id, []):
            if frame_number < interval["start_frame"]:
                break
            if interval["start_frame"] <= frame_number <= interval["stop_frame"]:
                label_name = interval["label"]
                break
        return self.class_to_idx.get(label_name, self.class_to_idx[self.background_label])

    def generate_frame_label_clips(self, one_video_file_path, container, stream, fps, worker_id):
        video_id = self.get_annotation_video_id(one_video_file_path)
        clip_len = self.frames_per_clip
        total_clip_len = clip_len * self.batch_size
        total_frames = stream.frames

        if total_frames == 0:
            intervals = self.annotations.get(video_id, [])
            total_frames = max((interval["stop_frame"] for interval in intervals), default=0)

        current_frame = 0
        while current_frame < total_frames:
            time_base = stream.time_base
            pts = int((current_frame / fps) / time_base)
            container.seek(
                offset=pts,
                any_frame=False,
                backward=True,
                stream=stream,
            )

            clip = []
            frame_idx = []
            action_labels = []

            for i, frame in enumerate(container.decode(video=0)):
                frame_number = current_frame + i
                img = frame.to_ndarray(format="rgb24")
                clip.append(img)
                frame_idx.append(frame_number)
                action_labels.append(self.get_frame_label(video_id, frame_number))

                if len(clip) == clip_len:
                    yield self.build_subclip(clip), action_labels, frame_idx, {
                        "video_id": video_id,
                        "source_video_id": one_video_file_path,
                        "fps": fps,
                        "worker": worker_id,
                        "label_mode": self.label_mode,
                    }
                    clip = []
                    frame_idx = []
                    action_labels = []

                if frame_number + 1 >= current_frame + total_clip_len:
                    break

            current_frame += total_clip_len

            if len(clip) > 0:
                while len(clip) < clip_len:
                    clip.append(clip[-1])
                    action_labels.append(action_labels[-1])

                yield self.build_subclip(clip), action_labels, frame_idx, {
                    "video_id": video_id,
                    "source_video_id": one_video_file_path,
                    "fps": fps,
                    "worker": worker_id,
                    "label_mode": self.label_mode,
                }

        container.close()

    def build_subclip(self, clip):
        clip = np.stack(clip, 0)  # THWC
        clip = np.transpose(clip, (3, 0, 1, 2))  # --> CTHW
        clip = torch.from_numpy(clip)
        return self.transform(clip)

    def get_trimmed_info(self, one_video_file_path):
        container = av.open(str(one_video_file_path))
        video_stream_id = 0
        stream = container.streams.video[video_stream_id]
        duration = float(container.duration) / av.time_base
        fps = float(stream.base_rate)
        return container, stream, duration, fps

    def __getitem__(self, index):
        raise NotImplementedError("error")

    def __len__(self):
        return len(self.video_file_path)
