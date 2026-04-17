import av
import os
import time
import torch
import random
import numpy as np

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
    ):
        super().__init__()
        self.video_path = video_path
        self.num_workers = num_workers
        self.ext = ext
        self.clip_duration = clip_duration
        self.video_edge_time = video_edge_time

        self.is_train = is_train
        self.transform = transform

        self.video_file_path, self.class_to_idx = self.get_video_file_paths_and_labels()
        self.num_elements = len(self.video_file_path) // self.num_workers

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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        # シードをworker IDと時間で初期化
        random.seed(worker_id + int(self.current_time))

        start_idx = worker_id * self.num_elements
        end_idx = min(start_idx + self.num_elements, len(self.video_file_path))

        random.shuffle(self.video_file_path)

        for one_video_file_path in self.video_file_path[start_idx:end_idx]:
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
                    clip = []
                    clip_num += 1
                    frame_idx = []

                    yield subclip, label, frame_idx, {
                        "video_id": one_video_file_path,
                        "duration": duration,
                        "fps": fps,
                        "worker": worker_id,
                    }
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
