import av
import os
import time
import torch
import random
import numpy as np


from pathlib import Path
from torch.utils.data import IterableDataset
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


def data_loader(
        clip_duration,
        video_edge_time,
        batch_size,
        num_workers,
        ext,
        train_dir,
        val_dir,
):

    train_dataset = SequentialVideoDataset(
        clip_duration,
        video_edge_time,
        num_workers,
        ext,
        train_dir,
        is_train=True,
        transform=transform_video(is_train=True),
    )

    val_dataset = SequentialVideoDataset(
        clip_duration,
        video_edge_time,
        num_workers,
        ext,
        val_dir,
        is_train=False,
        transform=transform_video(is_train=False),
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


def transform_video(is_train):
    transform_list = [
        UniformTemporalSubsample(16),
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
    new_batch = [[], [], [], []]
    for i in range(len(batch)):
        new_batch[0].append(batch[i][0])
        new_batch[1].append(batch[i][1])
        new_batch[2].append(batch[i][2])
        new_batch[3].append(batch[i][3])
    return new_batch


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
