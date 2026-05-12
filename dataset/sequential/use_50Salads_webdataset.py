import torch
import torch.autograd
import webdataset as wds
import numpy as np
import io
import av
import math
import argparse

from pathlib import Path
from functools import partial
from dataset.my_transforms import transform_factory
from utils.utils import info_from_json


def short_side(h, w, size):
    if min(h, w) <= size:
        return h, w  # do not resize for smaller frame size

    # https://github.com/facebookresearch/pytorchvideo/blob/a77729992bcf1e43bf5fa507c8dc4517b3d7bc4c/pytorchvideo/transforms/functional.py#L118
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    return new_h, new_w


def my_collate_50salads(batch):
    return batch[0], torch.utils.data.default_collate(batch[1]), batch[2], batch[3]


# @profile
def video_yield_decorder(input_tuble, transform, stride):
    clip_len = 16
    count = 0

    for sample in input_tuble:  # 1動画づつのループ
        video_bytes = sample["video.bin"]
        info_dict = sample["stats.json"]
        video_id = sample["__key__"]

        action_id = info_dict["ac_cate_ID"]
        action_id = [i for i in action_id if i != 1]
        start_frame_idx = info_dict["ac_start_frame_ID"][: len(action_id)]
        end_frame_idx = info_dict["ac_end_frame_ID"][: len(action_id)]
        n_actions = len(action_id)

        container = av.open(io.BytesIO(video_bytes))
        video_stream_id = 0
        stream = container.streams.video[video_stream_id]
        stream.thread_type = "AUTO"

        # new_h, new_w = short_side(
        #     stream.codec_context.height, stream.codec_context.width, 320
        # )

        num_make_clip = 0
        clip = []
        clip_id = []
        frame_idx = []
        action_index = 0
        is_clip_initialized = False

        for i, frame in enumerate(container.decode(stream)):
            if i % stride != 0:
                continue

            if not is_clip_initialized:
                # frame数がactionの先頭に行くまでcontinue
                if i < start_frame_idx[action_index]:
                    continue
                # frameの右端がactionの終わりを越えたらaction_indexを1増やす
                if end_frame_idx[action_index] < i + clip_len - 1:
                    action_index += 1
                    # アクション数がn_actionsを越えたら終わり
                    if action_index >= n_actions:
                        break
                    else:
                        continue
                clip_id = action_id[action_index]
                clip = []
                frame_idx = []
                is_clip_initialized = True

            img = frame.to_ndarray(format="rgb24")

            clip.append(img)
            frame_idx.append(i)

            if len(clip) == clip_len:
                if frame_idx[0] >= start_frame_idx[action_index]:
                    if frame_idx[15] <= end_frame_idx[action_index]:
                        count += 1
                clip = np.stack(clip, 0)  # THWC
                clip = np.transpose(clip, (3, 0, 1, 2))  # --> CTHW
                clip = torch.from_numpy(clip)
                num_make_clip += 1
                if transform is not None:
                    clip = transform(clip)
                print(clip_id, video_id)
                yield clip, clip_id, frame_idx, {
                    "video_id": video_id,
                    "ac_start_frame_ID": start_frame_idx[action_index],
                    "ac_end_frame_ID": end_frame_idx[action_index],
                    "worker": torch.utils.data.get_worker_info().id,
                }
                is_clip_initialized = False


def make_50salads(
    dataset, shards_url, dataset_size, transform, clip_shuffle_buffer_size, stride
):
    dataset = wds.WebDataset(shards_url)
    dataset = dataset.decode()

    decode_frame = partial(video_yield_decorder, transform=transform, stride=stride)
    dataset = dataset.compose(decode_frame)
    # dataset = dataset.shuffle(clip_shuffle_buffer_size)
    dataset = dataset.with_length(dataset_size)
    return dataset


def get_sequential_50salads_dataset(command_line_args: argparse.Namespace):
    args = command_line_args
    train_path = args.wds_50Salads_train_path
    val_path = args.wds_50Salads_val_path

    train_shards_path = [
        str(path) for path in Path(train_path).glob("*.tar") if not path.is_dir()
    ]
    val_shards_path = [
        str(path) for path in Path(val_path).glob("*.tar") if not path.is_dir()
    ]

    train_dataset_size, num_classes = info_from_json(train_path)
    val_dataset_size, num_classes = info_from_json(val_path)

    train_dataset = make_50salads(
        dataset=args.dataset_name,
        shards_url=train_shards_path,
        dataset_size=train_dataset_size,
        transform=transform_factory(is_train=True),
        clip_shuffle_buffer_size=10,
        stride=args.stride,
    )
    val_dataset = make_50salads(
        dataset=args.dataset_name,
        shards_url=val_shards_path,
        dataset_size=val_dataset_size,
        transform=transform_factory(is_train=False),
        clip_shuffle_buffer_size=10,
        stride=args.stride,
    )

    train_dataset = train_dataset.batched(args.batch_size)
    val_dataset = val_dataset.batched(args.batch_size)

    train_loader = wds.WebLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=None,
        collate_fn=my_collate_50salads,
    )
    val_loader = wds.WebLoader(
        val_dataset,
        num_workers=args.num_workers,
        batch_size=None,
        collate_fn=my_collate_50salads,
    )

    train_loader = train_loader.with_length(train_dataset_size // args.batch_size)
    val_loader = val_loader.with_length(val_dataset_size // args.batch_size)

    return train_loader, val_loader, num_classes
