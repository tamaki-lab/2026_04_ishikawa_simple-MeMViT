from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

import torch


def get_sample_field(sample: Any, field_name: str, tuple_index: int):
    """
    SequentialSample dataclass と tuple の両方に対応するための補助関数
    """

    if hasattr(sample, field_name):
        return getattr(sample, field_name)

    if is_dataclass(sample) and hasattr(sample, field_name):
        return getattr(sample, field_name)

    return sample[tuple_index]


def collate_y(ys: list[Any]) -> Any:
    """
    y を可能な範囲で batch 化する．

    対応例:
        int label:
            [1, 2, 3] -> torch.LongTensor([1, 2, 3])

        torch.Tensor:
            [tensor(...), tensor(...)] -> torch.stack(...)

        dict:
            [{"verb": 1, "noun": 2}, ...]
            -> {"verb": tensor(...), "noun": tensor(...)}

        list[int]:
            [[1, 1, 2], [2, 2, 3]]
            -> torch.LongTensor([[1, 1, 2], [2, 2, 3]])

        複雑な形式:
            そのまま list で返す．
    """

    if len(ys) == 0:
        return ys

    first = ys[0]

    if all(isinstance(y, int) for y in ys):
        return torch.tensor(ys, dtype=torch.long)

    if all(isinstance(y, torch.Tensor) for y in ys):
        try:
            return torch.stack(ys, dim=0)
        except RuntimeError:
            return ys

    if all(isinstance(y, dict) for y in ys):
        keys = first.keys()

        if not all(y.keys() == keys for y in ys):
            return ys

        return {
            key: collate_y([y[key] for y in ys])
            for key in keys
        }

    if all(isinstance(y, list) for y in ys):
        if all(
            all(isinstance(value, int) for value in y)
            for y in ys
        ):
            lengths = [len(y) for y in ys]

            if len(set(lengths)) == 1:
                return torch.tensor(ys, dtype=torch.long)

        return ys

    return ys


def sequential_collate_fn(batch: list[Any]):
    """
    Sequential Dataset 用の汎用 collate_fn

    戻り値:
        xs:
            torch.Tensor
            shape: B x C x T x H x W

        ys:
            task によって異なる。
            単純分類なら torch.LongTensor。
            AVA のような detection 系なら dict/list のまま返す場合がある

        frame_indices:
            各 sample がどの frame から作られたか

        infos:
            video_id, fps, duration, worker などの補助情報
    """

    xs = torch.stack(
        [
            get_sample_field(sample, "x", 0)
            for sample in batch
        ],
        dim=0,
    )

    ys = collate_y(
        [
            get_sample_field(sample, "y", 1)
            for sample in batch
        ]
    )

    frame_indices = [
        get_sample_field(sample, "frame_indices", 2)
        for sample in batch
    ]

    infos = [
        get_sample_field(sample, "info", 3)
        for sample in batch
    ]

    return xs, ys, frame_indices, infos


def classification_collate_fn(batch: list[Any]):
    """
    動画分類・行動認識など、y が単一の class index である場合の collate_fn

    VideoFolderSequentialDataset や，
    EpicKitchenSequentialDataset の action_recognition / action_anticipation で
    y が int の場合に使える．
    """

    xs = torch.stack(
        [
            get_sample_field(sample, "x", 0)
            for sample in batch
        ],
        dim=0,
    )

    ys = torch.tensor(
        [
            get_sample_field(sample, "y", 1)
            for sample in batch
        ],
        dtype=torch.long,
    )

    frame_indices = [
        get_sample_field(sample, "frame_indices", 2)
        for sample in batch
    ]

    infos = [
        get_sample_field(sample, "info", 3)
        for sample in batch
    ]

    return xs, ys, frame_indices, infos
