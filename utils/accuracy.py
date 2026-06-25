from typing import Tuple

import torch


def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: Tuple[int, ...] = (1,)
) -> Tuple[float, ...]:
    """Computes the accuracy over top-k predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411

    Args:
        logits (torch.Tensor): model logits of the batch.
            The shape is (B, L) for batchsize B and number of labels L
        labels (torch.Tensor): labels of the batch
            The shape is (B, )
        topk (tuple of int, optional):
            k for computing top-k accuracy. Defaults to (1,).
                topk=(1,) returns (top1,)
                topk=(1,5) returns (top1, top5)

    Returns:
        Tuple[float]: top1 accuracy, or list of top-k accuracy values
    """
    assert topk[0] == 1, "topk[0] should be top1"
    assert len(topk) >= 1

    with torch.no_grad():
        num_classes = logits.shape[1]
        maxk = min(max(topk), num_classes)
        batch_size = labels.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            effective_k = min(k, num_classes)
            correct_k = correct[:effective_k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item() * 100.0 / batch_size)

        return tuple(res)


def build_framewise_valid_mask(
    infos: list[dict],
    temporal_dim: int,
    device: torch.device,
) -> torch.Tensor:
    masks = []
    for info in infos:
        valid_mask = info.get("valid_mask")
        if valid_mask is None:
            mask = torch.ones(temporal_dim, dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
            if mask.numel() != temporal_dim:
                raise ValueError(
                    "valid_mask length must match the temporal dimension, "
                    f"but got {mask.numel()} for T={temporal_dim}."
                )
        masks.append(mask)
    return torch.stack(masks, dim=0)


def flatten_framewise_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 3:
        raise ValueError(
            f"Expected framewise logits shaped [batch, time, classes], but got {tuple(logits.shape)}."
        )
    if labels.ndim != 2:
        raise ValueError(
            f"Expected framewise labels shaped [batch, time], but got {tuple(labels.shape)}."
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "Framewise logits and labels must share batch/time dimensions, "
            f"but got logits={tuple(logits.shape)} and labels={tuple(labels.shape)}."
        )

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)

    if valid_mask is None:
        return flat_logits, flat_labels

    if valid_mask.shape != labels.shape:
        raise ValueError(
            "valid_mask must have the same shape as labels, "
            f"but got valid_mask={tuple(valid_mask.shape)} and labels={tuple(labels.shape)}."
        )

    flat_mask = valid_mask.reshape(-1).to(device=logits.device, dtype=torch.bool)
    return flat_logits[flat_mask], flat_labels[flat_mask]
