from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch

from .accuracy import (
    build_framewise_valid_mask,
    compute_topk_accuracy,
    flatten_framewise_logits_and_labels,
)

ClassificationLogits = torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
ClassificationLabels = torch.Tensor


class ValidationEvaluator(ABC):
    def reset(self) -> None:
        pass

    @abstractmethod
    def should_accumulate(self, infos: list[dict[str, Any]]) -> bool:
        pass

    @abstractmethod
    def update(
        self,
        logits: ClassificationLogits,
        labels: ClassificationLabels,
        infos: list[dict[str, Any]],
    ) -> None:
        pass

    @abstractmethod
    def compute(self) -> dict[str, float]:
        pass


class GroupedTopKAccuracyEvaluator(ValidationEvaluator):
    def __init__(
        self,
        topk: tuple[int, ...] = (1, 5),
        head_names: tuple[str, ...] | None = None,
    ):
        self.topk = topk
        self.head_names = head_names
        self.reset()

    def reset(self) -> None:
        self.group_logits: dict[
            str,
            list[torch.Tensor | tuple[torch.Tensor, ...]],
        ] = {}
        self.group_labels: dict[str, int | tuple[int, ...]] = {}
        self.group_strategies: dict[str, str] = {}

    def should_accumulate(self, infos: list[dict[str, Any]]) -> bool:
        return (
            len(infos) > 0
            and infos[0].get("evaluation_mode") == "grouped_topk"
        )

    def update(
        self,
        logits: ClassificationLogits,
        labels: ClassificationLabels,
        infos: list[dict[str, Any]],
    ) -> None:
        sample_logits_list, sample_labels_list = self._normalize_batch(
            logits=logits,
            labels=labels,
            infos=infos,
        )

        for sample_logits, sample_label, info in zip(
            sample_logits_list,
            sample_labels_list,
            infos,
        ):
            group_id = str(info["group_id"])
            aggregation_strategy = str(info.get("aggregation_strategy", "mean"))
            if aggregation_strategy not in {"mean", "last"}:
                raise ValueError(
                    "aggregation_strategy must be 'mean' or 'last', "
                    f"but got {aggregation_strategy}"
                )

            previous_strategy = self.group_strategies.get(group_id)
            if previous_strategy is not None and previous_strategy != aggregation_strategy:
                raise ValueError(
                    f"Inconsistent aggregation strategy for group_id={group_id}: "
                    f"{previous_strategy} vs {aggregation_strategy}"
                )

            self.group_strategies[group_id] = aggregation_strategy
            previous_label = self.group_labels.get(group_id)
            if previous_label is not None and previous_label != sample_label:
                raise ValueError(
                    f"Inconsistent labels for group_id={group_id}: "
                    f"{previous_label} vs {sample_label}"
                )
            self.group_labels[group_id] = sample_label

            if aggregation_strategy == "mean":
                self.group_logits.setdefault(group_id, []).append(sample_logits)
                continue

            if info.get("sequence_end", info.get("is_last")):
                self.group_logits[group_id] = [sample_logits]

    def compute(self) -> dict[str, float]:
        aggregated_logits = []
        aggregated_labels = []

        for group_id, logits_list in self.group_logits.items():
            if not logits_list:
                continue

            strategy = self.group_strategies[group_id]
            pooled_logits = self._pool_group_logits(logits_list, strategy)

            aggregated_logits.append(pooled_logits)
            aggregated_labels.append(self.group_labels[group_id])

        if not aggregated_logits:
            return {}

        first_logits = aggregated_logits[0]
        if isinstance(first_logits, tuple):
            return self._compute_multi_head_metrics(
                aggregated_logits=aggregated_logits,
                aggregated_labels=aggregated_labels,
            )

        logits_tensor = torch.stack(aggregated_logits, dim=0)
        labels_tensor = torch.tensor(aggregated_labels, dtype=torch.long)
        topk_values = compute_topk_accuracy(logits_tensor, labels_tensor, topk=self.topk)

        return {
            f"val_top{k}": value for k, value in zip(self.topk, topk_values)
        }

    def _normalize_batch(
        self,
        logits: ClassificationLogits,
        labels: ClassificationLabels,
        infos: list[dict[str, Any]],
    ) -> tuple[
        list[torch.Tensor | tuple[torch.Tensor, ...]],
        list[int | tuple[int, ...]],
    ]:
        batch_size = len(infos)
        if isinstance(logits, torch.Tensor):
            if logits.shape[0] != batch_size or labels.shape[0] != batch_size:
                raise ValueError(
                    "GroupedTopKAccuracyEvaluator received inconsistent batch sizes: "
                    f"logits={logits.shape[0]}, labels={labels.shape[0]}, infos={batch_size}"
                )
            return (
                [sample_logits.detach().cpu() for sample_logits in logits],
                [int(sample_label.item()) for sample_label in labels],
            )

        num_heads = len(logits)
        if labels.ndim != 2 or labels.shape != (batch_size, num_heads):
            raise ValueError(
                "Expected multi-head labels shaped [batch, num_heads], "
                f"but got {tuple(labels.shape)} for {num_heads} heads."
            )

        for head_logits in logits:
            if head_logits.shape[0] != batch_size:
                raise ValueError(
                    "GroupedTopKAccuracyEvaluator received inconsistent multi-head "
                    f"batch sizes: logits={head_logits.shape[0]}, infos={batch_size}"
                )

        sample_logits_list = [
            tuple(head_logits[index].detach().cpu() for head_logits in logits)
            for index in range(batch_size)
        ]
        sample_labels_list = [
            tuple(int(labels[index, head_idx].item()) for head_idx in range(num_heads))
            for index in range(batch_size)
        ]
        return sample_logits_list, sample_labels_list

    def _pool_group_logits(
        self,
        logits_list: list[torch.Tensor | tuple[torch.Tensor, ...]],
        strategy: str,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        last_logits = logits_list[-1]
        if strategy == "last":
            return last_logits

        if isinstance(last_logits, tuple):
            return tuple(
                torch.stack(
                    [sample_logits[head_idx] for sample_logits in logits_list],
                    dim=0,
                ).mean(dim=0)
                for head_idx in range(len(last_logits))
            )

        return torch.stack(logits_list, dim=0).mean(dim=0)

    def _compute_multi_head_metrics(
        self,
        aggregated_logits: list[tuple[torch.Tensor, ...]],
        aggregated_labels: list[int | tuple[int, ...]],
    ) -> dict[str, float]:
        num_heads = len(aggregated_logits[0])
        head_metric_names = self._get_multi_head_metric_names(num_heads)
        metrics: dict[str, float] = {}
        mean_metrics: dict[int, list[float]] = {k: [] for k in self.topk}

        for head_idx, head_metric_name in enumerate(head_metric_names):
            logits_tensor = torch.stack(
                [sample_logits[head_idx] for sample_logits in aggregated_logits],
                dim=0,
            )
            labels_tensor = torch.tensor(
                [sample_label[head_idx] for sample_label in aggregated_labels],
                dtype=torch.long,
            )
            topk_values = compute_topk_accuracy(
                logits_tensor,
                labels_tensor,
                topk=self.topk,
            )
            for k, value in zip(self.topk, topk_values):
                metrics[f"val_top{k}_{head_metric_name}"] = value
                mean_metrics[k].append(value)

        for k, values in mean_metrics.items():
            metrics[f"val_top{k}"] = sum(values) / len(values)

        return metrics

    def _get_multi_head_metric_names(self, num_heads: int) -> list[str]:
        if self.head_names is not None and len(self.head_names) == num_heads:
            return list(self.head_names)
        return [f"head{head_idx}" for head_idx in range(1, num_heads + 1)]


class FramewiseTopKAccuracyEvaluator(ValidationEvaluator):
    def __init__(self, topk: tuple[int, ...] = (1, 5)):
        self.topk = topk
        self.reset()

    def reset(self) -> None:
        self.correct_by_k = {k: 0.0 for k in self.topk}
        self.total = 0

    def should_accumulate(self, infos: list[dict[str, Any]]) -> bool:
        return (
            len(infos) > 0
            and infos[0].get("evaluation_mode") == "framewise"
        )

    def update(
        self,
        logits: ClassificationLogits,
        labels: ClassificationLabels,
        infos: list[dict[str, Any]],
    ) -> None:
        if not isinstance(logits, torch.Tensor):
            raise ValueError("FramewiseTopKAccuracyEvaluator expects tensor logits.")

        valid_mask = build_framewise_valid_mask(
            infos=infos,
            temporal_dim=labels.shape[1],
            device=logits.device,
        )
        flat_logits, flat_labels = flatten_framewise_logits_and_labels(
            logits=logits,
            labels=labels,
            valid_mask=valid_mask,
        )
        if flat_labels.numel() == 0:
            return

        maxk = min(max(self.topk), flat_logits.shape[1])
        _, pred = flat_logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(flat_labels.view(1, -1).expand_as(pred))

        for k in self.topk:
            effective_k = min(k, flat_logits.shape[1])
            correct_k = correct[:effective_k].reshape(-1).float().sum().item()
            self.correct_by_k[k] += correct_k

        self.total += flat_labels.numel()

    def compute(self) -> dict[str, float]:
        if self.total == 0:
            return {}

        return {
            f"val_top{k}": correct * 100.0 / self.total
            for k, correct in self.correct_by_k.items()
        }


class ValidationEvaluatorDispatcher(ValidationEvaluator):
    def __init__(self, evaluators: Iterable[ValidationEvaluator]):
        self.evaluators = list(evaluators)

    def reset(self) -> None:
        for evaluator in self.evaluators:
            evaluator.reset()

    def should_accumulate(self, infos: list[dict[str, Any]]) -> bool:
        return self._select_evaluator(infos) is not None

    def update(
        self,
        logits: ClassificationLogits,
        labels: ClassificationLabels,
        infos: list[dict[str, Any]],
    ) -> None:
        evaluator = self._select_evaluator(infos)
        if evaluator is None:
            raise ValueError("No validation evaluator matched the given batch infos.")
        evaluator.update(logits=logits, labels=labels, infos=infos)

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for evaluator in self.evaluators:
            metrics.update(evaluator.compute())
        return metrics

    def _select_evaluator(
        self,
        infos: list[dict[str, Any]],
    ) -> ValidationEvaluator | None:
        for evaluator in self.evaluators:
            if evaluator.should_accumulate(infos):
                return evaluator
        return None


def configure_validation_evaluator(
    head_names: tuple[str, ...] | None = None,
) -> ValidationEvaluator:
    return ValidationEvaluatorDispatcher(
        evaluators=[
            GroupedTopKAccuracyEvaluator(head_names=head_names),
            FramewiseTopKAccuracyEvaluator(),
        ]
    )
