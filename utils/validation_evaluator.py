from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch

from .accuracy import compute_topk_accuracy


class ValidationEvaluator(ABC):
    def reset(self) -> None:
        pass

    @abstractmethod
    def should_accumulate(self, infos: list[dict[str, Any]]) -> bool:
        pass

    @abstractmethod
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        infos: list[dict[str, Any]],
    ) -> None:
        pass

    @abstractmethod
    def compute(self) -> dict[str, float]:
        pass


class GroupedTopKAccuracyEvaluator(ValidationEvaluator):
    def __init__(self, topk: tuple[int, ...] = (1, 5)):
        self.topk = topk
        self.reset()

    def reset(self) -> None:
        self.group_logits: dict[str, list[torch.Tensor]] = {}
        self.group_labels: dict[str, int] = {}
        self.group_strategies: dict[str, str] = {}

    def should_accumulate(self, infos: list[dict[str, Any]]) -> bool:
        return (
            len(infos) > 0
            and infos[0].get("evaluation_mode") == "grouped_topk"
        )

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        infos: list[dict[str, Any]],
    ) -> None:
        if len(infos) != 1:
            raise ValueError(
                "Grouped sequential validation currently requires batch_size=1 "
                "so sequence-aware aggregation stays aligned with model state."
            )

        info = infos[0]
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
        self.group_labels[group_id] = int(labels[0].item())

        sample_logits = logits[0].detach().cpu()
        if aggregation_strategy == "mean":
            self.group_logits.setdefault(group_id, []).append(sample_logits)
            return

        if info.get("sequence_end", info.get("is_last")):
            self.group_logits[group_id] = [sample_logits]

    def compute(self) -> dict[str, float]:
        aggregated_logits = []
        aggregated_labels = []

        for group_id, logits_list in self.group_logits.items():
            if not logits_list:
                continue

            strategy = self.group_strategies[group_id]
            if strategy == "mean":
                pooled_logits = torch.stack(logits_list, dim=0).mean(dim=0)
            else:
                pooled_logits = logits_list[-1]

            aggregated_logits.append(pooled_logits)
            aggregated_labels.append(self.group_labels[group_id])

        if not aggregated_logits:
            return {}

        logits_tensor = torch.stack(aggregated_logits, dim=0)
        labels_tensor = torch.tensor(aggregated_labels, dtype=torch.long)
        topk_values = compute_topk_accuracy(logits_tensor, labels_tensor, topk=self.topk)

        return {
            f"val_top{k}": value for k, value in zip(self.topk, topk_values)
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
        logits: torch.Tensor,
        labels: torch.Tensor,
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


def configure_validation_evaluator() -> ValidationEvaluator:
    return ValidationEvaluatorDispatcher(
        evaluators=[
            GroupedTopKAccuracyEvaluator(),
        ]
    )
