from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.optim import Optimizer

from .gradient_transforms import GradientTransform


class GradientTransformOptimizer(Optimizer):
    def __init__(
        self,
        inner_optimizer: Optimizer,
        gradient_transform: GradientTransform,
        optimizer_name: str,
    ):
        self.inner_optimizer = inner_optimizer
        self.gradient_transform = gradient_transform
        self.optimizer_name = optimizer_name

        super().__init__(inner_optimizer.param_groups, inner_optimizer.defaults)
        self.param_groups = inner_optimizer.param_groups
        self.defaults = inner_optimizer.defaults
        self.state = inner_optimizer.state

    def step(self, closure: Callable[[], float] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.gradient_transform.apply(self.param_groups)
        self.inner_optimizer.step()
        self.state = self.inner_optimizer.state
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "wrapper": {
                "optimizer_name": self.optimizer_name,
                "transform_name": self.gradient_transform.name,
            },
            "inner_optimizer": self.inner_optimizer.state_dict(),
            "gradient_transform": self.gradient_transform.state_dict(self.param_groups),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "inner_optimizer" not in state_dict:
            self.inner_optimizer.load_state_dict(state_dict)
            self.gradient_transform.reset_state()
            self._sync_from_inner_optimizer()
            return

        self.inner_optimizer.load_state_dict(state_dict["inner_optimizer"])
        self._sync_from_inner_optimizer()
        self.gradient_transform.load_state_dict(
            self.param_groups,
            state_dict.get("gradient_transform", {}),
        )

    def add_param_group(self, param_group: dict) -> None:
        self.inner_optimizer.add_param_group(param_group)
        self._sync_from_inner_optimizer()

    def _sync_from_inner_optimizer(self) -> None:
        self.param_groups = self.inner_optimizer.param_groups
        self.defaults = self.inner_optimizer.defaults
        self.state = self.inner_optimizer.state
