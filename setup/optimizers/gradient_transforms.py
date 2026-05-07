from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Parameter


class GradientTransform(ABC):
    name = "gradient_transform"

    @abstractmethod
    def apply(self, param_groups: list[dict]) -> None:
        """Transform gradients in-place before the optimizer step."""

    @abstractmethod
    def state_dict(self, param_groups: list[dict]) -> dict:
        """Serialize transform state."""

    @abstractmethod
    def load_state_dict(self, param_groups: list[dict], state_dict: dict) -> None:
        """Restore transform state."""

    @abstractmethod
    def reset_state(self) -> None:
        """Clear transform state."""


class OrthogonalGradientTransform(GradientTransform):
    name = "orthogonal"

    def __init__(self, beta: float = 0.9, eps: float = 1e-12):
        self.beta = beta
        self.eps = eps
        self._state: dict[int, dict[str, Tensor]] = {}

    def apply(self, param_groups: list[dict]) -> None:
        for param_index, param in enumerate(self._iter_params(param_groups)):
            grad = param.grad
            if grad is None or grad.is_sparse:
                continue

            raw_grad = grad.detach()
            raw_grad_fp32 = raw_grad.to(torch.float32)
            state = self._state.get(param_index)
            ema = None if state is None else state.get("ema")

            if ema is not None:
                denom = torch.dot(ema.reshape(-1), ema.reshape(-1))
                if denom.item() > self.eps:
                    coeff = torch.dot(raw_grad_fp32.reshape(-1), ema.reshape(-1)) / denom
                    orthogonal_grad = raw_grad_fp32 - coeff * ema
                    grad.copy_(orthogonal_grad.to(dtype=grad.dtype))

            updated_ema = raw_grad_fp32.mul(1.0 - self.beta)
            if ema is not None:
                updated_ema = updated_ema + ema * self.beta

            self._state[param_index] = {"ema": updated_ema.clone()}

    def state_dict(self, param_groups: list[dict]) -> dict:
        param_count = sum(1 for _ in self._iter_params(param_groups))
        state = {}
        for param_index in range(param_count):
            if param_index not in self._state:
                continue
            state[param_index] = {
                name: value.detach().clone()
                for name, value in self._state[param_index].items()
            }

        return {
            "name": self.name,
            "beta": self.beta,
            "eps": self.eps,
            "state": state,
        }

    def load_state_dict(self, param_groups: list[dict], state_dict: dict) -> None:
        params = list(self._iter_params(param_groups))
        loaded_state: dict[int, dict[str, Tensor]] = {}

        for key, value in state_dict.get("state", {}).items():
            param_index = int(key)
            if param_index >= len(params):
                raise ValueError("transform state_dict does not match current parameters")

            param = params[param_index]
            loaded_state[param_index] = {
                name: tensor.detach().clone().to(device=param.device, dtype=torch.float32)
                for name, tensor in value.items()
            }

        self._state = loaded_state

    def reset_state(self) -> None:
        self._state = {}

    @staticmethod
    def _iter_params(param_groups: list[dict]):
        for param_group in param_groups:
            for param in param_group["params"]:
                if isinstance(param, Parameter):
                    yield param
