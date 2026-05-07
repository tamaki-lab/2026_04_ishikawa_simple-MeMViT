from dataclasses import dataclass
from typing import Literal


SupportedOptimizers = Literal[
    "SGD",
    "Adam",
    "AdamW",
    "OrthogonalSGD",
    "OrthogonalAdamW",
]


@dataclass(frozen=True)
class OptimizerConfig:
    optimizer_name: SupportedOptimizers
    lr: float
    weight_decay: float
    momentum: float = 0.9
    orthogonal_beta: float = 0.9
    orthogonal_eps: float = 1e-12
