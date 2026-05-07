from collections.abc import Iterator

from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from .optimizers import SupportedOptimizers
from .optimizers import configure_optimizer as _configure_optimizer


def configure_optimizer(
    optimizer_name: SupportedOptimizers,
    model_params: Iterator[Parameter],
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
    orthogonal_beta: float = 0.9,
    orthogonal_eps: float = 1e-12,
) -> Optimizer:
    """Build an optimizer from flat CLI-style arguments."""
    return _configure_optimizer(
        optimizer_name=optimizer_name,
        model_params=model_params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        orthogonal_beta=orthogonal_beta,
        orthogonal_eps=orthogonal_eps,
    )
