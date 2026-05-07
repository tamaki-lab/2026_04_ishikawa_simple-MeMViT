from collections.abc import Callable, Iterator

from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW, Optimizer, SGD

from .config import OptimizerConfig

BaseOptimizerBuilder = Callable[[Iterator[Parameter], OptimizerConfig], Optimizer]


def _build_sgd(model_params: Iterator[Parameter], config: OptimizerConfig) -> Optimizer:
    return SGD(
        model_params,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def _build_adam(model_params: Iterator[Parameter], config: OptimizerConfig) -> Optimizer:
    return Adam(
        model_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


def _build_adamw(model_params: Iterator[Parameter], config: OptimizerConfig) -> Optimizer:
    return AdamW(
        model_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


BASE_OPTIMIZER_REGISTRY: dict[str, BaseOptimizerBuilder] = {
    "SGD": _build_sgd,
    "Adam": _build_adam,
    "AdamW": _build_adamw,
}
