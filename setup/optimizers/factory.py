from collections.abc import Iterator

from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from .base import BASE_OPTIMIZER_REGISTRY
from .config import OptimizerConfig, SupportedOptimizers
from .gradient_transforms import OrthogonalGradientTransform
from .wrapper import GradientTransformOptimizer

OPTIMIZER_SPECS: dict[SupportedOptimizers, tuple[str, str | None]] = {
    "SGD": ("SGD", None),
    "Adam": ("Adam", None),
    "AdamW": ("AdamW", None),
    "OrthogonalSGD": ("SGD", "orthogonal"),
    "OrthogonalAdamW": ("AdamW", "orthogonal"),
}


def build_optimizer(
    model_params: Iterator[Parameter],
    config: OptimizerConfig,
) -> Optimizer:
    if config.optimizer_name not in OPTIMIZER_SPECS:
        raise ValueError("invalid optimizer_name")

    base_optimizer_name, gradient_transform_name = OPTIMIZER_SPECS[config.optimizer_name]
    base_optimizer = BASE_OPTIMIZER_REGISTRY[base_optimizer_name](model_params, config)

    if gradient_transform_name is None:
        return base_optimizer

    if gradient_transform_name == "orthogonal":
        return GradientTransformOptimizer(
            inner_optimizer=base_optimizer,
            gradient_transform=OrthogonalGradientTransform(
                beta=config.orthogonal_beta,
                eps=config.orthogonal_eps,
            ),
            optimizer_name=config.optimizer_name,
        )

    raise ValueError("invalid gradient_transform_name")


def configure_optimizer(
    optimizer_name: SupportedOptimizers,
    model_params: Iterator[Parameter],
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
    orthogonal_beta: float = 0.9,
    orthogonal_eps: float = 1e-12,
) -> Optimizer:
    config = OptimizerConfig(
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        orthogonal_beta=orthogonal_beta,
        orthogonal_eps=orthogonal_eps,
    )
    return build_optimizer(model_params=model_params, config=config)
