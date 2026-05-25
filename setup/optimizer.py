from typing import Iterator, Literal

from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim import SGD, Adam, AdamW

SupportedOptimizers = Literal["SGD", "Adam", "AdamW", "OrthogonalSGD", "OrthogonalAdamW"]


def configure_optimizer(
    optimizer_name: SupportedOptimizers,
    model_params: Iterator[Parameter],
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
    orthogonal_beta: float = 0.9,
    orthogonal_eps: float = 1e-12,
) -> Optimizer:
    """optimizer factory

    Args:
        optimizer_name (SupportedOptimizers): optimizer name (str).
            ["SGD", "Adam", "AdamW", "OrthogonalSGD", "OrthogonalAdamW"]
        model_params (Iterator[Parameter]): model parameters.
            Typically "model.parameters()"
        lr (float): learning rate.
        weight_decay (float): weight decay
        momentum (float, optional): momentum. Defaults to 0.9.
        orthogonal_beta (float, optional): placeholder for orthogonal
            optimizers' EMA coefficient. Accepted for API compatibility.
        orthogonal_eps (float, optional): placeholder for orthogonal
            optimizers' projection epsilon. Accepted for API compatibility.

    Raises:
        ValueError: invalide optimizer name given by command line

    Returns:
        Optimizer: optimizer
    """
    del orthogonal_beta, orthogonal_eps

    if optimizer_name == "SGD":
        return SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    if optimizer_name == "Adam":
        return Adam(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "AdamW":
        return AdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name in {"OrthogonalSGD", "OrthogonalAdamW"}:
        raise NotImplementedError(
            f"{optimizer_name} is accepted by the CLI but is not implemented in setup/optimizer.py yet."
        )

    raise ValueError("invalid optimizer_name")
