from typing import Literal
from dataclasses import dataclass


SupportedModels = Literal[
    "memvit",
]


@dataclass
class ModelConfig:
    model_name: SupportedModels = "memvit"
    cfg: dict = None
    torch_home: str = None
