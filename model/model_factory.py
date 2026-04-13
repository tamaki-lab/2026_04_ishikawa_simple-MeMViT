from model import (
    ClassificationBaseModel,
    ModelConfig,
    MemViT,
)


def configure_model(
        model_info: ModelConfig
) -> ClassificationBaseModel:
    """model factory

    model_info:
        model_info (ModelInfo): information for model

    Raises:
        ValueError: invalide model name given by command line

    Returns:
        ClassificationBaseModel: model
    """

    if model_info.model_name == 'memvit':
        model = MemViT(model_info.cfg)  # type: ignore[assignment]

    else:
        raise ValueError('invalid model_info.model_name')

    return model
