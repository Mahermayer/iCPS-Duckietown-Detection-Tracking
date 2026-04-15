from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from duckietown_seg.models.configurable_unet import ConfigurableUNet
from duckietown_seg.models.vanilla_unet import VanillaUNet


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "vanilla_unet": VanillaUNet,
    "configurable_unet": ConfigurableUNet,
}


def create_model(model_name: str, **kwargs: Any) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[model_name](**kwargs)


__all__ = ["MODEL_REGISTRY", "create_model", "VanillaUNet", "ConfigurableUNet"]

