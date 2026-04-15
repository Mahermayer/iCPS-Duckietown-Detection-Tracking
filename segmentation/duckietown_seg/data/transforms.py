from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class SegmentationResize:
    """Resize image-mask pairs with mask-safe interpolation."""

    size: tuple[int, int]

    def __init__(self, size: Sequence[int]) -> None:
        if len(size) != 2:
            raise ValueError("image_size must contain exactly two integers [height, width].")
        object.__setattr__(self, "size", (int(size[0]), int(size[1])))

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        target_size = (self.size[1], self.size[0])
        image = image.resize(target_size, resample=Image.BILINEAR)
        mask = mask.resize(target_size, resample=Image.NEAREST)
        return image, mask


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3], got {array.shape}.")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale mask with shape [H, W], got {array.shape}.")
    return torch.from_numpy(array).long()

