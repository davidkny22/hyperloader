"""Recognizable vision-model construction for image-folder anchors."""

from __future__ import annotations

from torch import nn


def build_resnet18(*, classes: int) -> nn.Module:
    """Construct an uninitialized torchvision ResNet-18 classification head."""
    if classes <= 1:
        raise ValueError("vision fine-tuning requires at least two classes")
    from torchvision.models import resnet18

    return resnet18(weights=None, num_classes=classes)
