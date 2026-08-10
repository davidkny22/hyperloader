"""Spawn-safe pipeline fixtures for decoder pin evidence."""

from __future__ import annotations

import torch

from hyperloader import Collate, Decode, Source, pipeline


def forbidden_decoder(_value: torch.Tensor) -> torch.Tensor:
    """Fail if a substituted stage invokes its declared refuge callable."""
    raise AssertionError("the selected provider was not installed into execution")


def stack_images(values: list[torch.Tensor]) -> torch.Tensor:
    """Collate decoded images without changing their storage bits."""
    return torch.stack(values)


def image_pipeline(encoded: list[torch.Tensor]):
    """Build an opted-in PNG pipeline over encoded tensor values."""
    return pipeline(
        Source(encoded, output_type=torch.Tensor),
        Decode(
            forbidden_decoder,
            input_type=torch.Tensor,
            output_type=torch.Tensor,
            codec="png",
            substitute=True,
        ),
        Collate(stack_images, input_type=torch.Tensor, output_type=torch.Tensor),
    )
