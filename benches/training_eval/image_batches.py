"""Image-classification batches for live fine-tuning cells."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImageBatch:
    """One image tensor, label vector, and precomputed source digest."""

    images: torch.Tensor
    labels: torch.Tensor
    digest: str

    def validate(self) -> None:
        """Require aligned image and label batch axes plus one SHA-256 digest."""
        if self.images.ndim != 4 or self.labels.ndim != 1:
            raise ValueError("image batches require NCHW images and one label axis")
        if self.images.shape[0] <= 0 or self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("image and label batch axes must align")
        try:
            decoded = bytes.fromhex(self.digest)
        except ValueError as error:
            raise ValueError("image batch digest must contain 32 bytes") from error
        if len(decoded) != 32:
            raise ValueError("image batch digest must contain 32 bytes")

    @property
    def samples(self) -> int:
        """Return the number of images in this batch."""
        return int(self.images.shape[0])
