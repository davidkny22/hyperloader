"""Image-folder dataset and standard fine-tuning collation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .image_batches import ImageBatch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TrainingImageFolder(Dataset[tuple[torch.Tensor, int, str]]):
    """Apply standard augmentation and retain a stable source identity per sample."""

    def __init__(self, root: Path, *, resolution: int) -> None:
        if resolution <= 0:
            raise ValueError("image resolution must be positive")
        from torchvision.datasets import ImageFolder
        from torchvision.transforms import (
            Compose,
            Normalize,
            RandomHorizontalFlip,
            RandomResizedCrop,
            ToTensor,
        )

        transform = Compose(
            (
                RandomResizedCrop(resolution, antialias=True),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(IMAGENET_MEAN, IMAGENET_STD),
            )
        )
        self._root = root.resolve()
        self._dataset = ImageFolder(self._root, transform=transform)
        self._digests = tuple(
            hashlib.sha256(
                f"{Path(path).resolve().relative_to(self._root).as_posix()}\0{label}".encode()
            ).hexdigest()
            for path, label in self._dataset.samples
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image, label = self._dataset[index]
        return image, int(label), self._digests[index]


def collate_image_batch(rows: list[tuple[torch.Tensor, int, str]]) -> ImageBatch:
    """Stack augmented images and bind their stable source identities in order."""
    if not rows:
        raise ValueError("image collation requires at least one sample")
    images, labels, digests = zip(*rows, strict=True)
    chain = hashlib.sha256()
    for digest in digests:
        chain.update(bytes.fromhex(digest))
    batch = ImageBatch(
        torch.stack(images),
        torch.tensor(labels, dtype=torch.int64),
        chain.hexdigest(),
    )
    batch.validate()
    return batch
