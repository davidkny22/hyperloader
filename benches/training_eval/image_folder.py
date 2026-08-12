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
        digests = []
        for path, label in self._dataset.samples:
            resolved = Path(path).resolve()
            digest = hashlib.sha256()
            digest.update(
                f"{resolved.relative_to(self._root).as_posix()}\0{label}\0".encode()
            )
            digest.update(hashlib.sha256(resolved.read_bytes()).digest())
            digests.append(digest.hexdigest())
        self._digests = tuple(digests)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image, label = self._dataset[index]
        return image, int(label), self._digests[index]

    @property
    def class_count(self) -> int:
        """Return the number of image-folder classes."""
        return len(self._dataset.classes)

    def identity_for_rows(self, rows: int) -> str:
        """Digest the exact ordered source subset used by one point."""
        if not 0 < rows <= len(self._digests):
            raise ValueError("image source row count is outside the dataset")
        digest = hashlib.sha256()
        for source_digest in self._digests[:rows]:
            digest.update(bytes.fromhex(source_digest))
        return digest.hexdigest()


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
