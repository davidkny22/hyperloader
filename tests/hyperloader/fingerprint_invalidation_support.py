"""Reusable datasets for installed fingerprint invalidation evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torchvision.datasets.folder import DatasetFolder


class FileDataset(DatasetFolder):
    """Expose an ordered file inventory without scanning the dataset root."""

    def __init__(self, root: Path, names: tuple[str, ...]) -> None:
        self.root = str(root)
        self.samples = [(str(root / name), index) for index, name in enumerate(names)]
        self.loader = bytes
        self.transform = None
        self.target_transform = None


def collate_tuple(values: list[Any]) -> tuple[Any, ...]:
    """Provide one installed callable identity."""
    return tuple(values)


def collate_list(values: list[Any]) -> list[Any]:
    """Provide a distinct installed callable identity."""
    return values


class OffsetSampler:
    """Expose one result-observable sampler setting."""

    def __init__(self, length: int, offset: int) -> None:
        self.length = length
        self.offset = offset

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        return iter(range(self.offset, self.offset + self.length))


class IterableValues:
    """Provide an iterable fingerprint surface without consuming it."""

    def __iter__(self):
        return iter((1, 2, 3))


def init_left(_worker: int) -> None:
    """Provide one iterable sharding initializer identity."""


def init_right(_worker: int) -> None:
    """Provide a distinct iterable sharding initializer identity."""
