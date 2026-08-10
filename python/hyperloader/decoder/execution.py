"""Execution binding for selected decoder providers."""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from hyperloader.stages import Decode, Pipeline

from .model import DecoderSelection


@dataclass(frozen=True, slots=True)
class PinnedDecoder:
    """Call one exact provider after validating its installed release."""

    backend: str
    version: str

    def __call__(self, value: Any) -> Any:
        """Decode one value through the selected backend."""
        function = _resolve_backend(self.backend, self.version)
        if self.backend.startswith("torchvision.io.decode_"):
            return function(_image_input(value))
        if self.backend == "torchcodec.decoders.AudioDecoder":
            decoder = function(value)
            return decoder.get_all_samples().data
        return function(value)


def bind_decoder_selections(
    dataset: Any, selections: tuple[DecoderSelection, ...]
) -> Any:
    """Return a pipeline whose opted-in Decode stages call their selected providers."""
    if not isinstance(dataset, Pipeline) or not selections:
        return dataset
    by_stage = {selection.stage: selection for selection in selections}
    stages = []
    for index, stage in enumerate(dataset.sample_stages):
        selection = by_stage.get(f"pipeline-decode-{index}")
        if (
            isinstance(stage, Decode)
            and selection is not None
            and selection.substituted
        ):
            stages.append(
                replace(
                    stage,
                    fn=PinnedDecoder(selection.backend, selection.version),
                )
            )
        else:
            stages.append(stage)
    return Pipeline(dataset.source, tuple(stages), dataset.collate_stage)


@lru_cache(maxsize=None)
def _resolve_backend(backend: str, version: str) -> Any:
    module_name, separator, attribute = backend.rpartition(".")
    if not separator:
        raise RuntimeError(f"decoder backend is not importable: {backend}")
    distribution = backend.partition(".")[0]
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"decoder backend {backend}@{version} is not installed"
        ) from error
    if installed.split("+", 1)[0] != version:
        raise RuntimeError(
            f"decoder backend {backend} requires {version}, found {installed}"
        )
    try:
        return getattr(importlib.import_module(module_name), attribute)
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            f"decoder backend is unavailable: {backend}@{version}"
        ) from error


def _image_input(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (str, Path)):
        from torchvision.io import read_file

        return read_file(str(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return torch.tensor(list(value), dtype=torch.uint8)
    raise TypeError(
        "image decoder input must be encoded bytes, a path, or a uint8 tensor"
    )
