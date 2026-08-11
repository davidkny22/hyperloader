"""Reusable pinned tensor buffers for refused source registrations."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class _LayoutPool:
    tensors: list[Any] = field(default_factory=list)


class PinnedTensorPool:
    """Copy tensor leaves once into reusable final pinned delivery buffers."""

    def __init__(self) -> None:
        self._layouts: dict[tuple[object, ...], _LayoutPool] = {}
        self.copied_bytes = 0

    def stage(self, value: Any) -> Any:
        """Stage every CPU tensor leaf while preserving the container structure."""
        return self._stage(value, ())

    def close(self) -> None:
        """Release cached pinned allocation ownership."""
        self._layouts.clear()

    def _stage(self, value: Any, path: tuple[object, ...]) -> Any:
        import torch

        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.is_pinned():
                return value
            return self._stage_tensor(value, path)
        if isinstance(value, dict):
            items = [
                (key, self._stage(item, (*path, key))) for key, item in value.items()
            ]
            try:
                return type(value)(items)
            except TypeError:
                result = value.copy()
                result.clear()
                result.update(items)
                return result
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(
                *(self._stage(item, (*path, index)) for index, item in enumerate(value))
            )
        if isinstance(value, tuple):
            values = tuple(
                self._stage(item, (*path, index)) for index, item in enumerate(value)
            )
            return values if type(value) is tuple else type(value)(values)
        if isinstance(value, list):
            return [
                self._stage(item, (*path, index)) for index, item in enumerate(value)
            ]
        return value

    def _stage_tensor(self, source: Any, path: tuple[object, ...]) -> Any:
        import torch

        key = (
            *path,
            source.dtype,
            tuple(source.shape),
            tuple(source.stride()),
        )
        layout = self._layouts.setdefault(key, _LayoutPool())
        target = None
        for candidate in layout.tensors:
            if sys.getrefcount(candidate) <= 3:
                target = candidate
                break
        if target is None:
            target = torch.empty_strided(
                tuple(source.shape),
                tuple(source.stride()),
                dtype=source.dtype,
                device="cpu",
                pin_memory=True,
            )
            layout.tensors.append(target)
        with torch.no_grad():
            target.copy_(source)
        self.copied_bytes += source.numel() * source.element_size()
        return target
