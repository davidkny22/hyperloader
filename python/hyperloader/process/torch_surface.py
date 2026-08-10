"""Lazy CPU-default-generator arming for seeded torch operations."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .sample_rng import CurrentSample, SampleRng


class TorchModuleSurface(TorchDispatchMode):
    """Seed the CPU default generator at the first seeded operation per sample."""

    def __init__(self, current: CurrentSample) -> None:
        super().__init__()
        self._current = current
        self._armed: SampleRng | None = None
        self._generator = torch.default_generator
        self.__enter__()

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        call_kwargs = {} if kwargs is None else kwargs
        if (
            torch.Tag.nondeterministic_seeded in func.tags
            and self._uses_default_generator(func, args, call_kwargs)
        ):
            self._ensure_armed()
        return func(*args, **call_kwargs)

    def _ensure_armed(self) -> None:
        sample = self._current.value
        if sample is not None and self._armed is not sample:
            self._generator.manual_seed(sample.torch_seed)
            self._armed = sample

    @staticmethod
    def _uses_default_generator(
        func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> bool:
        for index, argument in enumerate(func._schema.arguments):
            if argument.name != "generator":
                continue
            generator = kwargs.get("generator")
            if argument.name not in kwargs and index < len(args):
                generator = args[index]
            return generator is None
        return True

    def clear(self) -> None:
        """Remove this worker's persistent dispatch mode."""
        self.__exit__(None, None, None)
