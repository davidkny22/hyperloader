"""Per-sample random generators for portable user stages."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, overload

from hyperloader import _hyperloader

_ACC_TORCH = 4
_ACC_NUMPY = 5
_ACC_RANDOM = 6
_UINT64_MASK = (1 << 64) - 1
SampleRng = tuple[int, int, int]
_KEY = 1
_COORD = 2
_ACTIVE_SAMPLE: contextvars.ContextVar[SampleRng | None] = contextvars.ContextVar(
    "hyperloader_active_sample", default=None
)
_LOCAL = threading.local()


class _GeneratorCache:
    """Retain one lazily re-keyed generator of each kind per execution thread."""

    def __init__(self) -> None:
        self.tokens: dict[str, SampleRng] = {}
        self.generators: dict[str, Any] = {}

    def resolve(self, kind: str, sample: SampleRng) -> Any:
        generator = self.generators.get(kind)
        if generator is None:
            generator = _build_generator(kind, sample)
            self.generators[kind] = generator
            self.tokens[kind] = sample
        elif self.tokens.get(kind) is not sample:
            _rekey_generator(kind, generator, sample)
            self.tokens[kind] = sample
        return generator


def _cache() -> _GeneratorCache:
    cache = getattr(_LOCAL, "generator_cache", None)
    if cache is None:
        cache = _GeneratorCache()
        _LOCAL.generator_cache = cache
    return cache


def _stream_seed(sample: SampleRng, stream_id: int) -> int:
    words = _hyperloader._rng_block_from_key(
        sample[_KEY], sample[_COORD], 0, stream_id
    )
    return words[0] | (words[1] << 32)


def _numpy_state(generator: Any, sample: SampleRng) -> None:
    stream_key = (sample[_KEY] ^ _splitmix64(_ACC_NUMPY)) & _UINT64_MASK
    state = generator.bit_generator.state
    state["state"]["counter"][:] = (sample[_COORD], 0, 0, 0)
    state["state"]["key"][:] = (stream_key, 0)
    state["buffer"].fill(0)
    state["buffer_pos"] = 4
    state["has_uint32"] = 0
    state["uinteger"] = 0
    generator.bit_generator.state = state


def _build_generator(kind: str, sample: SampleRng) -> Any:
    if kind == "torch":
        import torch

        return torch.Generator().manual_seed(_stream_seed(sample, _ACC_TORCH))
    if kind == "numpy":
        import numpy as np

        generator = np.random.Generator(np.random.Philox(key=0, counter=0))
        _numpy_state(generator, sample)
        return generator
    from .process.random_surface import PhiloxRandom

    generator = PhiloxRandom(stream_id=_ACC_RANDOM)
    generator.rekey(sample[_KEY], sample[_COORD])
    return generator


def _rekey_generator(kind: str, generator: Any, sample: SampleRng) -> None:
    if kind == "torch":
        generator.manual_seed(_stream_seed(sample, _ACC_TORCH))
    elif kind == "numpy":
        _numpy_state(generator, sample)
    else:
        generator.rekey(sample[_KEY], sample[_COORD])


def _splitmix64(value: int) -> int:
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return value


@overload
def rng(kind: Literal["torch"] = "torch") -> Any: ...


@overload
def rng(kind: Literal["numpy", "random"]) -> Any: ...


def rng(kind: str = "torch") -> Any:
    """Return the current stage's independent per-sample generator."""
    if kind not in {"torch", "numpy", "random"}:
        raise ValueError("rng kind must be torch, numpy, or random")
    sample = _ACTIVE_SAMPLE.get()
    if sample is None:
        raise RuntimeError(
            "hyperloader.rng() is available only while user code runs inside "
            "a native loader stage"
        )
    return _cache().resolve(kind, sample)


@contextmanager
def _user_code_context(sample: SampleRng) -> Iterator[None]:
    """Expose accessors only for the dynamic extent of one user-code stage."""
    token = _ACTIVE_SAMPLE.set(sample)
    try:
        yield
    finally:
        _ACTIVE_SAMPLE.reset(token)
