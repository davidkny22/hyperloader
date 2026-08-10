"""Worker-local Philox core for the Python random module."""

from __future__ import annotations

import random
from typing import Any

from hyperloader import _hyperloader

from .sample_rng import COORD, KEY, CurrentSample, SampleRng

_STATE_RANDOM = 7
_UINT32_BITS = 32
_UINT32_MASK = (1 << _UINT32_BITS) - 1
_DRAW_LIMIT = 1 << 32
_STATE_MARKER = "hyperloader-philox"
_MODULE_METHODS = (
    "seed",
    "random",
    "uniform",
    "triangular",
    "randint",
    "choice",
    "randrange",
    "sample",
    "shuffle",
    "choices",
    "normalvariate",
    "lognormvariate",
    "expovariate",
    "vonmisesvariate",
    "gammavariate",
    "gauss",
    "betavariate",
    "binomialvariate",
    "paretovariate",
    "weibullvariate",
    "getstate",
    "setstate",
    "getrandbits",
    "randbytes",
)


class PhiloxRandom(random.Random):
    """Implement the standard Random surface over one engine Philox stream."""

    def __init__(self, current: CurrentSample | None = None) -> None:
        self._current = current
        self._armed: SampleRng | None = None
        self._key = 0
        self._coord = 0
        self._next_block = 0
        self._words: tuple[int, ...] = ()
        self._word_offset = 0
        self.gauss_next: float | None = None

    def rekey(self, key: int, coord: int) -> None:
        """Start the exact stream assigned to one sample coordinate."""
        self._key = key
        self._coord = coord
        self._next_block = 0
        self._words = ()
        self._word_offset = 0
        self.gauss_next = None

    def _ensure_armed(self) -> None:
        if self._current is None:
            return
        sample = self._current.value
        if sample is not None and self._armed is not sample:
            self.rekey(sample[KEY], sample[COORD])
            self._armed = sample

    def seed(self, a: Any = None, version: int = 2) -> None:
        """Start an explicitly requested standalone stream."""
        source = random.Random()
        source.seed(a, version=version)
        self.rekey(source.getrandbits(64), 0)
        self._armed = None if self._current is None else self._current.value

    def getstate(self) -> tuple[Any, ...]:
        """Return a round-trippable state for save and restore."""
        self._ensure_armed()
        return (
            _STATE_MARKER,
            self._key,
            self._coord,
            self._next_block,
            self._words,
            self._word_offset,
            self.gauss_next,
        )

    def setstate(self, state: tuple[Any, ...]) -> None:
        """Restore a state produced by this Philox-backed instance."""
        if len(state) != 7 or state[0] != _STATE_MARKER:
            raise ValueError("state does not describe a hyperloader Philox stream")
        _, key, coord, next_block, words, word_offset, gauss_next = state
        if not 0 <= next_block <= _DRAW_LIMIT:
            raise ValueError("random draw index is outside the Philox counter domain")
        if not 0 <= word_offset <= len(words) <= 4:
            raise ValueError("random state contains an invalid buffered-word range")
        self._key = int(key)
        self._coord = int(coord)
        self._next_block = int(next_block)
        self._words = tuple(int(word) & _UINT32_MASK for word in words)
        self._word_offset = int(word_offset)
        self.gauss_next = gauss_next
        self._armed = None if self._current is None else self._current.value

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Arm before consulting the inherited Gaussian value cache."""
        self._ensure_armed()
        return super().gauss(mu, sigma)

    def random(self) -> float:
        """Return one 53-bit uniform value from two successive stream words."""
        upper = self._next_word() >> 5
        lower = self._next_word() >> 6
        return (upper * 67_108_864.0 + lower) * (2.0**-53)

    def getrandbits(self, k: int) -> int:
        """Return an integer assembled from successive stream words."""
        if k < 0:
            raise ValueError("number of bits must be non-negative")
        if k == 0:
            return 0
        result = 0
        produced = 0
        while produced < k:
            width = min(_UINT32_BITS, k - produced)
            word = self._next_word()
            if width < _UINT32_BITS:
                word >>= _UINT32_BITS - width
            result |= word << produced
            produced += width
        return result

    def _next_word(self) -> int:
        self._ensure_armed()
        if self._word_offset == len(self._words):
            if self._next_block == _DRAW_LIMIT:
                raise OverflowError("random draw index exhausted the Philox counter domain")
            self._words = _hyperloader._rng_block_from_key(
                self._key,
                self._coord,
                self._next_block,
                _STATE_RANDOM,
            )
            self._next_block += 1
            self._word_offset = 0
        word = self._words[self._word_offset]
        self._word_offset += 1
        return word


class RandomModuleSurface:
    """Bind and restore the module-level Python random callables."""

    def __init__(self, current: CurrentSample) -> None:
        self.generator = PhiloxRandom(current)
        self._prior = {
            name: getattr(random, name) for name in _MODULE_METHODS if hasattr(random, name)
        }
        self._prior_instance = random._inst
        random._inst = self.generator
        for name in self._prior:
            setattr(random, name, getattr(self.generator, name))

    def clear(self) -> None:
        """Restore the module callables captured before worker binding."""
        random._inst = self._prior_instance
        for name, value in self._prior.items():
            setattr(random, name, value)
