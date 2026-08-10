"""Worker-local Philox binding for the NumPy random module."""

from __future__ import annotations

from typing import Any

from .sample_rng import COORD, KEY, CurrentSample, SampleRng

_STATE_NUMPY = 8
_UINT64_MASK = (1 << 64) - 1
_GENERATOR_METHODS = (
    "beta",
    "binomial",
    "bytes",
    "chisquare",
    "choice",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "geometric",
    "gumbel",
    "hypergeometric",
    "laplace",
    "logistic",
    "lognormal",
    "logseries",
    "multinomial",
    "multivariate_normal",
    "negative_binomial",
    "noncentral_chisquare",
    "noncentral_f",
    "normal",
    "pareto",
    "permutation",
    "poisson",
    "power",
    "random",
    "rayleigh",
    "shuffle",
    "standard_cauchy",
    "standard_exponential",
    "standard_gamma",
    "standard_normal",
    "standard_t",
    "triangular",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
    "zipf",
)
_COMPATIBILITY_METHODS = (
    "get_state",
    "rand",
    "randint",
    "randn",
    "random_integers",
    "random_sample",
    "ranf",
    "sample",
    "seed",
    "set_state",
)


def _splitmix64(value: int) -> int:
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return value


class NumpyModuleSurface:
    """Bind NumPy's module aliases to one persistent Philox Generator."""

    def __init__(self, current: CurrentSample) -> None:
        import numpy as np

        self._np = np
        self._current = current
        self._armed: SampleRng | None = None
        self._bit_generator = np.random.Philox(key=0, counter=0)
        self.generator = np.random.Generator(self._bit_generator)
        self._state = self._bit_generator.state
        self._counter = self._state["state"]["counter"]
        self._key = self._state["state"]["key"]
        self._buffer = self._state["buffer"]
        self._legacy_seed = 0
        self._legacy_state: Any = None
        self._legacy_dirty = True
        names = (*_GENERATOR_METHODS, *_COMPATIBILITY_METHODS)
        self._prior = {name: getattr(np.random, name) for name in names}
        for name in _GENERATOR_METHODS:
            setattr(np.random, name, self._bound_generator_method(name))
        np.random.random_sample = self._bound_generator_method("random")
        np.random.sample = self._bound_generator_method("random")
        np.random.ranf = self._bound_generator_method("random")
        np.random.rand = self.rand
        np.random.randn = self.randn
        np.random.randint = self.randint
        np.random.random_integers = self.random_integers
        np.random.seed = self.seed
        np.random.get_state = self.get_state
        np.random.set_state = self.set_state

    def rekey(self, key: int, coord: int) -> None:
        """Reset the persistent Generator to one exact sample stream."""
        stream_key = (key ^ _splitmix64(_STATE_NUMPY)) & _UINT64_MASK
        self._counter[:] = (coord, 0, 0, 0)
        self._key[:] = (stream_key, 0)
        self._buffer.fill(0)
        self._state["buffer_pos"] = 4
        self._state["has_uint32"] = 0
        self._state["uinteger"] = 0
        self._bit_generator.state = self._state
        self._legacy_seed = stream_key & ((1 << 32) - 1)
        self._legacy_dirty = True

    def _ensure_armed(self) -> None:
        sample = self._current.value
        if sample is not None and self._armed is not sample:
            self.rekey(sample[KEY], sample[COORD])
            self._armed = sample

    def _bound_generator_method(self, name: str) -> Any:
        method = getattr(self.generator, name)

        def bound(*args: Any, **kwargs: Any) -> Any:
            self._ensure_armed()
            return method(*args, **kwargs)

        bound.__name__ = name
        bound.__self__ = self.generator  # type: ignore[attr-defined]
        return bound

    def rand(self, *dims: int) -> Any:
        """Map the legacy variadic shape helper to Generator.random."""
        self._ensure_armed()
        return self.generator.random(size=dims or None)

    def randn(self, *dims: int) -> Any:
        """Map the legacy variadic shape helper to Generator.standard_normal."""
        self._ensure_armed()
        return self.generator.standard_normal(size=dims or None)

    def randint(
        self,
        low: int,
        high: int | None = None,
        size: Any = None,
        dtype: Any = int,
    ) -> Any:
        """Map legacy exclusive-high integers to Generator.integers."""
        self._ensure_armed()
        return self.generator.integers(low, high=high, size=size, dtype=dtype)

    def random_integers(
        self, low: int, high: int | None = None, size: Any = None
    ) -> Any:
        """Map legacy inclusive-high integers to Generator.integers."""
        self._ensure_armed()
        if high is None:
            low, high = 1, low
        return self.generator.integers(low, high=high, size=size, endpoint=True)

    def seed(self, seed: Any = None) -> None:
        """Seed the lazily constructed legacy RandomState."""
        legacy = self._legacy()
        legacy.seed(seed)
        self._legacy_dirty = False

    def get_state(self) -> Any:
        """Return the lazily constructed legacy RandomState state."""
        return self._legacy().get_state()

    def set_state(self, state: Any) -> None:
        """Restore the lazily constructed legacy RandomState state."""
        self._legacy().set_state(state)
        self._legacy_dirty = False

    def _legacy(self) -> Any:
        self._ensure_armed()
        if self._legacy_state is None:
            self._legacy_state = self._np.random.RandomState()
        if self._legacy_dirty:
            self._legacy_state.seed(self._legacy_seed)
            self._legacy_dirty = False
        return self._legacy_state

    def clear(self) -> None:
        """Restore the NumPy module callables captured before worker binding."""
        for name, value in self._prior.items():
            setattr(self._np.random, name, value)
