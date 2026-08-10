"""Lazy torch worker identity for the current sample."""

from __future__ import annotations

from typing import Any

_hyperloader_worker_info_factory: Any = None
_worker_info: Any = None


def _lazy_get_worker_info() -> Any:
    factory = _hyperloader_worker_info_factory
    return _worker_info if factory is None else factory()


class WorkerInfoContext:
    """Create one immutable WorkerInfo only when user code requests it."""

    def __init__(self, worker_id: int, worker_count: int, dataset: Any) -> None:
        from torch.utils.data._utils import worker as worker_module

        self._worker_module = worker_module
        self._worker_info_type = worker_module.WorkerInfo
        self._worker_id = worker_id
        self._worker_count = worker_count
        self._dataset = dataset
        self._seed: int | None = None
        self._current: Any = None
        self._factory = self._materialize
        self._getter = worker_module.get_worker_info
        self._original_code = self._getter.__code__
        self._prior_factory = getattr(
            worker_module, "_hyperloader_worker_info_factory", _MISSING
        )
        worker_module._hyperloader_worker_info_factory = self._factory
        self._getter.__code__ = _lazy_get_worker_info.__code__
        worker_module._worker_info = None

    def begin_sample(self, seed: int | None) -> None:
        """Set the identity inputs and discard any prior sample's object."""
        self._seed = seed
        self._current = None

    def _materialize(self) -> Any:
        if self._current is None:
            self._current = self._worker_info_type(
                id=self._worker_id,
                num_workers=self._worker_count,
                seed=self._seed,
                dataset=self._dataset,
            )
        return self._current

    def clear(self) -> None:
        """Restore torch's getter and release its worker-global references."""
        self._getter.__code__ = self._original_code
        if self._prior_factory is _MISSING:
            delattr(self._worker_module, "_hyperloader_worker_info_factory")
        else:
            self._worker_module._hyperloader_worker_info_factory = self._prior_factory
        self._worker_module._worker_info = None
        self._current = None


_MISSING = object()
