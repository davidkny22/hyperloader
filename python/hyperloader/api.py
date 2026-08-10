"""Public DataLoader surface and constructor-level contract validation."""

from __future__ import annotations

import weakref
import warnings
from collections.abc import Iterator
from typing import Any

from .config import AUTO, Auto, HyperConfig
from .epoch import EpochState
from .planner import BlackBoxPlan, TensorPlan, build_plan
from .process.factory import prepare_process_pool
from .process.seed import resolve_root_seed


def _require_nonnegative_workers(num_workers: int | Auto) -> None:
    if num_workers is AUTO:
        return
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers must be auto or an integer")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")


def _resolve_delivery(in_order: bool, delivery: str | Auto) -> str:
    expected = "in-order" if in_order else "on-completion"
    if delivery is AUTO or delivery == "auto":
        return expected
    if delivery not in {"in-order", "on-completion"}:
        raise ValueError("delivery must be auto, in-order, or on-completion")
    if delivery != expected:
        raise ValueError(f"delivery={delivery!r} conflicts with in_order={in_order!r}")
    return delivery


class DataLoader:
    """Load batches through hyperloader's contract-preserving execution engine."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Any = None,
        batch_sampler: Any = None,
        num_workers: int | Auto = AUTO,
        collate_fn: Any = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Any = None,
        multiprocessing_context: Any = None,
        generator: Any = None,
        *,
        prefetch_factor: int | None | Auto = AUTO,
        persistent_workers: bool = True,
        pin_memory_device: str = "",
        in_order: bool = True,
        seed: int | None = None,
        thread_safe: bool = False,
        mode: str = "native",
        delivery: str | Auto = AUTO,
        device: str | None = None,
        config: HyperConfig | None = None,
    ) -> None:
        """Validate and retain a loader configuration for plan construction."""
        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")
        if batch_sampler is not None and (
            batch_size != 1 or shuffle or sampler is not None or drop_last
        ):
            raise ValueError(
                "batch_sampler is mutually exclusive with batch_size, shuffle, sampler, "
                "and drop_last"
            )
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None")
        if batch_size is None and drop_last:
            raise ValueError("batch_size=None is mutually exclusive with drop_last")

        _require_nonnegative_workers(num_workers)
        if timeout < 0:
            raise ValueError("timeout must be nonnegative")
        if prefetch_factor is not AUTO and prefetch_factor is not None:
            if isinstance(prefetch_factor, bool) or prefetch_factor <= 0:
                raise ValueError("prefetch_factor must be auto, None, or positive")
            if num_workers == 0:
                raise ValueError(
                    "prefetch_factor can only be specified when num_workers is positive"
                )
        if mode not in {"native", "torch-compat"}:
            raise ValueError("mode must be native or torch-compat")

        resolved_config = config if config is not None else HyperConfig()
        if (
            seed is not None
            and resolved_config.seed is not None
            and seed != resolved_config.seed
        ):
            raise ValueError("seed conflicts with config.seed")
        resolved_seed = seed if seed is not None else resolved_config.seed

        process_ceiling = resolved_config.executor.process_ceiling
        if (
            process_ceiling is not AUTO
            and num_workers is not AUTO
            and process_ceiling != num_workers
        ):
            raise ValueError(
                "num_workers conflicts with config.executor.process_ceiling"
            )
        if (
            mode == "torch-compat"
            and resolved_config.executor.on_worker_death == "restart"
        ):
            raise ValueError("worker restart is unavailable in torch-compat mode")

        resolved_delivery = _resolve_delivery(in_order, delivery)
        configured_memory = resolved_config.memory.delivery_memory
        requested_memory = (
            "device" if device is not None else "pinned" if pin_memory else None
        )
        if (
            configured_memory != "auto"
            and requested_memory is not None
            and configured_memory != requested_memory
        ):
            raise ValueError(
                "memory.delivery_memory conflicts with pin_memory or device"
            )
        resolved_memory = (
            configured_memory
            if configured_memory != "auto"
            else requested_memory
            if requested_memory is not None
            else "auto"
        )

        self.dataset = dataset
        self.batch_size = None if batch_sampler is not None else batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.generator = generator
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device
        self.in_order = in_order
        self.seed = resolved_seed
        self.thread_safe = thread_safe
        self.mode = mode
        self.delivery = resolved_delivery
        self.device = device
        self.config = resolved_config
        self.delivery_memory = resolved_memory
        self.root_seed = resolve_root_seed(resolved_seed, generator)
        self._epoch_state = EpochState()
        self._abandon_notice_emitted = False
        self._process_pool: Any = None
        self._active_iterator_ref: Any = None
        self._plan = build_plan(dataset, shuffle)
        if (
            isinstance(self._plan, BlackBoxPlan)
            and num_workers is not AUTO
            and num_workers > 0
            and sampler is None
            and batch_sampler is None
            and collate_fn is None
            and mode == "native"
            and not thread_safe
        ):
            prepare_process_pool(self)

    def __iter__(self) -> Iterator[Any]:
        """Create an iterator over the selected native execution plan."""
        from .process.iterator import ProcessIterator
        from .tensor import TensorIterator

        if self.num_workers is AUTO or self.num_workers == 0:
            raise RuntimeError(
                "the requested hyperloader execution tier is not initialized"
            )
        if self.mode != "native" or self.thread_safe:
            raise RuntimeError(
                "the requested hyperloader execution mode is not initialized"
            )
        if self._plan is None:
            raise RuntimeError("iterable planning is not initialized")
        if self.sampler is not None or self.batch_sampler is not None:
            raise RuntimeError("user sampler planning is not initialized")
        if self.collate_fn is not None:
            raise RuntimeError("user collation planning is not initialized")
        auto_advanced = self._epoch_state.begin_iteration()
        if auto_advanced and not self._abandon_notice_emitted:
            warnings.warn(
                "A fresh iterator after partial delivery advanced the epoch. "
                "Call set_epoch(epoch) before iterating to replay explicitly.",
                UserWarning,
                stacklevel=2,
            )
            self._abandon_notice_emitted = True
        active = (
            None if self._active_iterator_ref is None else self._active_iterator_ref()
        )
        if active is not None and not active.complete:
            self.close()
        if isinstance(self._plan, TensorPlan):
            iterator = TensorIterator(self)
        else:
            iterator = ProcessIterator(self)
        self._active_iterator_ref = weakref.ref(iterator)
        return iterator

    @property
    def _epoch(self) -> int:
        """Return the epoch selected for the next map-style iterator."""
        return self._epoch_state.current

    def set_epoch(self, epoch: int) -> None:
        """Select an epoch and reset the next iterator to its first batch."""
        self._epoch_state.set_epoch(epoch)

    def close(self) -> None:
        """Release persistent process resources owned by this loader."""
        active = (
            None
            if getattr(self, "_active_iterator_ref", None) is None
            else self._active_iterator_ref()
        )
        if active is not None:
            active.invalidate()
        self._active_iterator_ref = None
        if getattr(self, "_process_pool", None) is not None:
            self._process_pool.close()
            self._process_pool = None

    def _collate_batch(self, batch: list[Any]) -> Any:
        """Collate an engine-produced batch through the native contract mirror."""
        from . import _hyperloader

        return _hyperloader._default_collate(batch)

    def __del__(self) -> None:
        self.close()
