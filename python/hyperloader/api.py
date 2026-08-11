"""Public DataLoader surface and constructor-level contract validation."""

from __future__ import annotations

import warnings
import weakref
from collections.abc import Iterator
from typing import Any

from .config import AUTO, Auto, HyperConfig
from .constructor import validate_constructor
from .control.machine_keeping import attach_machine_keeping
from .control.runtime import resolve_calibration
from .decoder import bind_decoder_selections, select_decoder_pins
from .epoch import EpochState
from .fingerprint import build_contract_fingerprint, build_dataset_fingerprint
from .memory import ByteLedger
from .planner import BlackBoxPlan, StagePlan, StructurePlan, TensorPlan, build_plan
from .process.factory import prepare_process_pool
from .process.seed import resolve_root_seed
from .profile import build_cost_profile, save_cost_profile
from .stages import Pipeline
from .telemetry import build_telemetry, telemetry_snapshot


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
        telemetry = build_telemetry(
            True if config is None else config.telemetry.enabled
        )
        resolved = validate_constructor(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            prefetch_factor,
            thread_safe,
            mode,
            in_order,
            delivery,
            seed,
            device,
            config,
        )
        resolved_config = resolved.config
        resolved_seed = resolved.seed

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
        self.delivery = resolved.delivery
        self.device = device
        self.config = resolved_config
        self.delivery_memory = resolved.delivery_memory
        self.root_seed = resolve_root_seed(resolved_seed, generator)
        self._epoch_state = EpochState()
        self._abandon_notice_emitted = False
        self._process_pool: Any = None
        self._thread_pool: Any = None
        self._native_batch_probe: Any = None
        self._native_batch_shape: Any = None
        self._active_iterator_ref: Any = None
        self._plan = build_plan(dataset, shuffle)
        self._memory_ledger = (
            ByteLedger("contiguous-tensor", "view")
            if isinstance(self._plan, TensorPlan) and not self._plan.shuffle
            else None
        )
        self._execution_dataset = (
            self._plan.execution_dataset
            if isinstance(self._plan, StructurePlan)
            else dataset
        )
        if (
            thread_safe
            and isinstance(self._plan, StagePlan)
            and not self._plan.sample_thread_safe
        ):
            raise ValueError(
                "thread_safe=True conflicts with an isolated pipeline sample stage"
            )
        self._sample_thread_safe = (
            self._plan.sample_thread_safe
            if isinstance(self._plan, StagePlan)
            else thread_safe
        )
        self._decoder_selections = select_decoder_pins(
            dataset, resolved_config.determinism.decoder_pins
        )
        self._execution_dataset = bind_decoder_selections(
            self._execution_dataset, self._decoder_selections
        )
        from .structured import bind_native_pipeline

        self._execution_dataset = bind_native_pipeline(
            self._execution_dataset,
            shuffle=bool(shuffle),
            worker_count=num_workers,
            growth=resolved_config.memory.growth,
        )
        self._dataset_fingerprint = build_dataset_fingerprint(
            dataset, resolved_config.determinism.fingerprint
        )
        from .structured import is_native_batch_path, prepare_native_batch

        if is_native_batch_path(self):
            prepare_native_batch(self)
        self._fingerprint = build_contract_fingerprint(self)
        self._machine_identity: Any = None
        self._cost_profile = build_cost_profile(self)
        self._calibration: Any = None
        self._machine_keeper: Any = None
        self._machine_keeper_cpus: tuple[int, ...] = ()
        self._controller: Any = None
        self._telemetry = telemetry
        self._last_frontier_report: dict[str, int | float | str] | None = None
        self._last_controller_report: (
            dict[str, int | float | str | bool | None] | None
        ) = None
        if (
            isinstance(self._plan, (BlackBoxPlan, StagePlan, StructurePlan))
            and num_workers is not AUTO
            and num_workers > 0
            and sampler is None
            and batch_sampler is None
            and collate_fn is None
            and mode == "native"
            and not self._sample_thread_safe
            and not is_native_batch_path(self)
        ):
            prepare_process_pool(self)
            self._fingerprint = build_contract_fingerprint(self)

    def __iter__(self) -> Iterator[Any]:
        """Create an iterator over the selected native execution plan."""
        from .process.iterator import ProcessIterator
        from .structured import StructuredIterator, is_native_batch_path
        from .tensor import TensorIterator

        if self.num_workers is AUTO or self.num_workers == 0:
            raise RuntimeError(
                "the requested hyperloader execution tier is not initialized"
            )
        if self.mode != "native":
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
        elif is_native_batch_path(self):
            iterator = StructuredIterator(self)
        elif self._sample_thread_safe:
            from .thread import ThreadIterator

            iterator = ThreadIterator(self)
        else:
            iterator = ProcessIterator(self)
        if self._calibration is None:
            self._calibration = resolve_calibration(self._machine_identity)
        iterator = attach_machine_keeping(self, iterator)
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
        save_cost_profile(self)
        active = (
            None
            if getattr(self, "_active_iterator_ref", None) is None
            else self._active_iterator_ref()
        )
        if active is not None:
            active.invalidate()
        self._active_iterator_ref = None
        self._native_batch_probe = None
        if getattr(self, "_machine_keeper", None) is not None:
            self._machine_keeper.close()
            self._machine_keeper = None
            self._machine_keeper_cpus = ()
        if getattr(self, "_process_pool", None) is not None:
            self._process_pool.close()
            self._process_pool = None
        if getattr(self, "_thread_pool", None) is not None:
            self._thread_pool.close()
            self._thread_pool = None
        execution_dataset = getattr(self, "_execution_dataset", None)
        if execution_dataset is not getattr(self, "dataset", None):
            close = getattr(execution_dataset, "close", None)
            if close is not None:
                close()

    def stats(self) -> dict[str, object]:
        """Return current telemetry and the latest completed epoch summary."""
        snapshot = telemetry_snapshot(
            self._telemetry,
            self._last_controller_report,
            self._active_iterator_ref,
        )
        memory_report = getattr(self._execution_dataset, "memory_report", None)
        if memory_report is not None:
            snapshot["memory"] = memory_report()
        elif self._memory_ledger is not None:
            snapshot["memory"] = self._memory_ledger.report()
        current = snapshot.get("current")
        if isinstance(current, dict):
            keeper = getattr(self, "_machine_keeper", None)
            current["machine_keeping_duty"] = (
                0.0 if keeper is None else float(keeper.duty())
            )
        return snapshot

    @property
    def decoder_pins(self) -> tuple[dict[str, object], ...]:
        """Disclose the platform-scoped decoder selections for this loader."""
        return tuple(selection.to_dict() for selection in self._decoder_selections)

    def _collate_batch(self, batch: list[Any]) -> Any:
        """Collate an engine-produced batch through the native contract mirror."""
        if isinstance(self.dataset, Pipeline):
            return self.dataset.collate(batch)
        from . import _hyperloader

        return _hyperloader._default_collate(batch)

    def __del__(self) -> None:
        self.close()
