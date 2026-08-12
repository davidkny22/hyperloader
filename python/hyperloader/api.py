"""Public DataLoader surface and constructor-level contract validation."""

from __future__ import annotations

import inspect
import warnings
import weakref
from collections.abc import Iterator
from typing import Any

from .config import AUTO, Auto, HyperConfig
from .constructor import validate_constructor
from .control.machine_keeping import attach_machine_keeping
from .control.runtime import resolve_calibration
from .decoder import bind_decoder_selections, select_decoder_pins
from .distributed import build_map_placement
from .distributed.runtime import capture_topology, validate_runtime_topology
from .epoch import EpochState
from .fingerprint import build_contract_fingerprint, build_dataset_fingerprint
from .memory import ByteLedger
from .memory.pinned import attach_pinned_delivery, configure_pinned_delivery
from .planner import BlackBoxPlan, StagePlan, StructurePlan, TensorPlan, build_plan
from .process.factory import prepare_process_pool
from .process.seed import resolve_root_seed
from .profile import build_cost_profile, save_cost_profile
from .stages import Pipeline
from .telemetry import build_telemetry, telemetry_snapshot

_PERSISTENT_DEFAULT = object()


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
        persistent_workers: bool | object = _PERSISTENT_DEFAULT,
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
        self._persistent_workers_explicit = (
            persistent_workers is not _PERSISTENT_DEFAULT
        )
        self.persistent_workers = (
            mode != "torch-compat"
            if persistent_workers is _PERSISTENT_DEFAULT
            else bool(persistent_workers)
        )
        self.pin_memory_device = pin_memory_device
        self.in_order = in_order
        self.seed = resolved_seed
        self.thread_safe = thread_safe
        self.mode = mode
        self.delivery = resolved.delivery
        self.device = device
        self.config = resolved_config
        self.delivery_memory = resolved.delivery_memory
        self.root_seed = (
            0 if mode == "torch-compat" else resolve_root_seed(resolved_seed, generator)
        )
        self._epoch_state = EpochState()
        self._resume_cursor_batches = 0
        self._resume_sampler_checksum = 0
        self._resume_delivered_bitmap = b""
        self._resume_iterable_state: Any = None
        self._iterable_snapshot_notice_emitted = False
        self._iterable_restart_notice_emitted = False
        self._sampler_runtime: Any = None
        self._abandon_notice_emitted = False
        self._process_pool: Any = None
        self._thread_pool: Any = None
        self._native_batch_probe: Any = None
        self._native_batch_shape: Any = None
        self._active_iterator_ref: Any = None
        self._telemetry = telemetry
        if mode == "torch-compat":
            from .compat import prepare

            prepare(self)
            return
        self._plan = build_plan(dataset, shuffle)
        self._distributed_topology: Any = None
        if self._plan is None:
            self._distributed_topology = capture_topology(
                resolved_config.distributed.rank,
                resolved_config.distributed.world_size,
            )
        self._map_placement = (
            None
            if self._plan is None or sampler is not None or batch_sampler is not None
            else build_map_placement(self)
        )
        self._iterable_payload: bytes | None = None
        if self._plan is None:
            from .iterable import prepare_iterable_source

            self._iterable_payload = prepare_iterable_source(self)
        self._memory_ledger = (
            ByteLedger("contiguous-tensor", "view")
            if (
                isinstance(self._plan, TensorPlan)
                and not self._plan.shuffle
                and self._map_placement.identity
            )
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

        self._machine_identity: Any = None
        self._calibration: Any = None
        self._pinned_delivery: Any = None
        if is_native_batch_path(self) and self._map_placement.identity:
            self._calibration = resolve_calibration(self._machine_identity)
            configure_pinned_delivery(self)
            prepare_native_batch(self)
        self._fingerprint = build_contract_fingerprint(self)
        self._cost_profile = build_cost_profile(self)
        self._machine_keeper: Any = None
        self._machine_keeper_cpus: tuple[int, ...] = ()
        self._machine_keeper_interrupt_cpus: tuple[int, ...] = ()
        self._machine_keeper_consumer_cpu: int | None = None
        self._machine_keeper_route_refresh_ns = 0
        self._machine_keeper_route_batches = 0
        self._machine_keeping_last_delivery_ns = 0
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
            and not (is_native_batch_path(self) and self._map_placement.identity)
        ):
            prepare_process_pool(self)
            self._fingerprint = build_contract_fingerprint(self)

    def __iter__(self) -> Iterator[Any]:
        """Create an iterator over the selected native execution plan."""
        if self.mode == "torch-compat":
            from .compat import iterate

            return iterate(self)
        from .process.iterator import ProcessIterator
        from .structured import StructuredIterator, is_native_batch_path
        from .tensor import TensorIterator

        iterable = self._plan is None
        if not iterable and (self.num_workers is AUTO or self.num_workers == 0):
            raise RuntimeError(
                "the requested hyperloader execution tier is not initialized"
            )
        if self.mode != "native":
            raise RuntimeError(
                "the requested hyperloader execution mode is not initialized"
            )
        if self.collate_fn is not None:
            raise RuntimeError("user collation planning is not initialized")
        if self._distributed_topology is not None:
            validate_runtime_topology(self._distributed_topology)
        auto_advanced = False
        if iterable:
            self._epoch_state.begin_iterable_iteration()
        else:
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
        if iterable:
            from .iterable import IterableIterator

            iterator = IterableIterator(self)
            self._resume_iterable_state = None
        elif self.sampler is not None or self.batch_sampler is not None:
            from .state import (
                StreamingSamplerIterator,
                UserBatchSamplerIterator,
                build_sampler_runtime,
            )

            self._sampler_runtime = build_sampler_runtime(self)
            iterator = (
                UserBatchSamplerIterator(self)
                if self.batch_sampler is not None
                else StreamingSamplerIterator(self)
            )
        elif isinstance(self._plan, TensorPlan):
            iterator = TensorIterator(self)
        elif is_native_batch_path(self) and self._map_placement.identity:
            iterator = StructuredIterator(self)
        elif self._sample_thread_safe:
            from .thread import ThreadIterator

            iterator = ThreadIterator(self)
        else:
            iterator = ProcessIterator(self)
        if self._calibration is None:
            self._calibration = resolve_calibration(self._machine_identity)
        pinned_delivery = configure_pinned_delivery(self)
        iterator = attach_pinned_delivery(pinned_delivery, iterator)
        iterator = attach_machine_keeping(self, iterator)
        self._resume_cursor_batches = 0
        self._resume_sampler_checksum = 0
        self._resume_delivered_bitmap = b""
        self._resume_iterable_state = None
        self._active_iterator_ref = weakref.ref(iterator)
        return iterator

    @property
    def _epoch(self) -> int:
        """Return the epoch selected for the next map-style iterator."""
        return self._epoch_state.current

    def set_epoch(self, epoch: int) -> None:
        """Select an epoch and reset the next iterator to its first batch."""
        self._epoch_state.set_epoch(epoch)
        if self.mode == "torch-compat":
            compat_loader = self._compat_loader
            if compat_loader is None:
                compat_loader = getattr(self, "_compat_reference", None)
            setter = getattr(
                None if compat_loader is None else compat_loader.sampler,
                "set_epoch",
                None,
            )
            if setter is not None:
                setter(epoch)
            self._resume_compat_state = None
        self._resume_cursor_batches = 0
        self._resume_sampler_checksum = 0
        self._resume_delivered_bitmap = b""

    def state_dict(self) -> dict[str, object]:
        """Capture the delivered coordinate for exact continuation."""
        if self.mode == "torch-compat":
            from .compat import capture_state

            return capture_state(self)
        if self._plan is None:
            from .iterable.state import capture_iterable_state

            return capture_iterable_state(self)
        from .state import capture_map_state

        return capture_map_state(self)

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a validated coordinate for the next iterator."""
        if self.mode == "torch-compat":
            from .compat import restore_state

            restore_state(self, state)
            return
        if self._plan is None:
            from .iterable.state import restore_iterable_state

            restore_iterable_state(self, state)
            return
        from .state import restore_map_state

        restore_map_state(self, state)

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
        self._sampler_runtime = None
        self._native_batch_probe = None
        if getattr(self, "_pinned_delivery", None) is not None:
            self._pinned_delivery.close()
            self._pinned_delivery = None
        if getattr(self, "_machine_keeper", None) is not None:
            self._machine_keeper.close()
            self._machine_keeper = None
            self._machine_keeper_cpus = ()
            self._machine_keeper_interrupt_cpus = ()
            self._machine_keeper_consumer_cpu = None
            self._machine_keeper_route_refresh_ns = 0
            self._machine_keeper_route_batches = 0
        self._machine_keeping_last_delivery_ns = 0
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
        if (
            self._pinned_delivery is not None
            and self._pinned_delivery.effective_memory == "pinned"
        ):
            memory = snapshot.setdefault("memory", {})
            if isinstance(memory, dict):
                self._pinned_delivery.compose_memory_report(memory)
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

    def _map_coordinate(self, position: int) -> int:
        """Map one rank-local native position to its global RNG coordinate."""
        return self._map_placement.coordinate(position)

    def _map_index(self, epoch: int, position: int) -> int:
        """Map one rank-local native position to its dataset index."""
        return self._map_placement.index(self._plan, self.root_seed, epoch, position)

    def __del__(self) -> None:
        self.close()


_PUBLIC_SIGNATURE = inspect.signature(DataLoader.__init__)
_PUBLIC_PARAMETERS = []
for _parameter in tuple(_PUBLIC_SIGNATURE.parameters.values())[1:]:
    if _parameter.name == "persistent_workers":
        _parameter = _parameter.replace(default=True)
    _PUBLIC_PARAMETERS.append(_parameter)
DataLoader.__signature__ = _PUBLIC_SIGNATURE.replace(parameters=_PUBLIC_PARAMETERS)
del _PUBLIC_PARAMETERS, _PUBLIC_SIGNATURE, _parameter
