"""Strict-order delivery over a bounded native execution frontier."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from ..state import (
    DeliveredBatchState,
    decode_delivered_bitmap,
    resume_sample_position,
)
from ..telemetry.delivery import build_delivery_telemetry
from .completion import CompletionBatchDelivery
from .exceptions import WorkerDied
from .factory import prepare_process_pool
from .frontier import FrontierRuntime, binding_cause
from .sizing import delivery_length, frontier_ceiling, frontier_depth


class ProcessIterator(Iterator[Any]):
    """Execute ahead within a fixed frontier and commit in sampler order."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._epoch = loader._epoch
        self._length = delivery_length(loader)
        resume_position = resume_sample_position(loader, self._length)
        self._on_completion = loader.delivery == "on-completion"
        width = loader.batch_size or 1
        total_batches = (self._length + width - 1) // width
        restored_batches = (
            decode_delivered_bitmap(
                loader._resume_cursor_batches,
                loader._resume_delivered_bitmap,
                total_batches,
            )
            if self._on_completion
            else set()
        )
        restored_samples = sum(
            min(width, self._length - ordinal * width) for ordinal in restored_batches
        )
        self._position = resume_position + restored_samples
        self._complete = False
        self._valid = True
        self._ready: dict[int, tuple[int, bytes, int]] = {}
        self._delivered = DeliveredBatchState(
            loader._resume_cursor_batches, restored_batches
        )
        self._completion = (
            CompletionBatchDelivery(self) if self._on_completion else None
        )
        self._schedule: FrontierRuntime | None = None
        self._worker_batches = False
        self._last_delivery_ns = time.perf_counter_ns()
        self._delivery_telemetry = build_delivery_telemetry(loader)
        if self._length:
            prepare_process_pool(loader)
            depth = frontier_depth(loader)
            batch_size = loader._process_pool.batch_size
            self._worker_batches = batch_size is not None
            schedule_start = (
                resume_position // batch_size
                if batch_size is not None
                else resume_position
            )
            schedule_length = (
                (self._length + batch_size - 1) // batch_size
                if batch_size is not None
                else self._length
            )
            schedule_depth = (
                max(1, (depth + batch_size - 1) // batch_size)
                if batch_size is not None
                else depth
            )
            ceiling = frontier_ceiling(loader)
            schedule_ceiling = (
                max(1, (ceiling + batch_size - 1) // batch_size)
                if batch_size is not None
                else ceiling
            )
            restored_positions = (
                set(restored_batches)
                if batch_size is not None
                else {
                    position
                    for ordinal in restored_batches
                    for position in range(
                        ordinal * width,
                        min((ordinal + 1) * width, self._length),
                    )
                }
            )
            if restored_positions:
                restored_span = max(restored_positions) - schedule_start + 1
                schedule_depth = max(schedule_depth, restored_span)
                schedule_ceiling = max(schedule_ceiling, restored_span)
            self._schedule = FrontierRuntime(
                schedule_length,
                schedule_depth,
                schedule_ceiling,
                loader._process_pool.worker_count,
                loader.config.factors.growth_mult,
                binding_cause(loader),
                self._dispatch_cost,
                start=schedule_start,
            )
            for position in sorted(restored_positions):
                self._schedule.seed_delivered(position)
            self._schedule.set_worker_count(loader._controller.width)
            self._fill_frontier()

    def __iter__(self) -> ProcessIterator:
        return self

    def __next__(self) -> Any:
        started = time.perf_counter_ns()
        previous_position = self._position
        try:
            value = self._next_value()
            if self._delivery_telemetry is not None:
                samples = self._position - previous_position
                self._delivery_telemetry.record_delivery(
                    samples,
                    self._loader._process_pool.bytes_sample * samples,
                    started,
                )
            return value
        finally:
            if self._schedule is not None:
                self._schedule.record_active(time.perf_counter_ns() - started)

    def _next_value(self) -> Any:
        """Produce one value while the public wrapper records active loader time."""
        if not self._valid:
            raise RuntimeError("process iterator is no longer active")
        try:
            if self._position >= self._length:
                self._finish_epoch()
                raise StopIteration
            if self._on_completion:
                ordinal, value, delivered_samples = self._completion.next_batch()
                self._position += delivered_samples
                self._delivered.mark(ordinal)
                self._loader._epoch_state.mark_delivered(self._epoch)
                self._adapt_controller(delivered_samples)
                return value
            batch_size = self._loader.batch_size
            if batch_size is None:
                position = self._position
                sample = self._next_sample(position)
                self._position += 1
                self._loader._epoch_state.mark_delivered(self._epoch)
                self._adapt_controller(1)
                return sample
            start = self._position
            stop = min(start + batch_size, self._length)
            batch = (
                self._next_worker_batch(start // batch_size)
                if self._worker_batches
                else self._next_batch(start, stop)
            )
            self._position = stop
            self._loader._epoch_state.mark_delivered(self._epoch)
            self._adapt_controller(stop - start)
            return batch
        except StopIteration:
            raise
        except WorkerDied:
            raise
        except BaseException:
            self._loader.close()
            raise

    def _next_sample(self, expected_position: int) -> Any:
        status, payload, worker = self._next_completion(expected_position)
        return self._loader._process_pool.decode(status, payload, worker)

    def _next_batch(self, start: int, stop: int) -> Any:
        pool = self._loader._process_pool
        samples = []
        for position in range(start, stop):
            status, payload, worker = self._next_completion(position)
            samples.append(pool.decode(status, payload, worker))
        return self._loader._collate_batch(samples)

    def _next_worker_batch(self, ordinal: int) -> Any:
        status, payload, worker = self._next_completion(ordinal)
        return self._loader._process_pool.decode_batch(status, payload, worker)

    def _next_completion(self, expected_position: int) -> tuple[int, bytes, int]:
        pool = self._loader._process_pool
        deadline = pool.deadline()
        while True:
            self._fill_frontier()
            position = self._schedule.try_commit()
            if position is not None:
                if position != expected_position:
                    raise RuntimeError("scheduler committed a noncontiguous position")
                status, payload, worker = self._ready.pop(position)
                self._fill_frontier()
                return status, payload, worker
            progressed = self._poll_completions()
            if not progressed:
                pool.check_workers(deadline)
                wait_started = time.perf_counter_ns()
                pool.wait_for_completion(deadline)
                self._schedule.record_wait(time.perf_counter_ns() - wait_started)
                if self._delivery_telemetry is not None:
                    self._delivery_telemetry.record_stall()

    def _fill_frontier(self) -> None:
        pool = self._loader._process_pool
        order = self._schedule.dispatch_order()
        retained_probe = pool.retained_probe_command
        if retained_probe in order:
            order.remove(retained_probe)
            order.insert(0, retained_probe)
        for selected_position in order:
            dispatch = self._schedule.dispatch_at(selected_position)
            if dispatch is None:
                continue
            position, worker = dispatch
            batch_size = pool.batch_size
            sample_position = (
                position * batch_size if batch_size is not None else position
            )
            sampler_runtime = self._loader._sampler_runtime
            coordinate = (
                self._loader._map_coordinate(sample_position)
                if sampler_runtime is None
                else sample_position
            )
            index = (
                self._loader._map_index(self._epoch, sample_position)
                if sampler_runtime is None
                else sampler_runtime.index(sample_position)
            )
            batch_len = (
                min(batch_size, self._length - sample_position)
                if batch_size is not None
                else 0
            )
            if not pool.try_submit(
                self._epoch,
                position,
                index,
                worker,
                batch_len=batch_len,
                coordinate=coordinate,
            ):
                return
            self._schedule.mark_dispatched(position, worker)

    def _dispatch_cost(self, position: int) -> float | None:
        """Estimate one transport command from its profiled sample positions."""
        profile = self._loader._cost_profile
        if profile is None:
            return None
        batch_size = self._loader._process_pool.batch_size
        if batch_size is None:
            return profile.estimate(position)
        start = position * batch_size
        stop = min(start + batch_size, self._length)
        estimates = [profile.estimate(sample) for sample in range(start, stop)]
        if any(estimate is None for estimate in estimates):
            return None
        return sum(estimate for estimate in estimates if estimate is not None)

    def _adapt_controller(self, delivered_samples: int) -> None:
        """Apply one cadenced controller decision by parking scheduler routes."""
        now_ns = time.perf_counter_ns()
        elapsed_ns = max(1, now_ns - self._last_delivery_ns)
        self._last_delivery_ns = now_ns
        if self._position >= self._length:
            self._schedule.consume_stall_flag()
            return
        batch_size = 1 if self._worker_batches else (self._loader.batch_size or 1)
        bytes_per_second = (
            self._loader._process_pool.bytes_sample
            * delivered_samples
            * 1_000_000_000.0
            / elapsed_ns
        )
        decision = self._loader._controller.observe(
            now_ns=now_ns,
            stalled=self._schedule.consume_stall_flag(),
            occupied=self._schedule.occupied,
            batch_size=batch_size,
            bytes_per_second=bytes_per_second,
        )
        if decision is None:
            return
        from hyperloader.control import decision_report

        self._schedule.set_worker_count(decision.width)
        self._loader._last_controller_report = decision_report(decision)
        if self._delivery_telemetry is not None:
            self._delivery_telemetry.record_controller(decision)

    def _poll_completions(self) -> bool:
        pool = self._loader._process_pool
        progressed = False
        for worker in range(pool.worker_count):
            completion = pool.try_receive(worker)
            if completion is None:
                continue
            position, status, payload, cost_ns = completion
            self._schedule.mark_completed(position, worker)
            self._ready[position] = (status, payload, worker)
            if self._on_completion:
                self._completion.observe(position)
            if status in {0, 2}:
                self._record_cost(position, cost_ns)
            progressed = True
        return progressed

    def _record_cost(self, position: int, cost_ns: int) -> None:
        profile = getattr(self._loader, "_cost_profile", None)
        if profile is None:
            return
        batch_size = self._loader._process_pool.batch_size
        if batch_size is None:
            profile.observe(position, cost_ns)
            return
        start = position * batch_size
        batch_len = min(batch_size, self._length - start)
        per_sample_ns = max(1, cost_ns // batch_len)
        for sample_position in range(start, start + batch_len):
            profile.observe(sample_position, per_sample_ns)

    def _finish_epoch(self) -> None:
        if not self._complete:
            self._loader._epoch_state.complete(self._epoch)
            from hyperloader.profile import save_cost_profile

            save_cost_profile(self._loader)
            if self._schedule is not None:
                self._loader._last_frontier_report = self._schedule.report()
            if self._delivery_telemetry is not None:
                self._delivery_telemetry.finish_epoch(self._epoch)
            self._complete = True

    def _flush_telemetry(self) -> None:
        if self._delivery_telemetry is not None:
            self._delivery_telemetry.flush()

    @property
    def complete(self) -> bool:
        """Report whether exhaustion advanced the loader epoch."""
        return self._complete

    @property
    def coordinate_epoch(self) -> int:
        """Return the epoch carried by this iterator's checkpoint coordinate."""
        return self._epoch

    @property
    def delivered_batches(self) -> int:
        """Return the strict delivered-batch prefix count."""
        if self._on_completion:
            return self._delivered.base
        batch_size = self._loader.batch_size or 1
        return (self._position + batch_size - 1) // batch_size

    @property
    def delivered_bitmap(self) -> bytes:
        """Encode delivered batches beyond the contiguous prefix."""
        return self._delivered.bitmap() if self._on_completion else b""

    @property
    def sampler_checksum(self) -> int:
        """Return the checksum through the delivered user-sampler prefix."""
        runtime = self._loader._sampler_runtime
        return 0 if runtime is None else runtime.checksum_at(self._position)

    def invalidate(self) -> None:
        """Prevent a replaced iterator from consuming a new pool's completions."""
        self._valid = False
