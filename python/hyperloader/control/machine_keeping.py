"""Consumer-gap routing for the native machine-keeping actuator."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

from hyperloader import _hyperloader


class MachineKeepingIterator(Iterator[Any]):
    """Measure consumer gaps and route the calibrated regime to native activity."""

    def __init__(self, loader: Any, iterator: Iterator[Any]) -> None:
        self._loader = loader
        self._iterator = iterator
        self._gap_started_ns = int(
            getattr(loader, "_machine_keeping_last_delivery_ns", 0)
        )
        tax = loader._calibration.idle_state_tax
        self._minimum_gap_ns = tax.minimum_gap_nanoseconds
        self._initial_duty = min(tax.warm_duty_fraction, loader.config.factors.f_warm)
        self._gapless_batches = 0
        self._gapless_limit = (
            loader.config.factors.f_cad_b * loader.config.factors.hysteresis
        )

    def __iter__(self) -> MachineKeepingIterator:
        return self

    def __next__(self) -> Any:
        now = time.perf_counter_ns()
        if self._gap_started_ns:
            gap_ns = now - self._gap_started_ns
            if gap_ns >= self._minimum_gap_ns:
                self._ensure_keeper()
                if self._loader._machine_keeper is not None:
                    self._loader._machine_keeper.observe_gap(gap_ns)
                self._gapless_batches = 0
            elif self._loader._machine_keeper is not None:
                self._gapless_batches += 1
                if self._gapless_batches >= self._gapless_limit:
                    self._park()
        try:
            value = next(self._iterator)
        except BaseException:
            self._park()
            raise
        self._gap_started_ns = time.perf_counter_ns()
        self._loader._machine_keeping_last_delivery_ns = self._gap_started_ns
        return value

    def _ensure_keeper(self) -> None:
        cpus = _consumer_cpus()
        if not cpus:
            return
        if self._loader._machine_keeper is not None:
            if cpus == self._loader._machine_keeper_cpus:
                return
            self._loader._machine_keeper.close()
        self._loader._machine_keeper = _hyperloader._MachineKeeper(
            cpus,
            self._loader.config.factors.f_warm,
            self._initial_duty,
            self._minimum_gap_ns,
        )
        self._loader._machine_keeper_cpus = cpus

    def _park(self) -> None:
        if self._loader._machine_keeper is not None:
            self._loader._machine_keeper.park()

    def _flush_telemetry(self) -> None:
        flush = getattr(self._iterator, "_flush_telemetry", None)
        if flush is not None:
            flush()

    @property
    def complete(self) -> bool:
        """Report the wrapped iterator's completion state."""
        return bool(self._iterator.complete)

    def invalidate(self) -> None:
        """Park machine keeping and invalidate the wrapped iterator."""
        self._park()
        self._iterator.invalidate()


def attach_machine_keeping(loader: Any, iterator: Iterator[Any]) -> Iterator[Any]:
    """Wrap only calibrated automatic execution regimes."""
    calibration = loader._calibration
    if (
        loader.config.control.machine_keeping == "off"
        or calibration is None
        or calibration.idle_state_tax is None
    ):
        return iterator
    return MachineKeepingIterator(loader, iterator)


def _consumer_cpus() -> tuple[int, ...]:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None:
        return ()
    return tuple(sorted(affinity(0)))
