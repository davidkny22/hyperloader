"""Explicit active-batch timing for diagnosis."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Event, Thread
from typing import Any


def run_probe(loader: Any, batches: int) -> dict[str, object]:
    """Consume a bounded fresh iterator and report the activity separately."""
    if not isinstance(batches, int) or isinstance(batches, bool) or not 1 <= batches <= 32:
        raise ValueError("probe_batches must be an integer from 1 through 32")
    if _has_active_iterator(loader):
        raise RuntimeError("active probing requires a loader with no live iterator")
    iterator = iter(loader)
    timings: list[int] = []

    def consume() -> None:
        for _ in range(batches):
            started = time.perf_counter_ns()
            try:
                next(iterator)
            except StopIteration:
                break
            timings.append(time.perf_counter_ns() - started)

    progress, sample_ns = _thread_progress(consume)
    elapsed = sum(timings)
    calibration_ns = max(sample_ns, 10_000_000)
    calibration, measured_calibration_ns = _thread_progress(
        lambda: time.sleep(calibration_ns / 1_000_000_000)
    )
    gil_release = _progress_fraction(
        progress, sample_ns, calibration, measured_calibration_ns
    )
    return {
        "active": True,
        "basis": "wall time around each consumed next() call",
        "consumed_batches": len(timings),
        "elapsed_ns": elapsed,
        "gil_calibration_ns": measured_calibration_ns,
        "gil_release_fraction": gil_release,
        "gil_sample_ns": sample_ns,
        "mean_batch_ns": None if not timings else elapsed / len(timings),
        "requested_batches": batches,
    }


def _has_active_iterator(loader: Any) -> bool:
    reference = getattr(loader, "_active_iterator_ref", None)
    if reference is not None and reference() is not None:
        return True
    return getattr(loader, "_iterator", None) is not None


def _thread_progress(action: Callable[[], None]) -> tuple[int, int]:
    stop = Event()
    started = Event()
    counter = [0]

    def count() -> None:
        started.set()
        while not stop.is_set():
            counter[0] += 1

    thread = Thread(target=count, name="hyperloader-diagnose-gil", daemon=True)
    thread.start()
    started.wait()
    before = time.perf_counter_ns()
    try:
        action()
    finally:
        elapsed = time.perf_counter_ns() - before
        stop.set()
        thread.join()
    return counter[0], elapsed


def _progress_fraction(
    observed: int, observed_ns: int, calibration: int, calibration_ns: int
) -> float | None:
    if not observed_ns or not calibration or not calibration_ns:
        return None
    observed_rate = observed / observed_ns
    calibration_rate = calibration / calibration_ns
    return min(1.0, observed_rate / calibration_rate)
