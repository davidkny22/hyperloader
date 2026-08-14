"""Bounded live-training target for residual-gap profiling."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from ..controls.processes import process_record
from .cpu_activity import diff_cpu_activity, snapshot_cpu_activity
from .fixture import DiagnosticFixture, build_fixture
from .segments import DiagnosticStep, summarize_timings


def profile_next_batch(feeder: Any) -> Any:
    """Expose feeder delivery as a stable profiler frame."""
    return feeder.next_batch()


def _profile_half(
    fixture: DiagnosticFixture, step: DiagnosticStep, system: str, seconds: float
) -> dict[str, Any]:
    feeder = fixture.feeders[system]
    cpu_before = snapshot_cpu_activity()
    controls_before = _feeder_snapshot(feeder)
    iterations = []
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        batch_started = time.perf_counter()
        batch = profile_next_batch(feeder)
        batch_returned = time.perf_counter()
        timing = step.run(batch)
        timing["host_next_batch_ms"] = 1000.0 * (batch_returned - batch_started)
        iterations.append(timing)
    finished = time.perf_counter()
    controls_after = _feeder_snapshot(feeder)
    cpu_after = snapshot_cpu_activity()
    return {
        "system": system,
        "elapsed_seconds": finished - started,
        "iterations": len(iterations),
        "iterations_per_second": len(iterations) / (finished - started),
        "segments": summarize_timings(iterations),
        "raw_segments": iterations,
        "per_core_cpu": diff_cpu_activity(cpu_before, cpu_after),
        "process_state_before": controls_before,
        "process_state_after": controls_after,
    }


def profile_hyperloader_half(
    fixture: DiagnosticFixture, step: DiagnosticStep, seconds: float
) -> dict[str, Any]:
    """Keep the hyperloader half identifiable in external stack samples."""
    return _profile_half(fixture, step, "hyperloader", seconds)


def profile_counterfactual_half(
    fixture: DiagnosticFixture, step: DiagnosticStep, seconds: float
) -> dict[str, Any]:
    """Keep the counterfactual half identifiable in external stack samples."""
    return _profile_half(fixture, step, "counterfactual", seconds)


def main() -> None:
    """Run two alternating diagnostic pairs after an external profiler attaches."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--product-commit", required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--half-seconds", type=float, default=15.0)
    parser.add_argument("--profile-ready-file", type=Path, required=True)
    parser.add_argument("--profiler-timeout-seconds", type=float, default=30.0)
    arguments = parser.parse_args()
    if arguments.half_seconds <= 0 or arguments.output.exists():
        raise ValueError("diagnostic duration must be positive and output must be new")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture(
        arguments.decision,
        output=arguments.output.parent,
        image_root=arguments.image_root,
    )
    step = DiagnosticStep(fixture.runner)
    try:
        for feeder in fixture.feeders.values():
            for _ in range(fixture.config.warmup_steps):
                step.run(feeder.next_batch())
        profile_start = threading.Event()
        signal.signal(signal.SIGUSR1, lambda _signum, _frame: profile_start.set())
        arguments.profile_ready_file.parent.mkdir(parents=True, exist_ok=True)
        arguments.profile_ready_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
        if not profile_start.wait(arguments.profiler_timeout_seconds):
            raise TimeoutError("sampling profiler did not arm before the timeout")
        halves = []
        halves.append(profile_counterfactual_half(fixture, step, arguments.half_seconds))
        halves.append(profile_hyperloader_half(fixture, step, arguments.half_seconds))
        halves.append(profile_hyperloader_half(fixture, step, arguments.half_seconds))
        halves.append(profile_counterfactual_half(fixture, step, arguments.half_seconds))
        report = {
            "kind": "training-residual-diagnostic",
            "harness_commit": arguments.harness_commit,
            "product_commit": arguments.product_commit,
            "point_id": fixture.config.point_id,
            "subject_workers": fixture.config.subject_workers,
            "subject_prefetch": fixture.config.subject_prefetch,
            "half_seconds": arguments.half_seconds,
            "consumer_process": process_record(os.getpid()),
            "halves": halves,
        }
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        fixture.close()


def _feeder_snapshot(feeder: Any) -> dict[str, Any]:
    capture = getattr(feeder, "control_snapshot", None)
    if capture is None:
        return {
            "system": getattr(feeder, "system", type(feeder).__name__),
            "consumer_process": process_record(os.getpid()),
        }
    return capture()


if __name__ == "__main__":
    main()
