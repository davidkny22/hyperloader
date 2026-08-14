"""CUDA and host timing segments for live training diagnostics."""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch.nn import functional

from ..training_step import TransformerStepRunner
from ..vision_step import VisionStepRunner


def profile_copy(runner: Any, batch: Any) -> tuple[torch.Tensor, ...]:
    """Expose training-batch transfer as a stable profiler frame."""
    if isinstance(runner, TransformerStepRunner):
        batch.validate()
        return (batch.tokens.to(runner.device, non_blocking=runner.non_blocking),)
    if isinstance(runner, VisionStepRunner):
        batch.validate()
        return (
            batch.images.to(runner.device, non_blocking=runner.non_blocking),
            batch.labels.to(runner.device, non_blocking=runner.non_blocking),
        )
    raise TypeError("unsupported diagnostic training runner")


def profile_compute(runner: Any, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Expose forward, backward, and optimizer launch work as one profiler frame."""
    runner.optimizer.zero_grad(set_to_none=True)
    if isinstance(runner, TransformerStepRunner):
        tokens = tensors[0]
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        with runner._autocast():
            logits = runner.model(inputs)
            loss = functional.cross_entropy(
                logits.reshape(-1, runner.vocabulary_size), targets.reshape(-1)
            )
    elif isinstance(runner, VisionStepRunner):
        images, labels = tensors
        with runner._autocast():
            loss = functional.cross_entropy(runner.model(images), labels)
    else:
        raise TypeError("unsupported diagnostic training runner")
    runner._scaler.scale(loss).backward()
    runner._scaler.step(runner.optimizer)
    runner._scaler.update()
    return loss.detach()


def profile_sync(device: torch.device) -> None:
    """Expose the blocking device wait as a stable profiler frame."""
    torch.cuda.synchronize(device)


class DiagnosticStep:
    """Run the exact training step with CUDA and host segment boundaries."""

    def __init__(self, runner: TransformerStepRunner | VisionStepRunner) -> None:
        if runner.device.type != "cuda":
            raise ValueError("training diagnostics require CUDA execution")
        self.runner = runner
        self._start = torch.cuda.Event(enable_timing=True)
        self._copy_end = torch.cuda.Event(enable_timing=True)
        self._compute_end = torch.cuda.Event(enable_timing=True)

    def run(self, batch: Any) -> dict[str, float]:
        """Execute one step and return host and CUDA timing segments."""
        host_started = time.perf_counter()
        self._start.record()
        copy_started = time.perf_counter()
        tensors = profile_copy(self.runner, batch)
        copy_returned = time.perf_counter()
        self._copy_end.record()
        compute_started = time.perf_counter()
        profile_compute(self.runner, tensors)
        compute_returned = time.perf_counter()
        self._compute_end.record()
        sync_started = time.perf_counter()
        profile_sync(self.runner.device)
        completed = time.perf_counter()
        return {
            "cuda_copy_ms": self._start.elapsed_time(self._copy_end),
            "cuda_compute_ms": self._copy_end.elapsed_time(self._compute_end),
            "cuda_total_ms": self._start.elapsed_time(self._compute_end),
            "host_copy_call_ms": 1000.0 * (copy_returned - copy_started),
            "host_compute_launch_ms": 1000.0 * (compute_returned - compute_started),
            "host_sync_ms": 1000.0 * (completed - sync_started),
            "host_total_ms": 1000.0 * (completed - host_started),
        }


def summarize_timings(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Summarize every timing field without discarding the raw iterations."""
    if not rows:
        raise ValueError("timing summaries require at least one iteration")
    result = {}
    for key in rows[0]:
        values = sorted(row[key] for row in rows)
        result[key] = {
            "mean": statistics.fmean(values),
            "median": _percentile(values, 0.5),
            "p95": _percentile(values, 0.95),
        }
    return result


def _percentile(values: list[float], quantile: float) -> float:
    position = quantile * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])
