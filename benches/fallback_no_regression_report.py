"""Validate and decide fallback-versus-Torch paired measurements."""

from __future__ import annotations

import random
import statistics
from typing import Any

MIN_PAIRS = 10
BOOTSTRAP_DRAWS = 10_000


def validate_report(report: dict[str, Any]) -> None:
    """Reject measurements that do not prove the installed equal-tuning path."""
    metadata = report["metadata"]
    if not metadata.get("public_path_verified"):
        raise ValueError("the installed public path was not verified")
    if not metadata.get("fallback_resolved"):
        raise ValueError("the native-free fallback did not resolve")
    if not metadata.get("equal_tuning"):
        raise ValueError("the compared systems did not receive equal tuning")
    pair_count = int(metadata["pair_count"])
    if pair_count < MIN_PAIRS:
        raise ValueError(f"pair count must be at least {MIN_PAIRS}")
    expected_names = {"fixed-record", "numpy-array"}
    if set(report["workloads"]) != expected_names:
        raise ValueError("both ordinary fallback workloads are required")
    for name, pairs in report["workloads"].items():
        if len(pairs) != pair_count:
            raise ValueError(f"{name} pair count does not match metadata")
        for ordinal, pair in enumerate(pairs):
            expected_order = "fallback-first" if ordinal % 2 == 0 else "torch-first"
            if pair["ordinal"] != ordinal or pair["order"] != expected_order:
                raise ValueError("pair order must alternate from a contiguous zero ordinal")
            fallback = pair["fallback"]
            torch = pair["torch"]
            if fallback["samples"] != torch["samples"]:
                raise ValueError("both systems must deliver the same sample count")
            if fallback["checksum"] != torch["checksum"]:
                raise ValueError("both systems must deliver identical values")
            if fallback["elapsed_ns"] <= 0 or torch["elapsed_ns"] <= 0:
                raise ValueError("elapsed times must be positive")


def evaluate_report(
    report: dict[str, Any], *, mutate: bool = False, draws: int = BOOTSTRAP_DRAWS
) -> dict[str, Any]:
    """Return per-workload bootstrap intervals and the strict no-regression decision."""
    validate_report(report)
    results: dict[str, Any] = {}
    decision = "PASS"
    for offset, (name, pairs) in enumerate(sorted(report["workloads"].items())):
        gains = []
        for pair in pairs:
            fallback = _throughput(pair["fallback"])
            torch = _throughput(pair["torch"])
            if mutate:
                fallback *= 0.5
            gains.append(100.0 * (fallback - torch) / torch)
        lower, upper = _bootstrap_interval(gains, seed=offset, draws=draws)
        mean = statistics.fmean(gains)
        status = "PASS" if lower >= 0.0 else "FAIL"
        if status == "FAIL":
            decision = "FAIL"
        results[name] = {
            "lower_percent": lower,
            "mean_percent": mean,
            "pairs": len(gains),
            "status": status,
            "upper_percent": upper,
        }
    return {"decision": decision, "workloads": results}


def _throughput(run: dict[str, Any]) -> float:
    return 1_000_000_000.0 * float(run["samples"]) / float(run["elapsed_ns"])


def _bootstrap_interval(
    values: list[float], *, seed: int, draws: int
) -> tuple[float, float]:
    if draws <= 0:
        raise ValueError("bootstrap draw count must be positive")
    generator = random.Random(seed)
    count = len(values)
    means = sorted(
        statistics.fmean(generator.choice(values) for _ in range(count))
        for _ in range(draws)
    )
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _quantile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction
