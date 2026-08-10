"""Summarize paired Spark RNG-install microbenchmark reports."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from rng_install import METRICS

INSTALL_LIMIT_NS = 6_000.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0x524E_4749_4E53_5441


def read_report(
    path: Path, expected_core: int
) -> tuple[dict[str, str], dict[str, list[float]]]:
    """Read and validate one raw trial CSV from the measurement harness."""
    metadata: dict[str, str] = {}
    measurements: dict[str, list[float]] = defaultdict(list)
    trials: dict[str, set[int]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if row[0] == "meta":
                metadata[row[1]] = row[2]
            elif row[0] == "data":
                metric, trial, _iterations, _elapsed, ns_per_op, checksum = row[1:]
                if metric not in METRICS:
                    raise ValueError(f"unknown RNG install metric {metric}")
                ordinal = int(trial)
                if ordinal in trials[metric]:
                    raise ValueError(f"duplicate trial {ordinal} for {metric}")
                if int(checksum) == 0:
                    raise ValueError(f"zero checksum for {metric} trial {ordinal}")
                trials[metric].add(ordinal)
                measurements[metric].append(float(ns_per_op))
    if metadata.get("core") != str(expected_core):
        raise ValueError(f"report does not describe expected core {expected_core}")
    if metadata.get("governor") != "performance":
        raise ValueError("report governor is not performance")
    if set(measurements) != set(METRICS):
        raise ValueError("report does not contain the complete metric set")
    expected_trials = int(metadata.get("trials", "0"))
    if expected_trials < 10 or any(
        len(values) != expected_trials for values in measurements.values()
    ):
        raise ValueError("report does not contain every required trial")
    return metadata, dict(measurements)


def interval(values: list[float]) -> tuple[float, float]:
    """Return the deterministic percentile interval over resampled means."""
    generator = random.Random(BOOTSTRAP_SEED)
    means = [
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    means.sort()
    return means[249], means[9749]


def summarize(measurements: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Compute descriptive statistics and bootstrap intervals per mechanism."""
    output = {}
    for metric in METRICS:
        values = measurements[metric]
        lower, upper = interval(values)
        output[metric] = {
            "ci95_lower_ns": lower,
            "ci95_upper_ns": upper,
            "mean_ns": statistics.fmean(values),
            "median_ns": statistics.median(values),
            "minimum_ns": min(values),
            "maximum_ns": max(values),
        }
    return output


def evaluate(
    performance_path: Path,
    efficiency_path: Path,
    performance_core: int,
    efficiency_core: int,
) -> dict[str, Any]:
    """Validate both cores and decide the complete install on the reference core."""
    performance_meta, performance_values = read_report(
        performance_path, performance_core
    )
    efficiency_meta, efficiency_values = read_report(efficiency_path, efficiency_core)
    for key in (
        "python",
        "torch",
        "numpy",
        "iterations",
        "warmup_iterations",
        "trials",
    ):
        if performance_meta.get(key) != efficiency_meta.get(key):
            raise ValueError(f"reports disagree on {key}")
    performance = summarize(performance_values)
    efficiency = summarize(efficiency_values)
    upper = performance["full_install"]["ci95_upper_ns"]
    return {
        "decision": "PASS" if upper <= INSTALL_LIMIT_NS else "FAIL",
        "efficiency": efficiency,
        "efficiency_core": efficiency_core,
        "efficiency_to_performance_ratio": {
            metric: efficiency[metric]["mean_ns"] / performance[metric]["mean_ns"]
            for metric in METRICS
        },
        "install_limit_ns": INSTALL_LIMIT_NS,
        "performance": performance,
        "performance_core": performance_core,
        "reference_upper_ns": upper,
        "runtime": {key: performance_meta[key] for key in ("python", "torch", "numpy")},
        "trials": int(performance_meta["trials"]),
    }


def main() -> None:
    """Write one machine-readable decision report from both raw core reports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--efficiency", type=Path, required=True)
    parser.add_argument("--performance-core", type=int, required=True)
    parser.add_argument("--efficiency-core", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate(
        arguments.performance,
        arguments.efficiency,
        arguments.performance_core,
        arguments.efficiency_core,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
