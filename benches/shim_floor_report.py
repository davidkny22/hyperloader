"""Decide per-tier shim and seed costs from paired Spark core reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from shim_floor import METRICS

SHIM_LIMIT_NS = 2_000.0
SEED_LIMIT_NS = 6_000.0
AGGREGATE_RATE = 80_000.0
AGGREGATE_LIMIT = 0.75
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0x5348_494D_464C_4F4F


def read_report(
    path: Path, expected_core: int
) -> tuple[dict[str, str], dict[str, list[float]]]:
    """Read one complete affinity-validated raw report."""
    metadata: dict[str, str] = {}
    values: dict[str, list[float]] = defaultdict(list)
    trials: dict[str, set[int]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if row[0] == "meta":
                metadata[row[1]] = row[2]
                continue
            if row[0] != "data":
                continue
            metric, trial, _iterations, _elapsed, ns_per_op, checksum = row[1:]
            if metric not in METRICS:
                raise ValueError(f"unknown shim-floor metric {metric}")
            ordinal = int(trial)
            if ordinal in trials[metric]:
                raise ValueError(f"duplicate trial {ordinal} for {metric}")
            if int(checksum) == 0:
                raise ValueError(f"zero checksum for {metric} trial {ordinal}")
            trials[metric].add(ordinal)
            values[metric].append(float(ns_per_op))
    if metadata.get("core") != str(expected_core):
        raise ValueError(f"report does not describe expected core {expected_core}")
    if metadata.get("governor") != "performance":
        raise ValueError("report governor is not performance")
    if set(values) != set(METRICS):
        raise ValueError("report does not contain the complete metric set")
    expected_trials = int(metadata.get("trials", "0"))
    if expected_trials < 10 or any(
        len(metric_values) != expected_trials for metric_values in values.values()
    ):
        raise ValueError("report does not contain every required trial")
    return metadata, dict(values)


def interval(values: list[float]) -> tuple[float, float]:
    """Return a deterministic percentile interval over resampled means."""
    generator = random.Random(BOOTSTRAP_SEED)
    means = [
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    means.sort()
    return means[249], means[9749]


def statistic(values: list[float]) -> dict[str, float]:
    """Summarize one raw or paired-derived primitive."""
    lower, upper = interval(values)
    return {
        "ci95_lower_ns": lower,
        "ci95_upper_ns": upper,
        "mean_ns": statistics.fmean(values),
        "median_ns": statistics.median(values),
        "minimum_ns": min(values),
        "maximum_ns": max(values),
    }


def tier_statistics(values: dict[str, list[float]], tier: str) -> dict[str, Any]:
    """Derive t_shim and t_seed from same-trial exact tier operations."""
    no_draw = values[f"{tier}_seed_no_draw"]
    seeded = values[f"{tier}_seed_all"]
    total = values[f"{tier}_shim_total"]
    components = {
        name: values[f"{tier}_seed_{name}"]
        for name in ("torch", "numpy", "random")
    }
    if len({len(no_draw), len(seeded), len(total), *(len(v) for v in components.values())}) != 1:
        raise ValueError(f"{tier} trial vectors differ in length")
    shim = [wrapper - seed for wrapper, seed in zip(total, no_draw, strict=True)]
    component_costs = {
        name: statistic(
            [cost - base for cost, base in zip(component, no_draw, strict=True)]
        )
        for name, component in components.items()
    }
    return {
        "seed_no_draw": statistic(no_draw),
        "seed_components": component_costs,
        "t_seed": statistic(seeded),
        "shim_total": statistic(total),
        "t_shim": statistic(shim),
    }


def evaluate_pair(
    performance_path: Path,
    efficiency_path: Path,
    performance_core: int,
    efficiency_core: int,
) -> dict[str, Any]:
    """Decide both tiers and the aggregate efficiency-core clause."""
    performance_meta, performance_values = read_report(
        performance_path, performance_core
    )
    efficiency_meta, efficiency_values = read_report(efficiency_path, efficiency_core)
    if os.environ.get("HYPERLOADER_SHIM_FLOOR_MUTATION") == "inflate-cost":
        performance_values = {
            metric: [
                value + 10_000.0
                if metric.endswith(("seed_all", "shim_total"))
                else value
                for value in values
            ]
            for metric, values in performance_values.items()
        }
    for key in (
        "python",
        "torch",
        "numpy",
        "gil_disabled_build",
        "gil_enabled",
        "iterations",
        "warmup_iterations",
        "trials",
    ):
        if performance_meta.get(key) != efficiency_meta.get(key):
            raise ValueError(f"reports disagree on {key}")
    if (
        performance_meta["gil_disabled_build"] == "True"
        and performance_meta["gil_enabled"] != "False"
    ):
        raise ValueError("free-threaded runtime restored the GIL before measurement")
    performance = {
        tier: tier_statistics(performance_values, tier)
        for tier in ("process", "thread")
    }
    efficiency = {
        tier: tier_statistics(efficiency_values, tier)
        for tier in ("process", "thread")
    }
    decisions = {}
    for tier in ("process", "thread"):
        shim_upper = performance[tier]["t_shim"]["ci95_upper_ns"]
        seed_upper = performance[tier]["t_seed"]["ci95_upper_ns"]
        perf_total = (
            max(0.0, performance[tier]["t_shim"]["mean_ns"])
            + performance[tier]["t_seed"]["mean_ns"]
        )
        eff_total = (
            max(0.0, efficiency[tier]["t_shim"]["mean_ns"])
            + efficiency[tier]["t_seed"]["mean_ns"]
        )
        ratio = eff_total / perf_total
        aggregate = (
            AGGREGATE_RATE * (max(0.0, shim_upper) + seed_upper) * 1e-9 * ratio
        )
        decisions[tier] = {
            "aggregate_eff_core_equivalents": aggregate,
            "efficiency_to_performance_ratio": ratio,
            "pass": shim_upper <= SHIM_LIMIT_NS
            and seed_upper <= SEED_LIMIT_NS
            and aggregate <= AGGREGATE_LIMIT,
            "t_seed_upper_ns": seed_upper,
            "t_shim_upper_ns": shim_upper,
        }
    return {
        "aggregate_limit_eff_core_equivalents": AGGREGATE_LIMIT,
        "aggregate_rate_samples_per_second": AGGREGATE_RATE,
        "decision": "PASS" if all(item["pass"] for item in decisions.values()) else "FAIL",
        "decisions": decisions,
        "efficiency": efficiency,
        "efficiency_core": efficiency_core,
        "performance": performance,
        "performance_core": performance_core,
        "runtime": {
            key: performance_meta[key]
            for key in (
                "python",
                "torch",
                "numpy",
                "gil_disabled_build",
                "gil_enabled",
            )
        },
        "seed_limit_ns": SEED_LIMIT_NS,
        "shim_limit_ns": SHIM_LIMIT_NS,
        "trials": int(performance_meta["trials"]),
    }


def main() -> None:  # pragma: no cover
    """Write one paired-core decision report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--efficiency", type=Path, required=True)
    parser.add_argument("--performance-core", type=int, required=True)
    parser.add_argument("--efficiency-core", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate_pair(
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
