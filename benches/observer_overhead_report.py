"""Validate paired passive-observer overhead measurements."""

from __future__ import annotations

import os
import random
import statistics
from typing import Any

MINIMUM_PAIRS = 20
BOOTSTRAP_DRAWS = 10_000
LOADER_KINDS = ("hyperloader", "torch")


def _interval(values: list[float]) -> tuple[float, float]:
    generator = random.Random(0x4F42534552564552)
    means = [
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    means.sort()
    return means[249], means[9749]


def validate_report(report: dict[str, Any]) -> None:
    metadata = report.get("metadata", {})
    required = {
        "batch_size",
        "batches_per_half",
        "pair_count",
        "platform",
        "public_path_verified",
        "python",
        "torch",
    }
    if set(metadata) != required:
        raise ValueError("measurement metadata fields do not match the required schema")
    if not metadata["public_path_verified"]:
        raise ValueError(
            "measurement did not resolve through the expected installed artifact"
        )
    count = metadata["pair_count"]
    if count < MINIMUM_PAIRS:
        raise ValueError("observer pair count is below the measurement floor")
    expected = [
        "observer-first" if index % 2 == 0 else "baseline-first"
        for index in range(count)
    ]
    for kind in LOADER_KINDS:
        cells = report.get("loaders", {}).get(kind, {})
        pairs = cells.get("pairs", [])
        noise = cells.get("noise_pairs", [])
        if len(pairs) != count or len(noise) != count:
            raise ValueError("observer or noise pair count does not match metadata")
        if [pair.get("order") for pair in pairs] != expected:
            raise ValueError("observer order did not alternate")
        for pair in [*pairs, *noise]:
            if pair["left_checksum"] != pair["right_checksum"]:
                raise ValueError("paired executions did not deliver identical values")
            if min(pair["left_wall_ns"], pair["right_wall_ns"]) < 10_000_000:
                raise ValueError("a measurement half is too short for stable timing")
    probe = report.get("active_probe", {})
    if probe.get("requested_batches") != probe.get("consumed_batches"):
        raise ValueError("active probe consumption does not match its request")


def evaluate_report(report: dict[str, Any]) -> dict[str, Any]:
    validate_report(report)
    decisions: dict[str, Any] = {}
    for kind in LOADER_KINDS:
        penalties = []
        cpu_costs = []
        cells = report["loaders"][kind]
        for pair in cells["pairs"]:
            if pair["order"] == "observer-first":
                observed_wall, baseline_wall = (
                    pair["left_wall_ns"],
                    pair["right_wall_ns"],
                )
                observed_cpu, baseline_cpu = pair["left_cpu_ns"], pair["right_cpu_ns"]
            else:
                baseline_wall, observed_wall = (
                    pair["left_wall_ns"],
                    pair["right_wall_ns"],
                )
                baseline_cpu, observed_cpu = pair["left_cpu_ns"], pair["right_cpu_ns"]
            penalties.append((observed_wall - baseline_wall) / baseline_wall)
            cpu_costs.append(float(observed_cpu - baseline_cpu))
        noise = [
            (1.0 if index % 2 == 0 else -1.0)
            * (pair["left_wall_ns"] - pair["right_wall_ns"])
            / pair["right_wall_ns"]
            for index, pair in enumerate(cells["noise_pairs"])
        ]
        if os.environ.get("HYPERLOADER_OBSERVER_MUTATION") == "inflate-cost":
            penalties = [value + 0.01 for value in penalties]
        interval = _interval(penalties)
        noise_interval = _interval(noise)
        noise_floor = max(abs(noise_interval[0]), abs(noise_interval[1]))
        mean = statistics.fmean(penalties)
        decisions[kind] = {
            "below_noise": abs(mean) <= noise_floor,
            "ci95_penalty": list(interval),
            "mean_cpu_ns_per_call": statistics.fmean(cpu_costs),
            "mean_penalty": mean,
            "noise_ci95": list(noise_interval),
            "noise_floor": noise_floor,
        }
    passed = all(value["below_noise"] for value in decisions.values())
    return {
        "active_probe": report["active_probe"],
        "decision": "PASS" if passed else "FAIL",
        "loaders": decisions,
    }
