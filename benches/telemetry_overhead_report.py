"""Validate paired public-path telemetry overhead measurements."""

from __future__ import annotations

import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

MINIMUM_PAIRS = 20
CPU_CORE_BUDGET = 0.001
TARGET_SAMPLE_RATE = 80_000
BATCH_SIZE = 64
BOOTSTRAP_DRAWS = 10_000


def load_report(path: Path) -> dict[str, Any]:
    """Load one raw measurement report."""
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_interval(values: list[float]) -> tuple[float, float]:
    generator = random.Random(0x54454C454D455452)
    means = [
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    means.sort()
    return means[249], means[9749]


def validate_report(report: dict[str, Any]) -> None:
    """Reject a report that does not prove the measurement assumptions."""
    metadata = report.get("metadata", {})
    required = {
        "batch_size",
        "batches_per_half",
        "extension_path",
        "pair_count",
        "platform",
        "process_clock_resolution_ns",
        "public_path_verified",
        "python",
        "target_sample_rate",
        "telemetry_summary_verified",
        "torch",
    }
    if set(metadata) != required:
        raise ValueError("measurement metadata fields do not match the required schema")
    if metadata["batch_size"] != BATCH_SIZE:
        raise ValueError("batch size does not match the shipped measurement configuration")
    if metadata["target_sample_rate"] != TARGET_SAMPLE_RATE:
        raise ValueError("target sample rate does not match the aggregate cost anchor")
    if not metadata["public_path_verified"]:
        raise ValueError("measurement did not resolve through the expected installed artifact")
    if not metadata["telemetry_summary_verified"]:
        raise ValueError("enabled measurement did not publish the expected epoch summary")
    if metadata["process_clock_resolution_ns"] > 1_000:
        raise ValueError("process CPU clock resolution is too coarse for this measurement")
    pair_count = metadata["pair_count"]
    telemetry_pairs = report.get("telemetry_pairs", [])
    noise_pairs = report.get("noise_pairs", [])
    if pair_count < MINIMUM_PAIRS or len(telemetry_pairs) != pair_count:
        raise ValueError("telemetry pair count is below the measurement floor")
    if len(noise_pairs) != pair_count:
        raise ValueError("noise pair count does not match the telemetry cells")
    expected_orders = ["enabled-first" if index % 2 == 0 else "disabled-first" for index in range(pair_count)]
    if [pair.get("order") for pair in telemetry_pairs] != expected_orders:
        raise ValueError("telemetry feeder order did not alternate")
    for pair in [*telemetry_pairs, *noise_pairs]:
        if pair["left_checksum"] != pair["right_checksum"]:
            raise ValueError("paired executions did not deliver identical values")
        if min(pair["left_wall_ns"], pair["right_wall_ns"]) < 10_000_000:
            raise ValueError("a measurement half is too short for stable timing")


def evaluate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Evaluate noise indistinguishability and absolute CPU occupancy."""
    validate_report(report)
    wall_penalties = []
    cpu_cost_ns_per_batch = []
    for pair in report["telemetry_pairs"]:
        if pair["order"] == "enabled-first":
            enabled_wall = pair["left_wall_ns"]
            enabled_cpu = pair["left_cpu_ns"]
            disabled_wall = pair["right_wall_ns"]
            disabled_cpu = pair["right_cpu_ns"]
        else:
            disabled_wall = pair["left_wall_ns"]
            disabled_cpu = pair["left_cpu_ns"]
            enabled_wall = pair["right_wall_ns"]
            enabled_cpu = pair["right_cpu_ns"]
        wall_penalties.append((enabled_wall - disabled_wall) / disabled_wall)
        cpu_cost_ns_per_batch.append(
            (enabled_cpu - disabled_cpu) / report["metadata"]["batches_per_half"]
        )
    noise_penalties = []
    for index, pair in enumerate(report["noise_pairs"]):
        sign = 1.0 if index % 2 == 0 else -1.0
        noise_penalties.append(
            sign * (pair["left_wall_ns"] - pair["right_wall_ns"]) / pair["right_wall_ns"]
        )
    mutation = os.environ.get("HYPERLOADER_TELEMETRY_MUTATION")
    if mutation == "inflate-cost":
        wall_penalties = [value + 0.01 for value in wall_penalties]
        cpu_cost_ns_per_batch = [value + 10_000.0 for value in cpu_cost_ns_per_batch]
    wall_interval = _bootstrap_interval(wall_penalties)
    noise_interval = _bootstrap_interval(noise_penalties)
    cpu_interval = _bootstrap_interval(cpu_cost_ns_per_batch)
    noise_floor = max(abs(noise_interval[0]), abs(noise_interval[1]))
    wall_mean = statistics.fmean(wall_penalties)
    batches_per_second = TARGET_SAMPLE_RATE / BATCH_SIZE
    cpu_core_fraction_upper = max(0.0, cpu_interval[1]) * batches_per_second / 1e9
    wall_below_noise = wall_interval[0] <= 0.0 <= wall_interval[1] and abs(wall_mean) <= noise_floor
    cpu_within_budget = cpu_core_fraction_upper <= CPU_CORE_BUDGET
    return {
        "cpu": {
            "budget_core_fraction": CPU_CORE_BUDGET,
            "ci95_ns_per_batch": list(cpu_interval),
            "core_fraction_upper": cpu_core_fraction_upper,
            "mean_ns_per_batch": statistics.fmean(cpu_cost_ns_per_batch),
            "target_batches_per_second": batches_per_second,
            "within_budget": cpu_within_budget,
        },
        "decision": "PASS" if wall_below_noise and cpu_within_budget else "FAIL",
        "wall": {
            "below_noise": wall_below_noise,
            "ci95_penalty": list(wall_interval),
            "mean_penalty": wall_mean,
            "noise_ci95": list(noise_interval),
            "noise_floor": noise_floor,
        },
    }
