"""Validate and summarize native RNG floor measurements from both Spark clusters."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_METRICS = {
    "sample_derivation",
    "native_draw",
    "permutation_131072",
    "permutation_300000",
    "permutation_1000000007",
}
PERMUTATION_LIMIT_NS = 500.0
SAMPLE_LIMIT_NS = 25.0


@dataclass(frozen=True)
class Report:
    """One pinned-core benchmark report."""

    metadata: dict[str, str]
    measurements: dict[str, list[float]]
    frequencies: list[int]
    checksums: list[int]


def parse_report(path: Path) -> Report:
    """Parse one benchmark CSV while retaining verifier-relevant metadata."""
    metadata: dict[str, str] = {}
    measurements: dict[str, list[float]] = {}
    frequencies: list[int] = []
    checksums: list[int] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if row[0] == "meta":
                if len(row) != 3 or row[1] in metadata:
                    raise ValueError("benchmark metadata is malformed or duplicated")
                metadata[row[1]] = row[2]
            elif row[0] == "data":
                if len(row) != 8:
                    raise ValueError("benchmark measurement row has the wrong width")
                measurements.setdefault(row[1], []).append(float(row[5]))
                checksums.append(int(row[6]))
                frequencies.append(int(row[7]))
            elif row[0] != "kind":
                raise ValueError("benchmark report contains an unknown row kind")
    return Report(metadata, measurements, frequencies, checksums)


def validate_report(report: Report, label: str, core: int) -> None:
    """Reject reports that do not prove the pinned measurement assumptions."""
    required = {
        "label",
        "core",
        "cpu_model",
        "governor",
        "max_freq_khz",
        "trials",
        "iterations",
        "warmup_iterations",
        "sample_derivation_blocks",
        "feistel_rounds",
    }
    if set(report.metadata) != required:
        raise ValueError("benchmark metadata fields do not match the required schema")
    if report.metadata["label"] != label or int(report.metadata["core"]) != core:
        raise ValueError("benchmark report came from the wrong cluster or core")
    if report.metadata["governor"] != "performance":
        raise ValueError("benchmark core governor must be performance")
    if report.metadata["sample_derivation_blocks"] != "1":
        raise ValueError("sample derivation must execute one Philox block")
    if report.metadata["feistel_rounds"] != "8":
        raise ValueError("permutation evaluation must execute eight Feistel rounds")
    trials = int(report.metadata["trials"])
    if trials < 10 or int(report.metadata["iterations"]) < 100_000:
        raise ValueError("benchmark repetition count is below the measurement floor")
    if set(report.measurements) != EXPECTED_METRICS:
        raise ValueError("benchmark metrics do not cover every required operation")
    if any(len(values) != trials for values in report.measurements.values()):
        raise ValueError("a benchmark metric is missing one or more trials")
    if not report.checksums or all(checksum == 0 for checksum in report.checksums):
        raise ValueError("benchmark checksums do not prove observable work")
    maximum = int(report.metadata["max_freq_khz"])
    if not report.frequencies or min(report.frequencies) < maximum * 0.9:
        raise ValueError("benchmark core frequency fell outside the pinned window")


def _interval(values: list[float]) -> tuple[float, float]:
    generator = random.Random(0x48595045524C4F41)
    means = []
    for _ in range(10_000):
        means.append(statistics.fmean(generator.choices(values, k=len(values))))
    means.sort()
    return means[249], means[9749]


def summarize(report: Report) -> dict[str, dict[str, float]]:
    """Compute descriptive statistics and deterministic bootstrap intervals."""
    summary = {}
    mutation = os.environ.get("HYPERLOADER_RNG_FLOOR_MUTATION")
    for metric, observed in sorted(report.measurements.items()):
        values = list(observed)
        if mutation == "inflate-sample" and metric == "sample_derivation":
            values = [value * 100.0 for value in values]
        lower, upper = _interval(values)
        summary[metric] = {
            "ci95_lower_ns": lower,
            "ci95_upper_ns": upper,
            "maximum_ns": max(values),
            "mean_ns": statistics.fmean(values),
            "median_ns": statistics.median(values),
            "minimum_ns": min(values),
        }
    return summary


def evaluate_reports(
    performance: Report, efficiency: Report, performance_core: int, efficiency_core: int
) -> dict[str, Any]:
    """Evaluate reference-core bounds and report cross-cluster ratios."""
    validate_report(performance, "perf", performance_core)
    validate_report(efficiency, "eff", efficiency_core)
    perf_summary = summarize(performance)
    eff_summary = summarize(efficiency)
    decisions = {
        "sample_derivation": {
            "limit_ns": SAMPLE_LIMIT_NS,
            "upper_ns": perf_summary["sample_derivation"]["ci95_upper_ns"],
        }
    }
    for metric in sorted(EXPECTED_METRICS):
        if metric.startswith("permutation_"):
            decisions[metric] = {
                "limit_ns": PERMUTATION_LIMIT_NS,
                "upper_ns": perf_summary[metric]["ci95_upper_ns"],
            }
    passed = all(item["upper_ns"] <= item["limit_ns"] for item in decisions.values())
    ratios = {
        metric: eff_summary[metric]["mean_ns"] / perf_summary[metric]["mean_ns"]
        for metric in sorted(EXPECTED_METRICS)
    }
    return {
        "decision": "PASS" if passed else "FAIL",
        "efficiency": eff_summary,
        "efficiency_to_performance_ratio": ratios,
        "performance": perf_summary,
        "reference_core_decisions": decisions,
    }


def main() -> None:
    """Parse two reports, write the summary, and return the gate decision."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--efficiency", type=Path, required=True)
    parser.add_argument("--performance-core", type=int, required=True)
    parser.add_argument("--efficiency-core", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = evaluate_reports(
        parse_report(arguments.performance),
        parse_report(arguments.efficiency),
        arguments.performance_core,
        arguments.efficiency_core,
    )
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if result["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
