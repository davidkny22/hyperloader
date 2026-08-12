"""Independently verify and summarize a completed overhead campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmark_protocol import evaluate
from overhead_results import clock_samples_valid, summarize_splits
from paired_benchmark import decode_observation

REGIMES = ("compute", "bandwidth")
HARD_CEILING_PERCENT = 2.0
TARGET_PERCENT = 1.0
BYTE_CEILING_GBPS = 0.5


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return document


def _read_cells(path: Path) -> list[dict[str, Any]]:
    cells = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not cells or not all(isinstance(cell, dict) for cell in cells):
        raise ValueError(f"{path} must contain JSON objects")
    return cells


def verify_regime(campaign: Path, regime: str) -> dict[str, Any]:
    """Recompute one terminal decision and validate its raw protocol evidence."""
    cells = _read_cells(campaign / f"{regime}-cells.jsonl")
    expected_orders = [
        ("counterfactual", "loader")
        if ordinal % 2 == 0
        else ("loader", "counterfactual")
        for ordinal in range(len(cells))
    ]
    observed_orders = [
        (cell["first"]["system"], cell["second"]["system"]) for cell in cells
    ]
    if observed_orders != expected_orders:
        raise AssertionError(f"{regime} feeder order did not alternate")
    for ordinal, cell in enumerate(cells):
        if cell["ordinal"] != ordinal or cell["uninterrupted"] is not True:
            raise AssertionError(
                f"{regime} cell {ordinal} broke continuity or ordinal order"
            )
        if (
            cell["first"]["duration_seconds"] != 45.0
            or cell["second"]["duration_seconds"] != 45.0
        ):
            raise AssertionError(
                f"{regime} cell {ordinal} did not use 45-second halves"
            )
        if not cell["first"]["warmed"] or not cell["second"]["warmed"]:
            raise AssertionError(f"{regime} cell {ordinal} includes an unwarmed half")
        if not clock_samples_valid(cell["raw"]["clock_samples"]):
            raise AssertionError(
                f"{regime} cell {ordinal} has invalid loaded clock samples"
            )
        resident_bytes = int(cell["raw"]["resident_bytes"])
        llc_bytes = int(cell["raw"]["llc_bytes"])
        if resident_bytes < 8 * llc_bytes:
            raise AssertionError(
                f"{regime} cell {ordinal} did not defeat LLC residency"
            )

    recomputed = evaluate(
        [decode_observation(cell) for cell in cells],
        threshold_percent=HARD_CEILING_PERCENT,
    )
    recorded = _read_json(campaign / f"{regime}-decision.json")
    if asdict(recomputed) != recorded:
        raise AssertionError(f"{regime} terminal decision does not reproduce")
    if recomputed.status != "pass" or recomputed.upper_percent >= TARGET_PERCENT:
        raise AssertionError(f"{regime} does not pass the sub-one-percent target")

    split = summarize_splits(cells)
    if float(split["explicit_overhead_gbps"]) > BYTE_CEILING_GBPS:
        raise AssertionError(f"{regime} exceeds the copied-byte ceiling")
    return {
        "decision": recorded,
        "byte_split": split,
        "cells": len(cells),
        "loaded_clock_min_mhz": min(
            int(sample["clock_mhz"])
            for cell in cells
            for sample in cell["raw"]["clock_samples"]
            if int(sample["utilization_percent"]) > 0
        ),
        "loaded_clock_max_mhz": max(
            int(sample["clock_mhz"])
            for cell in cells
            for sample in cell["raw"]["clock_samples"]
            if int(sample["utilization_percent"]) > 0
        ),
    }


def _validate_controls(
    environment: dict[str, Any],
    guard: dict[str, Any],
    *,
    expected_cpu_governor: str,
) -> int:
    requested_mhz = int(guard.get("requested_mhz", 0))
    if requested_mhz <= 0 or guard.get("command_returncode") != 0:
        raise AssertionError("clock guard did not execute the requested campaign")
    if environment["gpu_clock"] != f"locked-{requested_mhz}MHz":
        raise AssertionError("campaign environment and clock guard disagree")
    if environment["cpu_governor"] != expected_cpu_governor:
        raise AssertionError("campaign did not use the requested CPU governor")
    if not guard.get("reset_stdout"):
        raise AssertionError("clock guard has no reset evidence")
    return requested_mhz


def verify_campaign(
    campaign: Path,
    clock_control: Path,
    *,
    expected_cpu_governor: str,
) -> dict[str, Any]:
    """Verify the guard record and both phase-type terminal decisions."""
    environment = _read_json(campaign / "environment.json")
    guard = _read_json(clock_control)
    requested_mhz = _validate_controls(
        environment,
        guard,
        expected_cpu_governor=expected_cpu_governor,
    )
    return {
        "commit": environment["commit"],
        "machine": environment["machine"],
        "gpu_clock_request_mhz": requested_mhz,
        "clock_reset": True,
        "regimes": {regime: verify_regime(campaign, regime) for regime in REGIMES},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--clock-control", type=Path, required=True)
    parser.add_argument("--expected-cpu-governor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = verify_campaign(
        arguments.campaign,
        arguments.clock_control,
        expected_cpu_governor=arguments.expected_cpu_governor,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
