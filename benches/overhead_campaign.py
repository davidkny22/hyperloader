"""Run the preregistered Spark paired overhead campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark_protocol import EnvironmentMetadata, evaluate
from overhead_cell import run_cell
from overhead_environment import cpu_governor, platform_facts, total_llc_bytes
from overhead_results import clock_samples_valid, summarize_splits
from paired_benchmark import decode_observation

REQUESTED_GPU_CLOCK_MHZ = 2400


def _environment(arguments: argparse.Namespace) -> EnvironmentMetadata:
    facts = platform_facts()
    return EnvironmentMetadata(
        captured_at=datetime.now(timezone.utc).isoformat(),
        machine=arguments.machine,
        commit=arguments.commit,
        cpu_governor=cpu_governor(),
        gpu_clock=f"locked-{REQUESTED_GPU_CLOCK_MHZ}MHz",
        cache_regime="warm",
        benchmark_mode=True,
        concurrent_load=False,
        **facts,
    )


def _write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_regime(
    *,
    regime: str,
    output: Path,
    environment: EnvironmentMetadata,
    threshold_percent: float,
    half_seconds: float,
    llc_bytes: int,
    smoke: bool,
) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    observations = []
    cell_path = output / f"{regime}-cells.jsonl"
    for ordinal in range(1 if smoke else 40):
        cell = run_cell(
            ordinal=ordinal,
            regime=regime,
            environment=environment,
            llc_bytes=llc_bytes,
            half_seconds=half_seconds,
        )
        cells.append(cell)
        with cell_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(cell, sort_keys=True) + "\n")
        if not clock_samples_valid(cell["raw"]["clock_samples"]):
            raise RuntimeError(f"{regime} cell {ordinal} has no loaded GPU clock samples")
        if smoke:
            return {"status": "smoke", "cells": 1}
        observations.append(decode_observation(cell))
        decision = evaluate(observations, threshold_percent=threshold_percent)
        _write_json(output / f"{regime}-decision.json", asdict(decision))
        if decision.status != "collect":
            return {
                "decision": asdict(decision),
                "byte_split": summarize_splits(cells),
                "target_status": "pass" if decision.upper_percent < 1.0 else "fail",
                "target_percent": 1.0,
                "hard_ceiling_percent": threshold_percent,
            }
    raise RuntimeError("paired protocol did not terminate at its forty-cell cap")


def main() -> None:
    """Run both phase types and preserve every raw cell before deciding."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--threshold-percent", type=float, default=2.0)
    parser.add_argument("--half-seconds", type=float, default=45.0)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if not arguments.smoke and arguments.half_seconds != 45.0:
        raise ValueError("protocol cells require exact 45-second halves")
    arguments.output.mkdir(parents=True, exist_ok=False)
    environment = _environment(arguments)
    llc_bytes = total_llc_bytes()
    _write_json(arguments.output / "environment.json", asdict(environment))
    _write_json(
        arguments.output / "working-set.json",
        {"llc_bytes": llc_bytes, "llc_multiplier": 8},
    )
    results = {
        regime: _run_regime(
            regime=regime,
            output=arguments.output,
            environment=environment,
            threshold_percent=arguments.threshold_percent,
            half_seconds=arguments.half_seconds,
            llc_bytes=llc_bytes,
            smoke=arguments.smoke,
        )
        for regime in ("compute", "bandwidth")
    }
    _write_json(arguments.output / "summary.json", results)


if __name__ == "__main__":
    main()
