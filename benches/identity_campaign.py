"""Run the three transport-bound identity-dominance cells on Spark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark_protocol import EnvironmentMetadata, evaluate
from identity_cell import WORKLOAD_REGIMES, run_identity_cell
from overhead_campaign import REQUESTED_GPU_CLOCK_MHZ
from overhead_environment import cpu_governor, platform_facts, total_llc_bytes
from overhead_results import clock_samples_valid
from paired_benchmark import decode_observation


def _write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _environment(arguments: argparse.Namespace) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        captured_at=datetime.now(timezone.utc).isoformat(),
        machine=arguments.machine,
        commit=arguments.commit,
        cpu_governor=cpu_governor(),
        gpu_clock=f"locked-{REQUESTED_GPU_CLOCK_MHZ}MHz",
        cache_regime="warm",
        benchmark_mode=True,
        concurrent_load=False,
        **platform_facts(),
    )


def _run_workload(
    *,
    workload: str,
    output: Path,
    environment: EnvironmentMetadata,
    half_seconds: float,
    llc_bytes: int,
    smoke: bool,
) -> dict[str, object]:
    observations = []
    cell_path = output / f"{workload}-cells.jsonl"
    for ordinal in range(1 if smoke else 40):
        cell = run_identity_cell(
            ordinal=ordinal,
            workload=workload,
            environment=environment,
            llc_bytes=llc_bytes,
            half_seconds=half_seconds,
        )
        with cell_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(cell, sort_keys=True) + "\n")
        if not clock_samples_valid(cell["raw"]["clock_samples"]):
            raise RuntimeError(
                f"{workload} cell {ordinal} has no loaded GPU clock samples"
            )
        if smoke:
            return {"status": "smoke", "cells": 1}
        observations.append(decode_observation(cell))
        decision = evaluate(observations, threshold_percent=0.0)
        _write_json(output / f"{workload}-decision.json", asdict(decision))
        if decision.status != "collect":
            return {
                "decision": asdict(decision),
                "systems": {"counterfactual": "torch", "loader": "hyperloader"},
                "criterion": "bootstrap 95% upper penalty below zero",
            }
    raise RuntimeError(f"{workload} did not terminate at its forty-cell cap")


def main() -> None:
    """Run all identity cells and preserve each observation before deciding."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--machine", required=True)
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
        workload: _run_workload(
            workload=workload,
            output=arguments.output,
            environment=environment,
            half_seconds=arguments.half_seconds,
            llc_bytes=llc_bytes,
            smoke=arguments.smoke,
        )
        for workload in WORKLOAD_REGIMES
    }
    _write_json(arguments.output / "summary.json", results)


if __name__ == "__main__":
    main()
