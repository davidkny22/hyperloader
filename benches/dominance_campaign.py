"""Run the six-cell provisional dominance matrix on Spark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_protocol import EnvironmentMetadata
from benchmark_protocol.matrix import workload_names
from dominance_cell import run_dominance_cell
from dominance_protocol import (
    DominanceObservation,
    DominanceRun,
    SelectedConfig,
    decide,
)
from dominance_tuning import tune, tuning_budget
from dominance_workloads import make_workload
from overhead_campaign import REQUESTED_GPU_CLOCK_MHZ
from overhead_environment import cpu_governor, platform_facts
from overhead_results import clock_samples_valid


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


def _decode_run(document: dict[str, Any]) -> DominanceRun:
    from benchmark_protocol import TuningBudget

    return DominanceRun(
        system=document["system"],
        reference=document["reference"],
        workload=document["workload"],
        gpu_regime=document["gpu_regime"],
        throughput=float(document["throughput"]),
        duration_seconds=float(document["duration_seconds"]),
        warmed=bool(document["warmed"]),
        selected=SelectedConfig(**document["selected"]),
        tuning=TuningBudget(
            trials=int(document["tuning"]["trials"]),
            wall_seconds=float(document["tuning"]["wall_seconds"]),
            knobs=tuple(document["tuning"]["knobs"]),
        ),
        environment=EnvironmentMetadata(**document["environment"]),
    )


def _decode_observation(document: dict[str, Any]) -> DominanceObservation:
    return DominanceObservation(
        ordinal=int(document["ordinal"]),
        first=_decode_run(document["first"]),
        second=_decode_run(document["second"]),
        uninterrupted=bool(document["uninterrupted"]),
    )


def _compare(
    *,
    reference: str,
    workload: Any,
    selected: dict[str, SelectedConfig],
    environment: EnvironmentMetadata,
    output: Path,
    smoke: bool,
) -> dict[str, Any]:
    path = output / f"{workload.name}-{reference}-cells.jsonl"
    observations = []
    pairs = 1 if smoke else 5
    half_seconds = 2.0 if smoke else 45.0
    budget = tuning_budget(smoke=smoke)
    for ordinal in range(pairs):
        cell = run_dominance_cell(
            ordinal=ordinal,
            reference=reference,
            workload=workload,
            selected=selected,
            tuning=budget,
            environment=environment,
            half_seconds=half_seconds,
        )
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(cell, sort_keys=True) + "\n")
        if not clock_samples_valid(cell["raw"]["clock_samples"]):
            raise RuntimeError(
                f"{workload.name} {reference} cell {ordinal} lacks loaded clocks"
            )
        if not smoke:
            observations.append(_decode_observation(cell))
    if smoke:
        return {"pairs": 1, "status": "smoke"}
    decision = decide(observations)
    document = asdict(decision)
    _write_json(output / f"{workload.name}-{reference}-decision.json", document)
    return document


def summarize_results(
    results: dict[str, dict[str, dict[str, Any]]],
    *,
    workloads: tuple[str, ...],
    references: tuple[str, ...],
    smoke: bool,
) -> dict[str, Any]:
    """Summarize either the complete matrix or an explicitly selected subset."""
    if smoke:
        return {"status": "smoke", "workloads": results}
    passing = sum(
        all(cell["status"] in {"win", "tie"} for cell in comparisons.values())
        for comparisons in results.values()
    )
    full_matrix = workloads == workload_names() and references == ("torch", "spdl")
    required = 5 if full_matrix else len(workloads)
    return {
        "criterion": (
            "win or noise-bounded tie against both references in at least five cells"
            if full_matrix
            else "win or noise-bounded tie for every selected workload and reference"
        ),
        "passing_workloads": passing,
        "required_workloads": required,
        "status": "pass" if passing >= required else "fail",
        "workloads": results,
    }


def main() -> None:
    """Tune equally, compare all cells, and preserve each raw observation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--torchvision-version", required=True)
    parser.add_argument("--workload", action="append", choices=workload_names())
    parser.add_argument("--reference", action="append", choices=("torch", "spdl"))
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    workspace = arguments.output / "workloads"
    workspace.mkdir()
    environment = _environment(arguments)
    _write_json(arguments.output / "environment.json", asdict(environment))

    workloads = tuple(arguments.workload or workload_names())
    references = tuple(arguments.reference or ("torch", "spdl"))
    systems = ("hyperloader", *references)
    results = {}
    for name in workloads:
        workload = make_workload(
            name,
            workspace,
            torchvision_version=arguments.torchvision_version,
        )
        try:
            selected = {}
            tuning_records = {}
            for system in systems:
                selected[system], tuning_records[system] = tune(
                    system, workload, smoke=arguments.smoke
                )
            _write_json(
                arguments.output / f"{name}-tuning.json",
                tuning_records,
            )
            results[name] = {
                reference: _compare(
                    reference=reference,
                    workload=workload,
                    selected={
                        "hyperloader": selected["hyperloader"],
                        reference: selected[reference],
                    },
                    environment=environment,
                    output=arguments.output,
                    smoke=arguments.smoke,
                )
                for reference in references
            }
        finally:
            workload.close()

    summary = summarize_results(
        results,
        workloads=workloads,
        references=references,
        smoke=arguments.smoke,
    )
    _write_json(arguments.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
