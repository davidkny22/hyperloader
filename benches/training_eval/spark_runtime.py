"""Shared runtime command construction for guarded Spark training cells."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_spark_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add caller-resolved Spark controls shared by every training cell."""
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--lease-token", required=True)
    parser.add_argument("--clock-mhz", type=int, required=True)
    parser.add_argument("--accelerator-clock", required=True)
    parser.add_argument("--memory-clock", required=True)
    parser.add_argument("--cpu-governor", required=True)
    parser.add_argument("--power-profile", required=True)
    parser.add_argument("--ambient-probe-id", required=True)
    parser.add_argument("--cpu-set", required=True)
    parser.add_argument("--pythonpath", required=True)
    parser.add_argument("--spinner-library", type=Path, required=True)
    parser.add_argument("--machine-state-cpu", type=int, action="append", required=True)
    parser.add_argument("--machine-state-active-us", type=int, required=True)
    parser.add_argument("--machine-state-period-us", type=int, required=True)
    parser.add_argument("--half-seconds", type=float, default=45.0)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--max-half-width-percent", type=float, default=0.15)
    parser.add_argument("--threshold-percent", type=float, default=100.0)
    parser.add_argument("--bank-batches", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--tuning-candidate", action="append", required=True)
    parser.add_argument("--tuning-seconds", type=float, default=2.0)
    parser.add_argument("--tuning-warmup-steps", type=int, default=3)


def guarded_point_command(
    arguments: argparse.Namespace,
    *,
    output: Path,
    module: str,
    point_arguments: tuple[str, ...],
) -> list[str]:
    """Build one clocked and machine-state-matched public point command."""
    command = [
        sys.executable,
        "-m",
        "benches.spark_clock_guard",
        "--evidence",
        str(output.with_name(output.name + "-clock.json")),
        "--clock-mhz",
        str(arguments.clock_mhz),
        sys.executable,
        "-m",
        "benches.spark_machine_state_guard",
        "--evidence",
        str(output / "machine-state.json"),
        "--spinner-library",
        str(arguments.spinner_library),
    ]
    for core in arguments.machine_state_cpu:
        command.extend(("--core", str(core)))
    command.extend(
        (
            "--active-us",
            str(arguments.machine_state_active_us),
            "--period-us",
            str(arguments.machine_state_period_us),
            "taskset",
            "-c",
            arguments.cpu_set,
            "env",
            f"PYTHONPATH={arguments.pythonpath}",
            sys.executable,
            "-m",
            module,
            *point_arguments,
            "--evaluation-id",
            arguments.evaluation_id,
            "--output",
            str(output),
            "--machine",
            arguments.machine,
            "--commit",
            arguments.commit,
            "--lease-token",
            arguments.lease_token,
            "--accelerator-clock",
            arguments.accelerator_clock,
            "--memory-clock",
            arguments.memory_clock,
            "--cpu-governor",
            arguments.cpu_governor,
            "--power-profile",
            arguments.power_profile,
            "--ambient-probe-id",
            arguments.ambient_probe_id,
            "--half-seconds",
            str(arguments.half_seconds),
            "--min-pairs",
            str(arguments.min_pairs),
            "--max-pairs",
            str(arguments.max_pairs),
            "--max-half-width-percent",
            str(arguments.max_half_width_percent),
            "--threshold-percent",
            str(arguments.threshold_percent),
            "--bank-batches",
            str(arguments.bank_batches),
            "--warmup-steps",
            str(arguments.warmup_steps),
            "--bootstrap-draws",
            str(arguments.bootstrap_draws),
            "--tuning-seconds",
            str(arguments.tuning_seconds),
            "--tuning-warmup-steps",
            str(arguments.tuning_warmup_steps),
            "--machine-state-control",
            "native-alu-pulse",
        )
    )
    for candidate in arguments.tuning_candidate:
        command.extend(("--tuning-candidate", candidate))
    for core in arguments.machine_state_cpu:
        command.extend(("--machine-state-cpu", str(core)))
    command.extend(
        (
            "--machine-state-active-us",
            str(arguments.machine_state_active_us),
            "--machine-state-period-us",
            str(arguments.machine_state_period_us),
        )
    )
    return command
