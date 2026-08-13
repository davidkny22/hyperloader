"""Runtime machine-state controls recorded by live-training cells."""

from __future__ import annotations

import argparse


def add_machine_state_arguments(parser: argparse.ArgumentParser) -> None:
    """Add runtime-resolved machine-state controls to a point parser."""
    parser.add_argument(
        "--machine-state-control",
        choices=("none", "native-alu-pulse"),
        default="none",
    )
    parser.add_argument("--machine-state-cpu", type=int, action="append", default=[])
    parser.add_argument("--machine-state-active-us", type=int, default=0)
    parser.add_argument("--machine-state-period-us", type=int, default=0)


def machine_state_environment_fields(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Validate and normalize one caller-supplied machine-state record."""
    control = str(arguments.machine_state_control)
    cpus = tuple(arguments.machine_state_cpu)
    active = int(arguments.machine_state_active_us)
    period = int(arguments.machine_state_period_us)
    if control == "none":
        if cpus or active != 0 or period != 0:
            raise ValueError(
                "disabled machine-state control cannot name pulse settings"
            )
    elif (
        not cpus
        or len(set(cpus)) != len(cpus)
        or any(cpu < 0 for cpu in cpus)
        or active <= 0
        or active > period
    ):
        raise ValueError(
            "native machine-state control requires valid unique CPUs and duty"
        )
    return {
        "machine_state_control": control,
        "machine_state_cpus": cpus,
        "machine_state_active_microseconds": active,
        "machine_state_period_microseconds": period,
    }
