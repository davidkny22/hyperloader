"""Training machine-state configuration behavior."""

from __future__ import annotations

import argparse

import pytest

from benches.training_eval.machine_state import machine_state_environment_fields


def _arguments(**changes: object) -> argparse.Namespace:
    values = {
        "machine_state_control": "none",
        "machine_state_cpu": [],
        "machine_state_active_us": 0,
        "machine_state_period_us": 0,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_native_pulse_fields_preserve_runtime_controls() -> None:
    fields = machine_state_environment_fields(
        _arguments(
            machine_state_control="native-alu-pulse",
            machine_state_cpu=[2, 3],
            machine_state_active_us=4,
            machine_state_period_us=400,
        )
    )

    assert fields == {
        "machine_state_control": "native-alu-pulse",
        "machine_state_cpus": (2, 3),
        "machine_state_active_microseconds": 4,
        "machine_state_period_microseconds": 400,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        _arguments(machine_state_cpu=[2]),
        _arguments(
            machine_state_control="native-alu-pulse",
            machine_state_cpu=[2, 2],
            machine_state_active_us=4,
            machine_state_period_us=400,
        ),
        _arguments(
            machine_state_control="native-alu-pulse",
            machine_state_cpu=[2],
            machine_state_active_us=401,
            machine_state_period_us=400,
        ),
    ],
)
def test_incoherent_machine_state_fields_are_rejected(
    arguments: argparse.Namespace,
) -> None:
    with pytest.raises(ValueError, match="machine-state control"):
        machine_state_environment_fields(arguments)
