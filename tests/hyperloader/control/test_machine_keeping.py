"""Public iterator routing checks for calibrated native machine keeping."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from hyperloader import DataLoader, HyperConfig
from hyperloader.config import ControlConfig, FactorConfig
from hyperloader.control.machine_keeping import MachineKeepingIterator


class _Keeper:
    def __init__(self, *arguments: object) -> None:
        self.arguments = arguments
        self.gaps: list[int] = []
        self.parked = False
        self.closed = False

    def observe_gap(self, nanoseconds: int) -> None:
        self.gaps.append(nanoseconds)

    def park(self) -> None:
        self.parked = True

    def duty(self) -> float:
        return 0.04

    def close(self) -> None:
        self.closed = True


class MachineKeepingTest(unittest.TestCase):
    """Prove gap gating, native construction, telemetry, and the off refuge."""

    def test_qualifying_gap_constructs_once_and_preserves_output(self) -> None:
        loader = _loader()
        inner = iter((torch.tensor([1]), torch.tensor([2])))
        clock = iter((10, 20, 2_000_020, 2_000_030))
        keeper = _Keeper()
        with (
            mock.patch(
                "hyperloader.control.machine_keeping.time.perf_counter_ns",
                side_effect=lambda: next(clock),
            ),
            mock.patch(
                "hyperloader.control.machine_keeping._consumer_cpus", return_value=(19,)
            ),
            mock.patch(
                "hyperloader.control.machine_keeping._hyperloader._MachineKeeper",
                return_value=keeper,
                create=True,
            ) as factory,
        ):
            iterator = MachineKeepingIterator(loader, inner)
            self.assertEqual(next(iterator).item(), 1)
            self.assertEqual(next(iterator).item(), 2)

        factory.assert_called_once_with((19,), 0.05, 0.05, 1_930_000)
        self.assertEqual(keeper.gaps, [2_000_000])

    def test_public_stats_report_native_duty_and_close_releases_it(self) -> None:
        loader = DataLoader(torch.arange(8), batch_size=2, num_workers=1)
        keeper = _Keeper()
        loader._machine_keeper = keeper

        self.assertEqual(loader.stats()["current"]["machine_keeping_duty"], 0.04)
        loader.close()

        self.assertTrue(keeper.closed)

    def test_off_configuration_never_wraps_the_public_iterator(self) -> None:
        config = HyperConfig(control=ControlConfig(machine_keeping="off"))
        loader = DataLoader(torch.arange(8), batch_size=2, num_workers=1, config=config)
        loader._calibration = _calibration()

        iterator = iter(loader)

        self.assertNotIsInstance(iterator, MachineKeepingIterator)
        loader.close()


def _loader() -> SimpleNamespace:
    return SimpleNamespace(
        _calibration=_calibration(),
        _machine_keeper=None,
        _machine_keeper_cpus=(),
        config=SimpleNamespace(
            factors=FactorConfig(f_warm=0.05),
            control=ControlConfig(),
        ),
    )


def _calibration() -> SimpleNamespace:
    return SimpleNamespace(
        idle_state_tax=SimpleNamespace(
            minimum_gap_nanoseconds=1_930_000,
            warm_duty_fraction=0.05,
        )
    )


if __name__ == "__main__":
    unittest.main()
