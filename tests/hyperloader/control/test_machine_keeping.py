"""Public iterator routing checks for calibrated native machine keeping."""

from __future__ import annotations

import unittest
from collections.abc import Iterator
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
        self.deferred: list[int] = []
        self.closed = False

    def observe_gap(self, nanoseconds: int) -> None:
        self.gaps.append(nanoseconds)

    def park(self) -> None:
        self.parked = True

    def defer_park(self, nanoseconds: int) -> None:
        self.deferred.append(nanoseconds)

    def duty(self) -> float:
        return 0.04

    def close(self) -> None:
        self.closed = True


class _Route:
    def __init__(self, *routes: tuple[int, ...]) -> None:
        self._routes = iter(routes)
        self._last: tuple[int, ...] = ()

    def refresh(self) -> tuple[int, ...]:
        self._last = next(self._routes, self._last)
        return self._last


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
                "hyperloader.control.machine_keeping._current_cpu", return_value=19
            ),
            mock.patch(
                "hyperloader.control.machine_keeping.AcceleratorInterruptRoute.discover",
                return_value=_Route((0,)),
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

        factory.assert_called_once_with((0, 19), 0.05, 0.05, 1_930_000)
        self.assertEqual(keeper.gaps, [2_000_000])

    def test_public_stats_report_native_duty_and_close_releases_it(self) -> None:
        loader = DataLoader(torch.arange(8), batch_size=2, num_workers=1)
        keeper = _Keeper()
        loader._machine_keeper = keeper

        self.assertEqual(loader.stats()["current"]["machine_keeping_duty"], 0.04)
        loader.close()

        self.assertTrue(keeper.closed)

    def test_controller_cadence_refreshes_interrupt_and_consumer_cores(self) -> None:
        loader = _loader()
        loader._machine_keeping_last_delivery_ns = 1
        loader.config = SimpleNamespace(
            factors=FactorConfig(f_cad_s=1e-9, f_cad_b=1),
            control=ControlConfig(),
        )
        first_keeper = _Keeper()
        second_keeper = _Keeper()
        clock = iter((2_000_001, 2_000_010, 4_000_020, 4_000_030))
        with (
            mock.patch(
                "hyperloader.control.machine_keeping.time.perf_counter_ns",
                side_effect=lambda: next(clock),
            ),
            mock.patch(
                "hyperloader.control.machine_keeping._current_cpu",
                side_effect=(19, 19, 18),
            ),
            mock.patch(
                "hyperloader.control.machine_keeping.AcceleratorInterruptRoute.discover",
                return_value=_Route((0,), (12,)),
            ),
            mock.patch(
                "hyperloader.control.machine_keeping._hyperloader._MachineKeeper",
                side_effect=(first_keeper, second_keeper),
                create=True,
            ) as factory,
        ):
            iterator = MachineKeepingIterator(
                loader, iter((torch.tensor([1]), torch.tensor([2])))
            )
            self.assertEqual(next(iterator).item(), 1)
            self.assertEqual(next(iterator).item(), 2)

        self.assertEqual(
            [call.args[0] for call in factory.call_args_list], [(0, 19), (12, 18)]
        )
        self.assertTrue(first_keeper.closed)
        self.assertEqual(second_keeper.gaps, [2_000_010])

    def test_completed_iterator_rearms_before_the_next_epoch_first_batch(self) -> None:
        loader = _loader()
        keeper = _Keeper()
        clock = iter((10, 20, 2_000_020, 4_000_020, 4_000_030))
        with (
            mock.patch(
                "hyperloader.control.machine_keeping.time.perf_counter_ns",
                side_effect=lambda: next(clock),
            ),
            mock.patch(
                "hyperloader.control.machine_keeping._current_cpu", return_value=19
            ),
            mock.patch(
                "hyperloader.control.machine_keeping.AcceleratorInterruptRoute.discover",
                return_value=_Route((0,), (0,)),
            ),
            mock.patch(
                "hyperloader.control.machine_keeping._hyperloader._MachineKeeper",
                return_value=keeper,
                create=True,
            ) as factory,
        ):
            first = MachineKeepingIterator(loader, iter((torch.tensor([1]),)))
            self.assertEqual(next(first).item(), 1)
            with self.assertRaises(StopIteration):
                next(first)
            second = MachineKeepingIterator(loader, iter((torch.tensor([2]),)))
            self.assertEqual(next(second).item(), 2)

        factory.assert_called_once_with((0, 19), 0.05, 0.05, 1_930_000)
        self.assertEqual(keeper.gaps, [2_000_000, 4_000_000])
        self.assertEqual(keeper.deferred, [6_000_000_000])
        self.assertFalse(keeper.parked)
        self.assertEqual(loader._machine_keeping_last_delivery_ns, 4_000_030)

    def test_gapless_cadence_parks_only_after_controller_hysteresis(self) -> None:
        loader = _loader()
        loader.config = SimpleNamespace(
            factors=FactorConfig(f_cad_s=1e-9, f_cad_b=2, hysteresis=2),
            control=ControlConfig(),
        )
        keeper = _Keeper()
        loader._machine_keeper = keeper
        clock = iter((10, 20, 120, 130, 230, 240, 340, 350, 450, 460))
        with (
            mock.patch(
                "hyperloader.control.machine_keeping.time.perf_counter_ns",
                side_effect=lambda: next(clock),
            ),
            mock.patch(
                "hyperloader.control.machine_keeping._current_cpu", return_value=None
            ),
            mock.patch(
                "hyperloader.control.machine_keeping.AcceleratorInterruptRoute.discover",
                return_value=None,
            ),
        ):
            iterator = MachineKeepingIterator(
                loader,
                iter(torch.tensor([index]) for index in range(5)),
            )
            for index in range(4):
                self.assertEqual(next(iterator).item(), index)
                self.assertFalse(keeper.parked)
            self.assertEqual(next(iterator).item(), 4)

        self.assertTrue(keeper.parked)

    def test_gapless_cadence_waits_for_time_and_batch_limits(self) -> None:
        loader = _loader()
        loader.config = SimpleNamespace(
            factors=FactorConfig(f_cad_s=1.0, f_cad_b=1, hysteresis=2),
            control=ControlConfig(),
        )
        keeper = _Keeper()
        loader._machine_keeper = keeper
        loader._machine_keeping_last_delivery_ns = 1
        clock = iter(
            (
                10,
                999_000_020,
                1_000_000_020,
                1_999_000_020,
                2_000_000_021,
                2_000_000_030,
            )
        )
        with (
            mock.patch(
                "hyperloader.control.machine_keeping.time.perf_counter_ns",
                side_effect=lambda: next(clock),
            ),
            mock.patch(
                "hyperloader.control.machine_keeping._current_cpu", return_value=None
            ),
            mock.patch(
                "hyperloader.control.machine_keeping.AcceleratorInterruptRoute.discover",
                return_value=None,
            ),
        ):
            iterator = MachineKeepingIterator(
                loader,
                iter(torch.tensor([index]) for index in range(3)),
            )
            for index in range(2):
                self.assertEqual(next(iterator).item(), index)
                self.assertFalse(keeper.parked)
            self.assertEqual(next(iterator).item(), 2)

        self.assertTrue(keeper.parked)

    def test_off_configuration_never_wraps_the_public_iterator(self) -> None:
        config = HyperConfig(control=ControlConfig(machine_keeping="off"))
        loader = DataLoader(torch.arange(8), batch_size=2, num_workers=1, config=config)
        loader._calibration = _calibration()

        iterator = iter(loader)

        self.assertNotIsInstance(iterator, MachineKeepingIterator)
        loader.close()

    def test_user_exception_parks_without_a_rollover_grace(self) -> None:
        loader = _loader()
        keeper = _Keeper()
        loader._machine_keeper = keeper

        def failing_iterator() -> Iterator[torch.Tensor]:
            raise ValueError("boom")
            yield torch.tensor([0])

        iterator = MachineKeepingIterator(loader, failing_iterator())
        with self.assertRaisesRegex(ValueError, "boom"):
            next(iterator)

        self.assertTrue(keeper.parked)
        self.assertEqual(keeper.deferred, [])


def _loader() -> SimpleNamespace:
    return SimpleNamespace(
        _calibration=_calibration(),
        _machine_keeper=None,
        _machine_keeper_cpus=(),
        _machine_keeper_interrupt_cpus=(),
        _machine_keeper_consumer_cpu=None,
        _machine_keeping_last_delivery_ns=0,
        config=SimpleNamespace(
            factors=FactorConfig(f_warm=0.05),
            control=ControlConfig(),
        ),
    )


def _calibration() -> SimpleNamespace:
    return SimpleNamespace(
        staged_copy_tax=None,
        idle_state_tax=SimpleNamespace(
            minimum_gap_nanoseconds=1_930_000,
            warm_duty_fraction=0.05,
        ),
    )


if __name__ == "__main__":
    unittest.main()
