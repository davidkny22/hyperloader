"""Controller objective, cadence, clipping, and hysteresis checks."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from hyperloader.control import (
    AdaptiveController,
    ControllerObjective,
    CpuCluster,
    MachineIdentity,
    resolve_calibration,
    save_calibration,
)
from tests.hyperloader.control.test_record import _record


def _controller() -> AdaptiveController:
    return AdaptiveController(
        width_ceiling=4,
        cadence_seconds=1.0,
        cadence_batches=2,
        step_clip=1,
        shrink_hysteresis=3,
        objective=ControllerObjective(None),
    )


class AdaptiveControllerTest(unittest.TestCase):
    """Exercise lexicographic and time-bounded width motion."""

    def test_starvation_outranks_smaller_resource_score(self) -> None:
        objective = ControllerObjective(None)

        fed = objective.score(
            starvation=False,
            width=4,
            work_shape="compute",
            cluster="all",
            bytes_per_second=0.0,
        )
        starved = objective.score(
            starvation=True,
            width=1,
            work_shape="compute",
            cluster="all",
            bytes_per_second=0.0,
        )

        self.assertLess(fed, starved)

    def test_objective_interpolates_compute_and_bandwidth_curves(self) -> None:
        machine = MachineIdentity("controller-cpu", (CpuCluster("all", (0,)),), 4096)
        objective = ControllerObjective(_record(machine))

        score = objective.score(
            starvation=False,
            width=3,
            work_shape="compute",
            cluster="all",
            bytes_per_second=500_000.0,
            memory_penalty=0.002,
        )

        self.assertEqual(score[0], 0)
        self.assertAlmostEqual(score[1], 0.02)

    def test_shrink_requires_three_complete_cadences_and_clips_one(self) -> None:
        controller = _controller()
        now = 1_000_000_000
        decisions = []
        for cadence in range(3):
            self.assertIsNone(
                controller.observe(
                    now_ns=now + cadence * 1_000_000_000,
                    stalled=False,
                    occupied=4,
                    batch_size=1,
                )
            )
            decision = controller.observe(
                now_ns=now + (cadence + 1) * 1_000_000_000,
                stalled=False,
                occupied=4,
                batch_size=1,
            )
            decisions.append(decision)

        self.assertEqual([decision.width for decision in decisions], [4, 4, 3])

    def test_stall_unparks_one_route_at_the_next_cadence(self) -> None:
        controller = _controller()
        controller.width = 2
        controller.observe(now_ns=1_000_000_000, stalled=True, occupied=4, batch_size=1)

        decision = controller.observe(
            now_ns=2_000_000_000,
            stalled=False,
            occupied=4,
            batch_size=1,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.width, 3)
        self.assertEqual(decision.reason, "starvation")

    def test_runtime_loads_the_matching_machine_record(self) -> None:
        machine = MachineIdentity("controller-cpu", (CpuCluster("all", (0,)),), 4096)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "calibration" / f"{machine.cache_key}.json"
            save_calibration(_record(machine), path)
            with (
                mock.patch(
                    "hyperloader.control.runtime.detect_machine_identity",
                    return_value=machine,
                ),
                mock.patch(
                    "hyperloader.control.runtime.user_cache_root", return_value=root
                ),
            ):
                resolved = resolve_calibration()

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.machine.cache_key, machine.cache_key)


if __name__ == "__main__":
    unittest.main()
