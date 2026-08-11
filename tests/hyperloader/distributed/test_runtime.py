"""Distributed runtime topology capture and validation."""

from __future__ import annotations

import unittest
from unittest import mock

from hyperloader import AUTO, DataLoader, HyperConfig
from hyperloader.config import SchedulerConfig
from hyperloader.distributed.runtime import (
    CapturedTopology,
    capture_topology,
    validate_runtime_topology,
)
from torch import distributed


class RangeDataset:
    """Return deterministic map values for public-loader topology checks."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> int:
        return index


class RuntimeTopologyTest(unittest.TestCase):
    """Keep construction topology stable through iteration."""

    def test_auto_defaults_to_single_rank_before_runtime_initialization(self) -> None:
        with (
            mock.patch.object(distributed, "is_available", return_value=True),
            mock.patch.object(distributed, "is_initialized", return_value=False),
        ):
            topology = capture_topology(AUTO, AUTO)

        self.assertEqual(topology, CapturedTopology(0, 1, False, False))

    def test_auto_captures_an_initialized_runtime(self) -> None:
        with (
            mock.patch.object(distributed, "is_available", return_value=True),
            mock.patch.object(distributed, "is_initialized", return_value=True),
            mock.patch.object(distributed, "get_rank", return_value=3),
            mock.patch.object(distributed, "get_world_size", return_value=8),
        ):
            topology = capture_topology(AUTO, AUTO)

        self.assertEqual(topology, CapturedTopology(3, 8, True, True))

    def test_explicit_topology_does_not_require_a_process_group(self) -> None:
        self.assertEqual(
            capture_topology(2, 4),
            CapturedTopology(2, 4, True, False),
        )
        with self.assertRaisesRegex(TypeError, "both be explicit or both be auto"):
            capture_topology(AUTO, 4)

    def test_late_runtime_mismatch_names_both_topologies(self) -> None:
        topology = CapturedTopology(0, 1, False, False)
        with (
            mock.patch.object(distributed, "is_available", return_value=True),
            mock.patch.object(distributed, "is_initialized", return_value=True),
            mock.patch.object(distributed, "get_rank", return_value=1),
            mock.patch.object(distributed, "get_world_size", return_value=2),
            self.assertRaisesRegex(
                RuntimeError,
                "captured rank=0, world_size=1; current rank=1, world_size=2",
            ),
        ):
            validate_runtime_topology(topology)

    def test_matching_or_absent_runtime_is_accepted(self) -> None:
        topology = CapturedTopology(1, 2, True, True)
        with (
            mock.patch.object(distributed, "is_available", return_value=True),
            mock.patch.object(distributed, "is_initialized", return_value=True),
            mock.patch.object(distributed, "get_rank", return_value=1),
            mock.patch.object(distributed, "get_world_size", return_value=2),
        ):
            validate_runtime_topology(topology)
        with mock.patch.object(distributed, "is_available", return_value=False):
            validate_runtime_topology(topology)

    def test_public_loader_uses_runtime_topology_for_placement_and_identity(
        self,
    ) -> None:
        with (
            mock.patch.object(distributed, "is_available", return_value=True),
            mock.patch.object(distributed, "is_initialized", return_value=True),
            mock.patch.object(distributed, "get_rank", return_value=1),
            mock.patch.object(distributed, "get_world_size", return_value=2),
        ):
            loader = DataLoader(
                RangeDataset(),
                batch_size=2,
                num_workers=1,
                thread_safe=True,
                config=HyperConfig(scheduler=SchedulerConfig(profile_cache="off")),
            )
        try:
            values = {
                element.path: element.value for element in loader._fingerprint.elements
            }
            self.assertEqual(loader._map_placement.rank, 1)
            self.assertEqual(loader._map_placement.world_size, 2)
            self.assertEqual(values["placement.B_g"], 4)
        finally:
            loader.close()

    def test_public_iteration_rejects_late_process_group_initialization(self) -> None:
        with (
            mock.patch.object(distributed, "is_available", return_value=True),
            mock.patch.object(distributed, "is_initialized", return_value=False),
        ):
            loader = DataLoader(
                RangeDataset(),
                batch_size=1,
                num_workers=1,
                thread_safe=True,
            )
        try:
            with (
                mock.patch.object(distributed, "is_available", return_value=True),
                mock.patch.object(distributed, "is_initialized", return_value=True),
                mock.patch.object(distributed, "get_rank", return_value=0),
                mock.patch.object(distributed, "get_world_size", return_value=2),
                self.assertRaisesRegex(
                    RuntimeError,
                    "Construct a fresh DataLoader after process-group initialization",
                ),
            ):
                iter(loader)
        finally:
            loader.close()


if __name__ == "__main__":
    unittest.main()
