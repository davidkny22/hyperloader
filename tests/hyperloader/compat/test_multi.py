"""Torch-compatible process-lane behavior."""

from __future__ import annotations

import re
import unittest

import torch
from hyperloader import DataLoader

from .support import (
    FatalLaneDataset,
    InitializedLaneDataset,
    LaneDataset,
    compat_config,
    generator,
    initialize_lane,
    records,
)


class CompatMultiTest(unittest.TestCase):
    """Match torch assignment, initialization, persistence, and abandonment."""

    def test_mode_default_uses_torchs_nonpersistent_workers(self) -> None:
        candidate = DataLoader(
            LaneDataset(4),
            batch_size=1,
            num_workers=2,
            mode="torch-compat",
        )
        try:
            self.assertFalse(candidate.persistent_workers)
        finally:
            candidate.close()

    def test_worker_stream_and_round_robin_assignment_match_torch(self) -> None:
        reference = torch.utils.data.DataLoader(
            LaneDataset(),
            batch_size=3,
            shuffle=True,
            num_workers=2,
            generator=generator(401),
            persistent_workers=False,
        )
        expected = records(reference)

        candidate = DataLoader(
            LaneDataset(),
            batch_size=3,
            shuffle=True,
            num_workers=2,
            generator=generator(401),
            persistent_workers=False,
            mode="torch-compat",
        )
        try:
            self.assertEqual(records(candidate), expected)
            expected_lanes = ([0] * 3 + [1] * 3) * 4
            self.assertEqual([row[1] for row in expected], expected_lanes)
            self.assertTrue(all(row[4] == 1 for row in expected))
        finally:
            candidate.close()

    def test_persistent_workers_free_run_across_iterations(self) -> None:
        reference = torch.utils.data.DataLoader(
            LaneDataset(12),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator(503),
            persistent_workers=True,
        )
        try:
            expected = (records(reference), records(reference))
        finally:
            del reference

        candidate = DataLoader(
            LaneDataset(12),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator(503),
            persistent_workers=True,
            mode="torch-compat",
        )
        try:
            self.assertEqual((records(candidate), records(candidate)), expected)
        finally:
            candidate.close()

    def test_persistent_abandonment_keeps_torchs_live_lane_state(self) -> None:
        reference = torch.utils.data.DataLoader(
            LaneDataset(12),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator(557),
            persistent_workers=True,
        )
        try:
            first_reference = iter(reference)
            next(first_reference)
            expected = records(iter(reference))
        finally:
            del reference

        candidate = DataLoader(
            LaneDataset(12),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator(557),
            persistent_workers=True,
            mode="torch-compat",
        )
        try:
            first_candidate = iter(candidate)
            next(first_candidate)
            self.assertEqual(records(iter(candidate)), expected)
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                next(first_candidate)
        finally:
            candidate.close()

    def test_worker_initializer_observes_the_original_lane_dataset(self) -> None:
        reference = torch.utils.data.DataLoader(
            InitializedLaneDataset(),
            batch_size=2,
            num_workers=2,
            generator=generator(811),
            worker_init_fn=initialize_lane,
            persistent_workers=False,
        )
        expected = records(reference)

        candidate = DataLoader(
            InitializedLaneDataset(),
            batch_size=2,
            num_workers=2,
            generator=generator(811),
            worker_init_fn=initialize_lane,
            persistent_workers=False,
            mode="torch-compat",
            config=compat_config(),
        )
        try:
            self.assertEqual(records(candidate), expected)
            self.assertEqual({row[-1] for row in expected}, {100, 101})
        finally:
            candidate.close()

    def test_worker_death_is_fatal_in_snapshot_mode(self) -> None:
        reference = torch.utils.data.DataLoader(
            FatalLaneDataset(),
            batch_size=1,
            num_workers=2,
            timeout=2,
            persistent_workers=False,
        )
        with self.assertRaises(RuntimeError) as reference_error:
            next(iter(reference))

        candidate = DataLoader(
            FatalLaneDataset(),
            batch_size=1,
            num_workers=2,
            timeout=2,
            persistent_workers=False,
            mode="torch-compat",
            config=compat_config(),
        )
        try:
            with self.assertRaises(RuntimeError) as candidate_error:
                next(iter(candidate))
            self.assertEqual(
                re.sub(
                    r"pid\(s\) [^)]*",
                    "pid(s) <pid>",
                    str(candidate_error.exception),
                ),
                re.sub(
                    r"pid\(s\) [^)]*",
                    "pid(s) <pid>",
                    str(reference_error.exception),
                ),
            )
        finally:
            candidate.close()


if __name__ == "__main__":
    unittest.main()
