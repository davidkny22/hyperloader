"""Opt-in continuation for torch-compatible process lanes."""

from __future__ import annotations

import unittest

import torch
from hyperloader import DataLoader

from .support import (
    IterableLaneDataset,
    LaneDataset,
    compat_config,
    generator,
    records,
)


class CompatMultiStateTest(unittest.TestCase):
    """Restore the first undelivered batch on each free-running lane."""

    def test_snapshot_ring_restores_the_first_undelivered_batch_per_lane(self) -> None:
        reference = torch.utils.data.DataLoader(
            LaneDataset(),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator(607),
            persistent_workers=False,
        )
        expected = records(reference)

        source = DataLoader(
            LaneDataset(),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator(607),
            persistent_workers=False,
            mode="torch-compat",
            config=compat_config(),
        )
        iterator = iter(source)
        prefix = []
        for _ in range(3):
            batch = next(iterator)
            prefix.extend(
                tuple(int(value) for value in row)
                for row in zip(*(column.tolist() for column in batch), strict=True)
            )
        state = source.state_dict()
        self.assertEqual(source.state_dict(), state)
        self.assertGreater(state["sampler_position"], state["delivered_cursor"])
        source.close()

        resumed = DataLoader(
            LaneDataset(),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator(607),
            persistent_workers=False,
            mode="torch-compat",
            config=compat_config(),
        )
        try:
            resumed.load_state_dict(state)
            self.assertEqual(prefix + records(resumed), expected)
            self.assertEqual(state["assignment_phase"], 1)
            self.assertEqual(set(state["lane_states"]), {0, 1})
        finally:
            resumed.close()

    def test_multi_worker_state_is_opt_in_and_requires_the_same_width(self) -> None:
        disabled = DataLoader(
            LaneDataset(4),
            batch_size=1,
            num_workers=2,
            persistent_workers=False,
            mode="torch-compat",
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "compat_resume='on'"):
                disabled.state_dict()
        finally:
            disabled.close()

        enabled = DataLoader(
            LaneDataset(4),
            batch_size=1,
            num_workers=2,
            persistent_workers=False,
            mode="torch-compat",
            config=compat_config(),
        )
        try:
            state = enabled.state_dict()
        finally:
            enabled.close()
        changed = DataLoader(
            LaneDataset(4),
            batch_size=1,
            num_workers=3,
            persistent_workers=False,
            mode="torch-compat",
            config=compat_config(),
        )
        try:
            with self.assertRaisesRegex(ValueError, "same num_workers"):
                changed.load_state_dict(state)
        finally:
            changed.close()

    def test_snapshot_ring_restores_a_reused_persistent_epoch(self) -> None:
        reference = torch.utils.data.DataLoader(
            LaneDataset(16),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator(701),
            persistent_workers=True,
        )
        try:
            records(reference)
            expected = records(reference)
        finally:
            del reference

        source = DataLoader(
            LaneDataset(16),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator(701),
            persistent_workers=True,
            mode="torch-compat",
            config=compat_config(),
        )
        records(source)
        iterator = iter(source)
        prefix = []
        for _ in range(3):
            batch = next(iterator)
            prefix.extend(
                tuple(int(value) for value in row)
                for row in zip(*(column.tolist() for column in batch), strict=True)
            )
        state = source.state_dict()
        source.close()

        resumed = DataLoader(
            LaneDataset(16),
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator(701),
            persistent_workers=True,
            mode="torch-compat",
            config=compat_config(),
        )
        try:
            resumed.load_state_dict(state)
            self.assertEqual(prefix + records(resumed), expected)
        finally:
            resumed.close()

    def test_iterable_resume_rejects_cross_sample_source_state(self) -> None:
        candidate = DataLoader(
            IterableLaneDataset(),
            batch_size=2,
            num_workers=2,
            persistent_workers=False,
            mode="torch-compat",
            config=compat_config(),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "map-style dataset"):
                candidate.state_dict()
        finally:
            candidate.close()

    def test_out_of_order_resume_rejects_an_ambiguous_delivered_cursor(self) -> None:
        candidate = DataLoader(
            LaneDataset(8),
            batch_size=2,
            num_workers=2,
            persistent_workers=False,
            in_order=False,
            mode="torch-compat",
            config=compat_config(),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "in_order=True"):
                candidate.state_dict()
        finally:
            candidate.close()


if __name__ == "__main__":
    unittest.main()
