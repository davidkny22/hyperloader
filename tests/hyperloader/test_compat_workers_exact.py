"""Installed worker-structure differential for Torch compatibility mode."""

from __future__ import annotations

import unittest

import torch
from hyperloader import DataLoader

from .compat_workers.support import (
    CompletionOrderDataset,
    FixedBatchSampler,
    HungDataset,
    WorkerIterable,
    WorkerViewDataset,
    collate_worker_rows,
    initialize_worker,
    records,
    seeded_generator,
)


class CompatWorkersExactTest(unittest.TestCase):
    """Compare every result-observable worker argument with Torch."""

    def test_map_worker_topology_and_initializer_match(self) -> None:
        expected = records(
            torch.utils.data.DataLoader(
                WorkerViewDataset(),
                batch_size=2,
                num_workers=2,
                generator=seeded_generator(),
                worker_init_fn=initialize_worker,
                prefetch_factor=1,
            )
        )
        candidate = DataLoader(
            WorkerViewDataset(),
            batch_size=2,
            num_workers=2,
            generator=seeded_generator(),
            worker_init_fn=initialize_worker,
            prefetch_factor=1,
            mode="torch-compat",
        )
        try:
            self.assertEqual(records(candidate), expected)
            self.assertEqual({row[1] for row in expected}, {0, 1})
            self.assertEqual({row[6] for row in expected}, {100, 101})
        finally:
            candidate.close()

    def test_in_order_false_completion_structure_matches(self) -> None:
        reference = torch.utils.data.DataLoader(
            CompletionOrderDataset(8),
            batch_size=1,
            num_workers=2,
            generator=seeded_generator(),
            in_order=False,
            prefetch_factor=2,
        )
        candidate = DataLoader(
            CompletionOrderDataset(8),
            batch_size=1,
            num_workers=2,
            generator=seeded_generator(),
            in_order=False,
            prefetch_factor=2,
            mode="torch-compat",
        )
        try:
            self.assertEqual(
                sorted(records(candidate)),
                sorted(records(reference)),
            )
        finally:
            candidate.close()

    def test_persistent_free_run_and_nonpersistent_respawn_match(self) -> None:
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                reference = torch.utils.data.DataLoader(
                    WorkerViewDataset(8),
                    batch_size=2,
                    num_workers=2,
                    generator=seeded_generator(),
                    persistent_workers=persistent,
                )
                candidate = DataLoader(
                    WorkerViewDataset(8),
                    batch_size=2,
                    num_workers=2,
                    generator=seeded_generator(),
                    persistent_workers=persistent,
                    mode="torch-compat",
                )
                try:
                    self.assertEqual(
                        (records(candidate), records(candidate)),
                        (records(reference), records(reference)),
                    )
                finally:
                    candidate.close()

    def test_iterable_lane_identity_and_exhaustion_match(self) -> None:
        expected = records(
            torch.utils.data.DataLoader(
                WorkerIterable(),
                batch_size=2,
                num_workers=2,
                generator=seeded_generator(),
            )
        )
        candidate = DataLoader(
            WorkerIterable(),
            batch_size=2,
            num_workers=2,
            generator=seeded_generator(),
            mode="torch-compat",
        )
        try:
            self.assertEqual(records(candidate), expected)
        finally:
            candidate.close()

    def test_custom_batch_sampler_and_collate_match(self) -> None:
        reference = torch.utils.data.DataLoader(
            WorkerViewDataset(4),
            batch_sampler=FixedBatchSampler(),
            num_workers=2,
            generator=seeded_generator(),
            collate_fn=collate_worker_rows,
        )
        candidate = DataLoader(
            WorkerViewDataset(4),
            batch_sampler=FixedBatchSampler(),
            num_workers=2,
            generator=seeded_generator(),
            collate_fn=collate_worker_rows,
            mode="torch-compat",
        )
        try:
            self.assertEqual(tuple(candidate), tuple(reference))
        finally:
            candidate.close()

    def test_explicit_spawn_context_matches(self) -> None:
        reference = torch.utils.data.DataLoader(
            WorkerViewDataset(4),
            batch_size=2,
            num_workers=2,
            generator=seeded_generator(),
            multiprocessing_context="spawn",
        )
        candidate = DataLoader(
            WorkerViewDataset(4),
            batch_size=2,
            num_workers=2,
            generator=seeded_generator(),
            multiprocessing_context="spawn",
            mode="torch-compat",
        )
        try:
            self.assertEqual(records(candidate), records(reference))
        finally:
            candidate.close()

    def test_timeout_exception_matches(self) -> None:
        reference = torch.utils.data.DataLoader(
            HungDataset(),
            batch_size=1,
            num_workers=2,
            timeout=0.05,
        )
        candidate = DataLoader(
            HungDataset(),
            batch_size=1,
            num_workers=2,
            timeout=0.05,
            mode="torch-compat",
        )
        try:
            with self.assertRaises(RuntimeError) as expected:
                next(iter(reference))
            with self.assertRaises(RuntimeError) as actual:
                next(iter(candidate))
            self.assertEqual(str(actual.exception), str(expected.exception))
        finally:
            candidate.close()

    def test_invalid_argument_topologies_match_torch(self) -> None:
        cases = (
            {"num_workers": 0, "persistent_workers": True},
            {"num_workers": 0, "prefetch_factor": 2},
            {"num_workers": 2, "multiprocessing_context": "not-a-context"},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(Exception) as expected:
                    torch.utils.data.DataLoader(WorkerViewDataset(), **arguments)
                with self.assertRaises(type(expected.exception)):
                    DataLoader(
                        WorkerViewDataset(),
                        mode="torch-compat",
                        **arguments,
                    )


if __name__ == "__main__":
    unittest.main()
