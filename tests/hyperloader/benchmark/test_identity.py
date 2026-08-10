"""Identity-dominance workload and equal-tuning checks."""

from __future__ import annotations

import unittest
import importlib
import sys
from pathlib import Path

import torch
from hyperloader.planner import BlackBoxPlan, build_plan

BENCHES = Path(__file__).parents[3] / "benches"
sys.path.insert(0, str(BENCHES))

identity_cell = importlib.import_module("identity_cell")
identity_feeders = importlib.import_module("identity_feeders")
identity_workloads = importlib.import_module("identity_workloads")
overhead_feeders = importlib.import_module("overhead_feeders")

WORKLOAD_REGIMES = identity_cell.WORKLOAD_REGIMES
EQUAL_FRONTIER_DEPTH = identity_feeders.EQUAL_FRONTIER_DEPTH
PREFETCH_FACTOR = identity_feeders.PREFETCH_FACTOR
hyperloader_arguments = identity_feeders.hyperloader_arguments
torch_loader_arguments = identity_feeders.torch_loader_arguments
make_identity_dataset = identity_workloads.make_identity_dataset
BATCH_SIZE = overhead_feeders.BATCH_SIZE
WORKERS = overhead_feeders.WORKERS


class IdentityHarnessTest(unittest.TestCase):
    """Verify pinned datasets, routing inputs, and tuning equivalence."""

    def test_three_workloads_collate_to_identical_dense_batches(self) -> None:
        batches = []
        for workload in WORKLOAD_REGIMES:
            dataset = make_identity_dataset(workload, 1)
            self.assertEqual(len(dataset), BATCH_SIZE)
            batches.append(torch.utils.data.default_collate(list(dataset)))

        for batch in batches:
            self.assertEqual(batch.shape, (64, 512))
            self.assertEqual(batch.dtype, torch.int64)
            self.assertTrue(torch.equal(batch, batches[0]))

    def test_unknown_workload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown identity workload"):
            make_identity_dataset("unknown", 1)

    def test_all_identity_workloads_select_the_black_box_refuge(self) -> None:
        for workload in WORKLOAD_REGIMES:
            plan = build_plan(make_identity_dataset(workload, 1), False)
            self.assertIsInstance(plan, BlackBoxPlan)

    def test_prefetch_budget_is_equal_in_sample_coordinates(self) -> None:
        torch_arguments = torch_loader_arguments()
        hyper_arguments = hyperloader_arguments()

        for name in (
            "batch_size",
            "num_workers",
            "prefetch_factor",
            "persistent_workers",
            "worker_init_fn",
            "multiprocessing_context",
        ):
            self.assertEqual(torch_arguments[name], hyper_arguments[name])
        self.assertEqual(torch_arguments["num_workers"], WORKERS)
        self.assertEqual(torch_arguments["prefetch_factor"], PREFETCH_FACTOR)
        self.assertTrue(torch_arguments["persistent_workers"])
        self.assertEqual(torch_arguments["multiprocessing_context"], "forkserver")
        self.assertEqual(
            hyper_arguments["config"].scheduler.frontier_depth,
            EQUAL_FRONTIER_DEPTH,
        )


if __name__ == "__main__":
    unittest.main()
