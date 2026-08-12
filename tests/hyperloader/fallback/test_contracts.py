"""Pure-Python contract primitives and bounded transport checks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from hyperloader import _hyperloader
from hyperloader.fallback import rng
from hyperloader.fallback.schedule import StaticSchedule
from hyperloader.fallback.telemetry import Telemetry
from hyperloader.fallback.transport import ProcessResources, WorkerEndpoint


class FallbackContractTest(unittest.TestCase):
    """Compare fallback primitives with the installed native contract."""

    def test_sealed_contract_vectors_reproduce_without_native_code(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        vectors = json.loads(
            (repository / "oracles" / "contract-vectors" / "vectors.json").read_text(
                encoding="utf-8"
            )
        )
        for vector in vectors["philox"]["vectors"]:
            self.assertEqual(
                list(
                    rng.rng_block(
                        vector["root_seed"],
                        vector["epoch"],
                        vector["coord"],
                        vector["draw_index"],
                        vector["stream_id"],
                    )
                ),
                vector["words"],
            )
        for vector in vectors["permutations"]:
            permutation = (
                rng.materialized_permutation(
                    vector["root_seed"], vector["epoch"], vector["domain"]
                )
                if vector["regime"] == "materialized"
                else None
            )
            for position, expected in vector["points"]:
                actual = (
                    permutation[position]
                    if permutation is not None
                    else rng.feistel_permute(
                        vector["root_seed"],
                        vector["epoch"],
                        vector["domain"],
                        position,
                    )
                )
                self.assertEqual(actual, expected)
        for vector in vectors["placements"]:
            for rank in vector["ranks"]:
                actual = rng.rank_placements(
                    vector["root_seed"],
                    vector["epoch"],
                    vector["dataset_len"],
                    vector["batch_size"],
                    vector["world_size"],
                    rank["rank"],
                    vector["drop_last"],
                    vector["exact_count"],
                )
                self.assertEqual([list(item) for item in actual], rank["items"])

    def test_rng_permutation_and_placement_match_native(self) -> None:
        for root_seed, epoch, coordinate in ((0, 0, 0), (17, 3, 2**40 + 9)):
            self.assertEqual(
                rng.rng_block(root_seed, epoch, coordinate, 5, 7),
                _hyperloader._rng_block(root_seed, epoch, coordinate, 5, 7),
            )
            self.assertEqual(
                rng.sample_rng_context(root_seed, epoch, coordinate),
                _hyperloader._sample_rng_context(root_seed, epoch, coordinate),
            )
        for domain in (3, 65_536, 131_072, 300_000):
            for position in (0, domain // 2, domain - 1):
                self.assertEqual(
                    rng.permutation_index(29, 4, domain, position),
                    _hyperloader._permutation_index(29, 4, domain, position),
                )
        arguments = (31, 2, 37, 3, 4, 1)
        self.assertEqual(
            rng.rank_placements(*arguments),
            _hyperloader._rank_placements(*arguments),
        )

    def test_schedule_preserves_bounded_strict_and_ready_commits(self) -> None:
        schedule = StaticSchedule(0, 6, 3, 2)
        self.assertEqual(schedule.dispatch_candidates(), [0, 1, 2])
        for position, worker in ((0, 0), (1, 1), (2, 0)):
            schedule.mark_dispatched(position, worker)
        schedule.mark_completed(2, 0)
        self.assertEqual(schedule.try_commit_ready(2), 2)
        self.assertEqual(schedule.delivered_positions(), [2])
        schedule.mark_completed(0, 0)
        self.assertEqual(schedule.try_commit(), 0)
        schedule.mark_completed(1, 1)
        self.assertEqual(schedule.try_commit(), 1)
        self.assertEqual(schedule.delivered_positions(), [])
        self.assertEqual(schedule.dispatch_candidates(), [3, 4, 5])

    def test_shared_arena_round_trip_is_bounded(self) -> None:
        resources = ProcessResources(1, 1, 64, 64)
        endpoint = WorkerEndpoint(*resources.descriptor(0))
        try:
            self.assertTrue(resources.try_submit(2, 7, 11, 0, 0))
            self.assertFalse(resources.try_submit(2, 8, 12, 0, 0))
            command = endpoint.try_recv()
            self.assertIsNotNone(command)
            assert command is not None
            self.assertTrue(endpoint.try_complete_ready(command, b"payload", 19))
            self.assertEqual(resources.try_receive(0), (7, 0, b"payload", 19))
            self.assertTrue(resources.try_submit(2, 8, 12, 0, 0))
        finally:
            endpoint.close()
            resources.close()

    def test_telemetry_keeps_the_stable_summary_shape(self) -> None:
        telemetry = Telemetry()
        self.assertEqual(telemetry.registry(), _hyperloader._Telemetry.registry())
        for latency in (10, 20, 30, 40):
            telemetry.record_delivery(2, 16, latency)
        telemetry.record_stall()
        telemetry.record_controller(2, 1, "ceiling", True, 0.02, "bandwidth")
        current = telemetry.snapshot()["current"]
        self.assertEqual(current["delivered_samples"], 8)
        self.assertEqual(
            current["delivery_latency_ns"], {"p50": 31, "p95": 63, "p99": 63}
        )
        self.assertEqual(current["ceiling_binds"], 1)


if __name__ == "__main__":
    unittest.main()
