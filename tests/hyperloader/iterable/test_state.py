"""Stateful iterable snapshot and replay behavior."""

from __future__ import annotations

import random
import unittest
import warnings
from typing import Any

from hyperloader import DataLoader, HyperConfig
from hyperloader.config import FactorConfig
from torch.utils.data import get_worker_info


class StatefulRange:
    """Expose a replayable sharded range with coordinate-bound random output."""

    def __init__(self, stop: int = 24, *, large_lane: int | None = None) -> None:
        self.stop = stop
        self.large_lane = large_lane
        self.position = 0
        self.slot = 0
        self.stride = 1
        self.lane = 0

    def shard(self, rank: int, world: int, lane: int, lanes: int) -> None:
        self.slot = rank * lanes + lane
        self.stride = world * lanes
        self.lane = lane

    def __iter__(self):
        values = range(self.slot, self.stop, self.stride)
        while self.position < len(values):
            value = values[self.position]
            self.position += 1
            yield value, random.getrandbits(32)

    def state_dict(self) -> dict[str, Any]:
        padding = "x" * 2048 if self.large_lane == self.lane else ""
        return {"position": self.position, "padding": padding}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.position = int(state["position"])


class IncompleteProtocol:
    """Supply only one half of the stateful-source protocol."""

    def __iter__(self):
        yield 1

    def state_dict(self) -> dict[str, int]:
        return {"position": 0}


class PlainRange:
    """Partition a replay-incapable range through the logical worker view."""

    def __init__(self, stop: int = 24) -> None:
        self.stop = stop

    def __iter__(self):
        info = get_worker_info()
        lane = 0 if info is None else info.id
        lanes = 1 if info is None else info.num_workers
        for value in range(lane, self.stop, lanes):
            yield value, random.getrandbits(32)


def _rows(batch: Any) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(value), int(bits))
        for value, bits in zip(batch[0].tolist(), batch[1].tolist(), strict=True)
    )


def _stream(loader: DataLoader) -> list[tuple[tuple[int, int], ...]]:
    return [_rows(batch) for batch in loader]


def _loader(
    *,
    cadence: int | str = 1,
    maximum_bytes: int = 4 * 1024 * 1024,
    large_lane: int | None = None,
) -> DataLoader:
    return DataLoader(
        StatefulRange(large_lane=large_lane),
        batch_size=2,
        num_workers=2,
        seed=941,
        config=HyperConfig(
            factors=FactorConfig(
                f_snap=cadence,
                f_snap_bytes=maximum_bytes,
            )
        ),
    )


def _plain_loader() -> DataLoader:
    return DataLoader(
        PlainRange(),
        batch_size=2,
        num_workers=2,
        seed=941,
    )


class IterableStateTest(unittest.TestCase):
    """Keep source replay exact across snapshots and replay fallback."""

    def test_cut_points_restore_identical_remaining_batches(self) -> None:
        baseline_loader = _loader()
        try:
            baseline = _stream(baseline_loader)
        finally:
            baseline_loader.close()

        for cut in (0, 1, 3, 5):
            with self.subTest(cut=cut):
                source = _loader()
                iterator = iter(source)
                prefix = [_rows(next(iterator)) for _ in range(cut)]
                state = source.state_dict()
                source.close()
                resumed = _loader()
                try:
                    resumed.load_state_dict(state)
                    self.assertEqual(prefix + _stream(resumed), baseline)
                finally:
                    resumed.close()

    def test_coarse_and_mixed_snapshot_modes_replay_exactly(self) -> None:
        baseline_loader = _loader(cadence=3, maximum_bytes=256, large_lane=1)
        try:
            baseline = _stream(baseline_loader)
        finally:
            baseline_loader.close()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            source = _loader(cadence=3, maximum_bytes=256, large_lane=1)
            iterator = iter(source)
            prefix = [_rows(next(iterator)) for _ in range(4)]
            state = source.state_dict()
        source.close()
        lanes = state["lanes"]
        self.assertIsNotNone(lanes[0]["snapshot"])
        self.assertIsNone(lanes[1]["snapshot"])
        self.assertLess(
            lanes[0]["snapshot_arrival"], lanes[0]["delivered_arrival"]
        )
        self.assertTrue(any("f_snap_bytes" in str(item.message) for item in caught))

        resumed = _loader(cadence=3, maximum_bytes=256, large_lane=1)
        try:
            resumed.load_state_dict(state)
            self.assertEqual(prefix + _stream(resumed), baseline)
        finally:
            resumed.close()

    def test_snapshot_off_replays_from_arrival_zero(self) -> None:
        baseline_loader = _loader(cadence="off")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                baseline = _stream(baseline_loader)
            finally:
                baseline_loader.close()
            source = _loader(cadence="off")
            iterator = iter(source)
            prefix = [_rows(next(iterator)) for _ in range(4)]
            state = source.state_dict()
            source.close()
            self.assertTrue(all(lane["snapshot"] is None for lane in state["lanes"]))
            resumed = _loader(cadence="off")
            try:
                resumed.load_state_dict(state)
                self.assertEqual(prefix + _stream(resumed), baseline)
            finally:
                resumed.close()

    def test_lane_recovery_replays_to_its_delivered_cursor(self) -> None:
        baseline_loader = _loader(cadence=3)
        try:
            baseline = _stream(baseline_loader)
        finally:
            baseline_loader.close()
        source = _loader(cadence=3)
        iterator = iter(source)
        try:
            prefix = [_rows(next(iterator)) for _ in range(5)]
            iterator.recover_lane(0)
            self.assertEqual(prefix + [_rows(batch) for batch in iterator], baseline)
        finally:
            source.close()

    def test_protocol_requires_both_callable_methods(self) -> None:
        loader = DataLoader(IncompleteProtocol(), batch_size=1, num_workers=1)
        try:
            with self.assertRaisesRegex(TypeError, "state_dict and load_state_dict"):
                iter(loader)
        finally:
            loader.close()

    def test_plain_source_resume_and_recovery_restart_the_full_epoch(self) -> None:
        baseline_loader = _plain_loader()
        try:
            baseline = _stream(baseline_loader)
        finally:
            baseline_loader.close()

        source = _plain_loader()
        iterator = iter(source)
        next(iterator)
        next(iterator)
        state = source.state_dict()
        source.close()
        self.assertTrue(all(not lane["stateful"] for lane in state["lanes"]))

        resumed = _plain_loader()
        try:
            resumed.load_state_dict(state)
            with self.assertWarnsRegex(UserWarning, "restarts its epoch"):
                self.assertEqual(_stream(resumed), baseline)
        finally:
            resumed.close()

        recovered = _plain_loader()
        recovered_iterator = iter(recovered)
        next(recovered_iterator)
        try:
            with self.assertWarnsRegex(UserWarning, "restarts its epoch"):
                recovered_iterator.recover_lane(0)
            self.assertEqual(
                [_rows(batch) for batch in recovered_iterator],
                baseline,
            )
        finally:
            recovered.close()

    def test_plain_lane_checkpoint_rejects_a_source_snapshot(self) -> None:
        source = _plain_loader()
        iterator = iter(source)
        next(iterator)
        state = source.state_dict()
        source.close()
        state["lanes"][0]["snapshot"] = b"invalid"
        state["lanes"][0]["snapshot_arrival"] = 0
        resumed = _plain_loader()
        try:
            with self.assertRaisesRegex(ValueError, "plain iterable lane"):
                resumed.load_state_dict(state)
        finally:
            resumed.close()


if __name__ == "__main__":
    unittest.main()
