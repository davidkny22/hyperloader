"""Installed public gate for exact stateful iterable source replay."""

from __future__ import annotations

import json
import os
import random
import unittest
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import hyperloader
from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import FactorConfig
from hyperloader.iterable.runtime import IterableLaneRuntime


class ReplaySource:
    """Expose lane-sized source states with exact replay semantics."""

    def __init__(self, stop: int = 31) -> None:
        self.stop = stop
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
        return {
            "position": self.position,
            "padding": "x" * 2048 if self.lane == 1 else "",
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.position = int(state["position"])


def _loader(*, lanes: int = 2) -> DataLoader:
    return DataLoader(
        ReplaySource(),
        batch_size=2,
        num_workers=lanes,
        seed=1051,
        config=HyperConfig(
            factors=FactorConfig(f_snap=3, f_snap_bytes=256)
        ),
    )


def _batch(batch: Any) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(value), int(bits))
        for value, bits in zip(batch[0].tolist(), batch[1].tolist(), strict=True)
    )


def _stream(loader: DataLoader) -> list[tuple[tuple[int, int], ...]]:
    return [_batch(batch) for batch in loader]


def _skip_replay(self: IterableLaneRuntime, lane: Any, target: int) -> None:
    del self, lane, target


class SourceReplayGate(unittest.TestCase):
    """Prove snapshots and replay-from-start compose at delivered cut points."""

    def test_cut_points_and_mixed_snapshot_modes_are_exact(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is not None:
            root = Path(expected_root).resolve()
            self.assertTrue(Path(hyperloader.__file__).resolve().is_relative_to(root))
            self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline_loader = _loader()
            try:
                baseline = _stream(baseline_loader)
            finally:
                baseline_loader.close()

        mutation = (
            mock.patch.object(IterableLaneRuntime, "_replay_to", _skip_replay)
            if os.environ.get("HYPERLOADER_SOURCE_REPLAY_MUTATION") == "skip-replay"
            else nullcontext()
        )
        metrics = []
        with mutation, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for cut in (0, 1, 3, 5, 7):
                with self.subTest(cut=cut):
                    source = _loader()
                    iterator = iter(source)
                    prefix = [_batch(next(iterator)) for _ in range(cut)]
                    state = source.state_dict()
                    source.close()
                    resumed = _loader()
                    try:
                        resumed.load_state_dict(state)
                        suffix = _stream(resumed)
                    finally:
                        resumed.close()
                    self.assertEqual(prefix + suffix, baseline)
                    snapshots = sum(
                        lane["snapshot"] is not None for lane in state["lanes"]
                    )
                    metrics.append(
                        {
                            "cut": cut,
                            "prefix_batches": len(prefix),
                            "remaining_batches": len(suffix),
                            "snapshot_lanes": snapshots,
                        }
                    )

        metrics_path = os.environ.get("HYPERLOADER_SOURCE_REPLAY_METRICS")
        if metrics_path is not None:
            Path(metrics_path).write_text(
                json.dumps({"cuts": metrics}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def test_resume_rejects_changed_logical_lane_count(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            source = _loader()
            iterator = iter(source)
            next(iterator)
            state = source.state_dict()
            source.close()
            changed = _loader(lanes=1)
            try:
                with self.assertRaisesRegex(
                    ValueError,
                    "fingerprint|logical lane count",
                ):
                    changed.load_state_dict(state)
            finally:
                changed.close()


if __name__ == "__main__":
    unittest.main()
