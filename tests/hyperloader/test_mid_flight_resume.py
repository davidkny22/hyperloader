"""Installed public gate for strict-order resume across speculative work."""

from __future__ import annotations

import json
import os
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader import state as state_module
from hyperloader.config import SchedulerConfig

_CAPTURE_MAP_STATE = state_module.capture_map_state


class SkewedRangeDataset:
    """Keep a bounded frontier populated with out-of-order map work."""

    def __init__(self, length: int = 16) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        time.sleep(0.018 if index % 6 == 0 else 0.001)
        return index


def _loader() -> DataLoader:
    return DataLoader(
        SkewedRangeDataset(),
        batch_size=2,
        num_workers=4,
        seed=211,
        config=HyperConfig(
            scheduler=SchedulerConfig(frontier_depth=8, profile_cache="off")
        ),
    )


def _flatten(iterator: Any) -> list[int]:
    values: list[int] = []
    for batch in iterator:
        values.extend(int(value) for value in batch.tolist())
    return values


def _core_iterator(iterator: Any) -> Any:
    core = iterator
    while hasattr(core, "_iterator"):
        core = core._iterator
    return core


def _advance_checkpoint(loader: DataLoader) -> dict[str, object]:
    state = _CAPTURE_MAP_STATE(loader)
    state["cursor"] = int(state["cursor"]) + 1
    return state


class MidFlightResumeGate(unittest.TestCase):
    """Prove delivered-prefix continuation at metamorphic frontier cuts."""

    def test_speculative_cut_points_resume_without_duplicate_or_skip(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is not None:
            self.assertTrue(
                Path(_hyperloader.__file__)
                .resolve()
                .is_relative_to(Path(expected_root).resolve())
            )

        reference = _loader()
        try:
            expected = _flatten(reference)
        finally:
            reference.close()

        evidence = []
        for cut_batches in (0, 1, 3, 6):
            source = _loader()
            iterator = iter(source)
            prefix = []
            try:
                for _ in range(cut_batches):
                    prefix.extend(int(value) for value in next(iterator).tolist())
                core = _core_iterator(iterator)
                occupied = core._schedule.occupied
                self.assertGreater(occupied, 0)
                mutation = (
                    mock.patch.object(
                        state_module,
                        "capture_map_state",
                        _advance_checkpoint,
                    )
                    if os.environ.get("HYPERLOADER_MID_FLIGHT_MUTATION")
                    == "advance-cursor"
                    else nullcontext()
                )
                with mutation:
                    state = source.state_dict()
                self.assertEqual(state["cursor"], cut_batches)
            finally:
                source.close()

            resumed = _loader()
            try:
                resumed.load_state_dict(state)
                actual = prefix + _flatten(resumed)
            finally:
                resumed.close()
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), len(set(actual)))
            evidence.append(
                {
                    "cut_batches": cut_batches,
                    "frontier_occupied": occupied,
                    "state_cursor": state["cursor"],
                }
            )

        metrics_path = os.environ.get("HYPERLOADER_MID_FLIGHT_METRICS")
        if metrics_path is not None:
            Path(metrics_path).write_text(
                json.dumps({"cut_points": evidence}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def test_advanced_cursor_is_observably_not_equivalent(self) -> None:
        reference = _loader()
        source = _loader()
        iterator = iter(source)
        try:
            expected = _flatten(reference)
            prefix = []
            for _ in range(2):
                prefix.extend(int(value) for value in next(iterator).tolist())
            state = source.state_dict()
        finally:
            reference.close()
            source.close()

        state["cursor"] = int(state["cursor"]) + 1
        resumed = _loader()
        try:
            resumed.load_state_dict(state)
            actual = prefix + _flatten(resumed)
        finally:
            resumed.close()

        self.assertNotEqual(actual, expected)
        self.assertEqual(actual, expected[:4] + expected[6:])


if __name__ == "__main__":
    unittest.main()
