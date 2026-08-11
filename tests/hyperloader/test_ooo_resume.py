"""Installed public gate for completion-order resume across open windows."""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path

import hyperloader
from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import SchedulerConfig


class HeadSkewDataset:
    """Keep the first batch open while later batches become deliverable."""

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> int:
        time.sleep(0.2 if index < 2 else 0.001)
        return index


def _loader(*, thread_safe: bool = False) -> DataLoader:
    return DataLoader(
        HeadSkewDataset(),
        batch_size=2,
        num_workers=4,
        seed=431,
        in_order=False,
        thread_safe=thread_safe,
        config=HyperConfig(
            scheduler=SchedulerConfig(frontier_depth=8, profile_cache="off")
        ),
    )


def _batch_tuple(batch: object) -> tuple[int, ...]:
    return tuple(int(value) for value in batch.tolist())  # type: ignore[attr-defined]


def _decode_bitmap(cursor: int, bitmap: bytes) -> list[int]:
    return [
        cursor + byte_index * 8 + bit
        for byte_index, byte in enumerate(bitmap)
        for bit in range(8)
        if byte & (1 << bit)
    ]


class OutOfOrderResumeGate(unittest.TestCase):
    """Prove exact continuation across open completion-order windows."""

    def test_open_window_cuts_resume_without_duplicate_or_skip(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is not None:
            root = Path(expected_root).resolve()
            self.assertTrue(Path(hyperloader.__file__).resolve().is_relative_to(root))
            self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))

        expected = {
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
        }
        evidence = []
        for thread_safe in (False, True):
            for cut_deliveries in (1, 2, 3):
                with self.subTest(
                    thread_safe=thread_safe, cut_deliveries=cut_deliveries
                ):
                    source = _loader(thread_safe=thread_safe)
                    iterator = iter(source)
                    delivered = []
                    try:
                        for _ in range(cut_deliveries):
                            delivered.append(_batch_tuple(next(iterator)))
                        state = source.state_dict()
                    finally:
                        source.close()

                    bitmap = bytes(state["delivered_bitmap"])
                    if os.environ.get("HYPERLOADER_OOO_MUTATION") == "drop-bitmap":
                        state["delivered_bitmap"] = b""
                    cursor = int(state["cursor"])
                    delivered_ordinals = _decode_bitmap(cursor, bitmap)
                    self.assertEqual(cursor, 0)
                    self.assertEqual(len(delivered_ordinals), cut_deliveries)
                    self.assertNotIn(cursor, delivered_ordinals)

                    resumed = _loader(thread_safe=thread_safe)
                    try:
                        resumed.load_state_dict(state)
                        remaining = [_batch_tuple(batch) for batch in resumed]
                    finally:
                        resumed.close()

                    actual = delivered + remaining
                    self.assertEqual(set(actual), expected)
                    self.assertEqual(len(actual), len(expected))
                    self.assertEqual(len(actual), len(set(actual)))
                    evidence.append(
                        {
                            "cut_deliveries": cut_deliveries,
                            "cursor": cursor,
                            "delivered_bitmap": bitmap.hex(),
                            "delivered_ordinals": delivered_ordinals,
                            "remaining_batches": len(remaining),
                            "tier": "thread" if thread_safe else "process",
                        }
                    )

        metrics_path = os.environ.get("HYPERLOADER_OOO_METRICS")
        if metrics_path is not None:
            Path(metrics_path).write_text(
                json.dumps({"cut_points": evidence}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def test_bitmap_cannot_claim_the_contiguous_gap(self) -> None:
        source = _loader()
        iterator = iter(source)
        try:
            next(iterator)
            state = source.state_dict()
        finally:
            source.close()

        state["delivered_bitmap"] = b"\x01"
        resumed = _loader()
        try:
            with self.assertRaisesRegex(ValueError, "bit zero"):
                resumed.load_state_dict(state)
        finally:
            resumed.close()


if __name__ == "__main__":
    unittest.main()
