"""Completion-order process delivery and bounded bitmap state."""

from __future__ import annotations

import time
import unittest

from hyperloader import DataLoader, HyperConfig
from hyperloader.config import SchedulerConfig


class HeadSkewDataset:
    """Delay the first batch so later batches complete observably first."""

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> int:
        time.sleep(0.2 if index < 2 else 0.001)
        return index


def _loader(*, in_order: bool, thread_safe: bool = False) -> DataLoader:
    return DataLoader(
        HeadSkewDataset(),
        batch_size=2,
        num_workers=4,
        seed=307,
        in_order=in_order,
        thread_safe=thread_safe,
        config=HyperConfig(
            scheduler=SchedulerConfig(frontier_depth=8, profile_cache="off")
        ),
    )


def _batch_tuple(batch: object) -> tuple[int, ...]:
    return tuple(int(value) for value in batch.tolist())  # type: ignore[attr-defined]


class CompletionDeliveryTest(unittest.TestCase):
    """Preserve composition while allowing completion-order sequencing."""

    def test_completion_order_changes_sequence_not_batch_composition(self) -> None:
        strict = _loader(in_order=True)
        completion = _loader(in_order=False)
        try:
            strict_batches = [_batch_tuple(batch) for batch in strict]
            completion_batches = [_batch_tuple(batch) for batch in completion]
        finally:
            strict.close()
            completion.close()

        self.assertEqual(sorted(completion_batches), sorted(strict_batches))
        self.assertNotEqual(completion_batches, strict_batches)
        self.assertEqual(completion_batches[0], (2, 3))

    def test_checkpoint_bitmap_tracks_delivered_batches_beyond_the_prefix(self) -> None:
        loader = _loader(in_order=False)
        iterator = iter(loader)
        try:
            first = _batch_tuple(next(iterator))
            state = loader.state_dict()
            remaining = [_batch_tuple(batch) for batch in iterator]
        finally:
            loader.close()

        self.assertEqual(first, (2, 3))
        self.assertEqual(state["cursor"], 0)
        self.assertEqual(state["delivered_bitmap"], b"\x02")
        self.assertLessEqual(len(state["delivered_bitmap"]), 1)
        self.assertEqual(
            sorted([first, *remaining]),
            [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
        )

    def test_thread_completion_order_preserves_batch_composition(self) -> None:
        strict = _loader(in_order=True, thread_safe=True)
        completion = _loader(in_order=False, thread_safe=True)
        try:
            strict_batches = [_batch_tuple(batch) for batch in strict]
            completion_batches = [_batch_tuple(batch) for batch in completion]
        finally:
            strict.close()
            completion.close()

        self.assertEqual(sorted(completion_batches), sorted(strict_batches))
        self.assertNotEqual(completion_batches, strict_batches)
        self.assertEqual(completion_batches[0], (2, 3))

    def test_strict_state_rejects_a_completion_bitmap(self) -> None:
        source = _loader(in_order=True)
        target = _loader(in_order=True)
        try:
            state = source.state_dict()
            state["delivered_bitmap"] = b"\x01"
            with self.assertRaisesRegex(ValueError, "empty delivered_bitmap"):
                target.load_state_dict(state)
        finally:
            source.close()
            target.close()


if __name__ == "__main__":
    unittest.main()
