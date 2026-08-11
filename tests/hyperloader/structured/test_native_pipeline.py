"""Public-path checks for pinned image and tokenized text native batches."""

from __future__ import annotations

import unittest

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import decode_png, encode_png

from hyperloader import (
    Collate,
    DataLoader,
    Decode,
    HyperConfig,
    Source,
    Transform,
    pipeline,
)
from hyperloader.config import MemoryConfig


def forbidden_decode(_value: torch.Tensor) -> torch.Tensor:
    """Fail when a selected provider does not replace the refuge callable."""
    raise AssertionError("selected decoder did not execute")


def identity(value: torch.Tensor) -> torch.Tensor:
    """Provide an intentionally user-owned transform for refuge coverage."""
    return value


def custom_stack(values: list[torch.Tensor]) -> torch.Tensor:
    """Provide a semantically familiar but unregistered collation function."""
    return torch.stack(values)


class NativePipelineTest(unittest.TestCase):
    """Exercise exact native families and their black-box boundaries."""

    def test_pinned_png_batches_return_the_final_stack_directly(self) -> None:
        encoded = [
            encode_png((torch.arange(60, dtype=torch.uint8) + offset).reshape(3, 4, 5))
            for offset in range(4)
        ]
        dataset = pipeline(
            Source(encoded, output_type=torch.Tensor),
            Decode(
                forbidden_decode,
                input_type=torch.Tensor,
                output_type=torch.Tensor,
                codec="png",
                substitute=True,
            ),
            Collate(torch.stack, input_type=torch.Tensor, output_type=torch.Tensor),
        )
        expected = torch.stack([decode_png(value) for value in encoded[:2]])
        loader = DataLoader(dataset, batch_size=2, num_workers=2, seed=23)
        try:
            iterator = iter(loader)
            self.assertEqual(set(loader._execution_dataset._futures), {2})
            prefetched = loader._execution_dataset._futures[2].result(timeout=5.0)
            self.assertEqual(tuple(prefetched.batch.shape), (2, 3, 4, 5))
            actual = next(iterator)
            self.assertTrue(torch.equal(actual, expected))
            self.assertIsNone(loader._process_pool)
            report = loader.stats()["memory"]
            self.assertEqual(report["source_class"], "pinned-decode")
            self.assertEqual(report["delivery"], "single-write")
            self.assertEqual(report["bytes_beyond_irreducible"], 0)
            self.assertFalse(report["variable_shape"])
            self.assertEqual(report["produced_batches"], 1)
            self.assertEqual(actual.untyped_storage().nbytes(), actual.numel())
            self.assertEqual(report["slot_capacity_bytes"], 128)
            loader.close()
            self.assertTrue(torch.equal(actual, expected))
        finally:
            loader.close()

    def test_variable_token_sequences_use_one_final_padding_write(self) -> None:
        tokens = [
            torch.tensor([1]),
            torch.tensor([2, 3]),
            torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
            torch.tensor([8, 9, 10]),
        ]
        dataset = pipeline(
            Source(
                tokens,
                output_type=torch.Tensor,
                thread_safety="thread-safe",
            ),
            Collate(pad_sequence, input_type=torch.Tensor, output_type=torch.Tensor),
        )
        loader = DataLoader(dataset, batch_size=2, num_workers=2, seed=29)
        try:
            batches = list(loader)
            self.assertTrue(torch.equal(batches[0], pad_sequence(tokens[:2])))
            self.assertTrue(torch.equal(batches[1], pad_sequence(tokens[2:])))
            self.assertIsNone(loader._process_pool)
            report = loader.stats()["memory"]
            self.assertEqual(report["source_class"], "tokenized-text")
            self.assertTrue(report["variable_shape"])
            self.assertEqual(report["minimum_sample_bytes"], 8)
            self.assertEqual(report["maximum_sample_bytes"], 64)
            self.assertEqual(report["produced_batches"], 2)
            self.assertEqual(report["produced_samples"], 4)
            self.assertEqual(report["overflow_events"], 1)
            self.assertEqual(report["slot_capacity_bytes"], 128)
            self.assertEqual(report["growth_events"], 1)
            self.assertEqual(report["regions"], 3)
            self.assertEqual(batches[0].untyped_storage().nbytes(), 32)
            self.assertEqual(batches[1].untyped_storage().nbytes(), 128)
            self.assertEqual(
                report["loader_written_bytes"],
                sum(batch.numel() * batch.element_size() for batch in batches),
            )
        finally:
            loader.close()

    def test_variable_output_strict_growth_raises_at_the_overrun(self) -> None:
        tokens = [
            torch.tensor([1]),
            torch.tensor([2, 3]),
            torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
            torch.tensor([8, 9, 10]),
        ]
        dataset = pipeline(
            Source(tokens, output_type=torch.Tensor),
            Collate(pad_sequence, input_type=torch.Tensor, output_type=torch.Tensor),
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            seed=30,
            config=HyperConfig(memory=MemoryConfig(growth="strict-error")),
        )
        iterator = iter(loader)
        try:
            self.assertTrue(torch.equal(next(iterator), pad_sequence(tokens[:2])))
            with self.assertRaisesRegex(RuntimeError, "growth is disabled"):
                next(iterator)
            report = loader.stats()["memory"]
            self.assertEqual(report["overflow_events"], 1)
            self.assertEqual(report["slot_capacity_bytes"], 32)
        finally:
            loader.close()

    def test_held_fixed_batches_grow_a_complete_slot_class(self) -> None:
        tokens = [torch.tensor([index, index + 1]) for index in range(6)]
        dataset = pipeline(
            Source(tokens, output_type=torch.Tensor),
            Collate(torch.stack, input_type=torch.Tensor, output_type=torch.Tensor),
        )
        loader = DataLoader(dataset, batch_size=2, num_workers=2, seed=30)
        try:
            batches = list(loader)
            report = loader.stats()["memory"]
            self.assertEqual(len(batches), 3)
            self.assertEqual(report["growth_events"], 1)
            self.assertEqual(report["hold_events"], 1)
            self.assertEqual(report["overflow_events"], 0)
            loader.close()
            self.assertTrue(torch.equal(batches[0], torch.stack(tokens[:2])))
        finally:
            loader.close()

    def test_custom_collate_and_user_transform_retain_process_refuge(self) -> None:
        tokens = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        custom = pipeline(
            Source(tokens, output_type=torch.Tensor),
            Collate(custom_stack, input_type=torch.Tensor, output_type=torch.Tensor),
        )
        transformed = pipeline(
            Source(tokens, output_type=torch.Tensor),
            Transform(identity, input_type=torch.Tensor, output_type=torch.Tensor),
            Collate(torch.stack, input_type=torch.Tensor, output_type=torch.Tensor),
        )
        for dataset in (custom, transformed):
            loader = DataLoader(dataset, batch_size=2, num_workers=1, seed=31)
            try:
                self.assertIsNotNone(loader._process_pool)
                self.assertTrue(torch.equal(next(iter(loader)), torch.stack(tokens)))
                self.assertNotIn("memory", loader.stats())
            finally:
                loader.close()


if __name__ == "__main__":
    unittest.main()
