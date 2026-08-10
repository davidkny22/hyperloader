"""Fixtures and equality oracle for installed stage-parity checks."""

from __future__ import annotations

import random
import struct
import unittest
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import default_collate

from hyperloader import (
    Collate,
    Decode,
    Pipeline,
    Source,
    ThreadSafety,
    Transform,
    pipeline,
    rng,
)

RAW_VALUES = tuple(str(index).encode("ascii") for index in range(11))


def decode_index(value: bytes) -> int:
    """Decode one source token."""
    return int(value)


def render_global_sample(index: int) -> dict[str, object]:
    """Render one nested sample through the installed global RNG shims."""
    return {
        "index": index,
        "nested": (
            torch.tensor([index, index + 1], dtype=torch.int64),
            np.float64(np.random.random()),
        ),
        "python": random.random(),
        "torch": torch.rand((), dtype=torch.float64),
    }


def render_accessor_sample(index: int) -> tuple[int, float, float, float]:
    """Render one sample using only sanctioned per-sample generators."""
    return (
        index,
        float(torch.rand((), generator=rng()).item()),
        float(rng("numpy").random()),
        rng("random").random(),
    )


class GlobalBlackBox:
    """Execute the staged global-RNG operations as one black-box item call."""

    def __len__(self) -> int:
        return len(RAW_VALUES)

    def __getitem__(self, index: int) -> dict[str, object]:
        return render_global_sample(decode_index(RAW_VALUES[index]))


class AccessorBlackBox:
    """Execute the staged accessor operations as one black-box item call."""

    def __len__(self) -> int:
        return len(RAW_VALUES)

    def __getitem__(self, index: int) -> tuple[int, float, float, float]:
        return render_accessor_sample(decode_index(RAW_VALUES[index]))


class ExplodingBlackBox:
    """Raise at one deterministic item position."""

    def __len__(self) -> int:
        return 7

    def __getitem__(self, index: int) -> int:
        return raise_at_four(index)


def raise_at_four(index: int) -> int:
    """Raise a stable exception at the fifth source position."""
    if index == 4:
        raise ValueError("stage parity sentinel")
    return index


def global_pipeline() -> Pipeline[object, object]:
    """Build the isolated process-stage fixture."""
    return pipeline(
        Source(RAW_VALUES, output_type=bytes),
        Decode(decode_index, input_type=bytes, output_type=int),
        Transform(render_global_sample, input_type=int, output_type=dict),
        Collate(default_collate, input_type=dict, output_type=dict),
    )


def accessor_pipeline() -> Pipeline[object, object]:
    """Build the declared thread-stage fixture."""
    safety = ThreadSafety.THREAD_SAFE
    return pipeline(
        Source(RAW_VALUES, output_type=bytes, thread_safety=safety),
        Decode(
            decode_index,
            input_type=bytes,
            output_type=int,
            thread_safety=safety,
        ),
        Transform(
            render_accessor_sample,
            input_type=int,
            output_type=tuple,
            thread_safety=safety,
        ),
        Collate(default_collate, input_type=tuple, output_type=list),
    )


def exploding_pipeline() -> Pipeline[object, object]:
    """Build the deterministic exception fixture."""
    return pipeline(
        Source(tuple(range(7)), output_type=int),
        Transform(raise_at_four, input_type=int, output_type=int),
        Collate(default_collate, input_type=int, output_type=torch.Tensor),
    )


def skip_sample_chain(self: Pipeline[object, object], index: int) -> object:
    """Plant a severed pipeline-to-sample-stage connection."""
    return self.source(index)


def assert_contract_equal(
    test: unittest.TestCase, actual: object, expected: object
) -> None:
    """Apply the recursive bit-exact parity relation to two values."""
    test.assertIs(type(actual), type(expected))
    if isinstance(actual, torch.Tensor):
        test.assertEqual(actual.dtype, expected.dtype)
        test.assertEqual(actual.shape, expected.shape)
        test.assertEqual(actual.layout, expected.layout)
        test.assertEqual(actual.stride(), expected.stride())
        test.assertEqual(actual.device, expected.device)
        actual_bits = actual.detach().cpu().contiguous().view(torch.uint8)
        expected_bits = expected.detach().cpu().contiguous().view(torch.uint8)
        test.assertTrue(torch.equal(actual_bits, expected_bits))
        return
    if isinstance(actual, Mapping):
        test.assertEqual(list(actual), list(expected))
        for key in actual:
            assert_contract_equal(test, actual[key], expected[key])
        return
    if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes, bytearray)):
        test.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_contract_equal(test, actual_item, expected_item)
        return
    if isinstance(actual, float):
        test.assertEqual(struct.pack("!d", actual), struct.pack("!d", expected))
        return
    test.assertEqual(actual, expected)
