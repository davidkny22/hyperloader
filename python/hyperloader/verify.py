"""Offline equivalence evidence for declared thread-safe datasets."""

from __future__ import annotations

import pickle
import struct
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

from .api import DataLoader


def verify(
    dataset: Any,
    *,
    samples: int = 16,
    seed: int = 0,
    num_workers: int = 2,
) -> dict[str, object]:
    """Compare a dataset sample set bit-exactly across process and thread tiers."""
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError("samples must be an integer")
    if samples < 0:
        raise ValueError("samples must be nonnegative")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers must be an integer")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    payload = pickle.dumps(dataset, protocol=5)
    process_dataset = pickle.loads(payload)
    thread_dataset = pickle.loads(payload)
    process = DataLoader(
        process_dataset,
        batch_size=None,
        num_workers=num_workers,
        seed=seed,
    )
    threaded = DataLoader(
        thread_dataset,
        batch_size=None,
        num_workers=num_workers,
        seed=seed,
        thread_safe=True,
    )
    try:
        process_values = list(islice(process, samples))
        thread_values = list(islice(threaded, samples))
    finally:
        process.close()
        threaded.close()

    mismatch = _first_mismatch(process_values, thread_values)
    return {
        "bit_exact": mismatch is None,
        "compared_samples": min(len(process_values), len(thread_values)),
        "first_mismatch": mismatch,
    }


def _first_mismatch(left: list[Any], right: list[Any]) -> int | None:
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
        if not _bit_equal(left_value, right_value):
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _bit_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return struct.pack("=d", left) == struct.pack("=d", right)
    if isinstance(left, (str, bytes, int, bool, type(None))):
        return left == right
    if isinstance(left, Mapping):
        return list(left) == list(right) and all(
            _bit_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, Sequence):
        return len(left) == len(right) and all(
            _bit_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if _is_torch_tensor(left):
        return _tensor_identity(left) == _tensor_identity(right)
    if _is_numpy_array(left):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.strides == right.strides
            and left.tobytes(order="A") == right.tobytes(order="A")
        )
    if _is_numpy_scalar(left):
        return left.dtype == right.dtype and left.tobytes() == right.tobytes()
    try:
        equal = left == right
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _is_torch_tensor(value: Any) -> bool:
    module = type(value).__module__
    return module == "torch" or module.startswith("torch.")


def _tensor_identity(value: Any) -> tuple[Any, ...]:
    tensor = value.detach().cpu()
    if str(tensor.layout) != "torch.strided":
        return (
            str(value.dtype),
            tuple(value.shape),
            str(value.layout),
            str(value.device),
            repr(tensor),
        )
    return (
        str(value.dtype),
        tuple(value.shape),
        str(value.layout),
        tuple(value.stride()),
        str(value.device),
        bytes(tensor.contiguous().clone().untyped_storage()),
    )


def _is_numpy_array(value: Any) -> bool:
    return type(value).__module__.startswith("numpy") and hasattr(value, "tobytes")


def _is_numpy_scalar(value: Any) -> bool:
    return (
        type(value).__module__.startswith("numpy")
        and hasattr(value, "dtype")
        and hasattr(value, "tobytes")
    )
