"""Differential matrix for native engine collation and pinned torch."""

from __future__ import annotations

import math
import os
import unittest
from collections import OrderedDict, namedtuple
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from torch.utils.data._utils import collate as torch_collate
except ImportError as error:
    raise unittest.SkipTest("torch and NumPy are required for the collation oracle") from error

from hyperloader import DataLoader, _hyperloader


Point = namedtuple("Point", ("x", "y"))


class TaggedDict(dict):
    """Mutable mapping whose extra state must survive reconstruction."""


class FrozenMap(Mapping):
    """Immutable mapping constructible from another mapping."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class BrokenMap(FrozenMap):
    """Immutable mapping whose constructor forces torch's dict fallback."""

    def __init__(self, values: Mapping[str, Any], token: object) -> None:
        super().__init__(values)
        self.token = token


class TaggedList(list):
    """Mutable sequence whose extra state must survive reconstruction."""


class FrozenSequence(Sequence):
    """Immutable sequence constructible from an iterable."""

    def __init__(self, values) -> None:
        self._values = tuple(values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


class BrokenSequence(FrozenSequence):
    """Sequence whose constructor forces torch's list fallback."""

    def __init__(self, values, token: object) -> None:
        super().__init__(values)
        self.token = token


class RegisteredType:
    """Custom type used to prove the explicit registration boundary."""


class UnknownType:
    """Unsupported leaf used to compare torch's default error."""


def _value_cases() -> list[tuple[str, list[Any]]]:
    tensor_dtypes = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.complex64,
        torch.complex128,
    )
    numpy_dtypes = (
        np.bool_,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    )
    cases: list[tuple[str, list[Any]]] = []
    for dtype in tensor_dtypes:
        cases.append(
            (
                f"tensor-{dtype}",
                [torch.tensor([1, 2], dtype=dtype), torch.tensor([3, 4], dtype=dtype)],
            )
        )
    cases.append(
        (
            "tensor-noncontiguous",
            [torch.arange(12).reshape(3, 4).t(), torch.arange(12, 24).reshape(3, 4).t()],
        )
    )
    cases.append(("tensor-scalars", [torch.tensor(1), torch.tensor(2)]))
    for dtype in numpy_dtypes:
        cases.append(
            (
                f"ndarray-{np.dtype(dtype)}",
                [np.array([1, 2], dtype=dtype), np.array([3, 4], dtype=dtype)],
            )
        )
        if dtype is not np.uint64:
            cases.append(
                (
                    f"numpy-scalar-{np.dtype(dtype)}",
                    [dtype(1), dtype(2)],
                )
            )
    tagged_dicts = [TaggedDict(a=1), TaggedDict(a=2)]
    tagged_dicts[0].tag = "preserved"
    tagged_dicts[1].tag = "ignored-like-torch"
    tagged_lists = [TaggedList([1, 2]), TaggedList([3, 4])]
    tagged_lists[0].tag = "preserved"
    tagged_lists[1].tag = "ignored-like-torch"
    token = object()
    cases.extend(
        [
            ("python-floats", [1.25, -2.5]),
            ("python-ints", [1, 2]),
            ("python-bools", [True, False]),
            ("strings", ["alpha", "beta"]),
            ("bytes", [b"alpha", b"beta"]),
            ("dict", [{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}]),
            (
                "ordered-dict",
                [
                    OrderedDict((("a", 1), ("b", 2))),
                    OrderedDict((("a", 3), ("b", 4))),
                ],
            ),
            ("tagged-dict", tagged_dicts),
            ("frozen-map", [FrozenMap({"a": 1}), FrozenMap({"a": 2})]),
            (
                "mapping-fallback",
                [BrokenMap({"a": 1}, token), BrokenMap({"a": 2}, token)],
            ),
            ("namedtuple", [Point(1, 2.0), Point(3, 4.0)]),
            ("tuple-backward-compatibility", [(1, 2), (3, 4)]),
            ("list", [[1, 2], [3, 4]]),
            ("tagged-list", tagged_lists),
            (
                "frozen-sequence",
                [FrozenSequence([1, 2]), FrozenSequence([3, 4])],
            ),
            (
                "sequence-fallback",
                [BrokenSequence([1, 2], token), BrokenSequence([3, 4], token)],
            ),
            (
                "nested",
                [
                    {"point": Point(np.int16(1), [2.0, True])},
                    {"point": Point(np.int16(3), [4.0, False])},
                ],
            ),
        ]
    )
    return cases


def _error_cases() -> list[tuple[str, list[Any]]]:
    sparse = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse()
    return [
        ("none", [None, None]),
        ("ragged", [[1, 2], [3]]),
        ("numpy-object", [np.array([object()], dtype=object)]),
        ("numpy-string", [np.array(["a"]), np.array(["b"])]),
        ("numpy-uint64-scalar", [np.uint64(1), np.uint64(2)]),
        ("unknown", [UnknownType(), UnknownType()]),
        ("tensor-shape", [torch.ones(2), torch.ones(3)]),
        ("sparse", [sparse, sparse]),
    ]


def _tensor_bytes(value: torch.Tensor) -> bytes:
    contiguous = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return bytes(contiguous.tolist())


def _assert_equal(test: unittest.TestCase, expected: Any, actual: Any) -> None:
    test.assertIs(type(actual), type(expected))
    if isinstance(expected, torch.Tensor):
        test.assertEqual(actual.dtype, expected.dtype)
        test.assertEqual(actual.shape, expected.shape)
        test.assertEqual(actual.stride(), expected.stride())
        test.assertEqual(actual.layout, expected.layout)
        test.assertEqual(actual.device, expected.device)
        test.assertEqual(_tensor_bytes(actual), _tensor_bytes(expected))
    elif isinstance(expected, Mapping):
        test.assertEqual(list(actual), list(expected))
        for key in expected:
            _assert_equal(test, expected[key], actual[key])
        if hasattr(expected, "tag"):
            test.assertEqual(actual.tag, expected.tag)
    elif isinstance(expected, tuple) and hasattr(expected, "_fields"):
        for expected_value, actual_value in zip(expected, actual, strict=True):
            _assert_equal(test, expected_value, actual_value)
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        test.assertEqual(len(actual), len(expected))
        for expected_value, actual_value in zip(expected, actual, strict=True):
            _assert_equal(test, expected_value, actual_value)
        if hasattr(expected, "tag"):
            test.assertEqual(actual.tag, expected.tag)
    elif isinstance(expected, float) and math.isnan(expected):
        test.assertTrue(math.isnan(actual))
    else:
        test.assertEqual(actual, expected)


class CollateEquivalenceTest(unittest.TestCase):
    """Compare the installed native path with the pinned torch oracle."""

    @classmethod
    def setUpClass(cls) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is None:
            raise unittest.SkipTest("the installed gate runner provides the package root")
        module_path = Path(_hyperloader.__file__).resolve()
        if not module_path.is_relative_to(Path(expected_root).resolve()):
            raise AssertionError("native extension did not resolve from the isolated install")
        cls.loader = DataLoader([], seed=0)
        print(f"TORCH_VERSION={torch.__version__}")
        print(f"COLLATE_VALUE_CASES={len(_value_cases())}")
        print(f"COLLATE_ERROR_CASES={len(_error_cases())}")

    def test_value_structure_and_dtype_matrix(self) -> None:
        for name, batch in _value_cases():
            with self.subTest(name=name):
                expected = torch.utils.data.default_collate(batch)
                actual = self.loader._collate_batch(batch)
                if (
                    os.environ.get("HYPERLOADER_COLLATE_MUTATION") == "flip-int"
                    and name == "python-ints"
                ):
                    actual = actual.clone()
                    actual[0] += 1
                _assert_equal(self, expected, actual)

    def test_error_type_and_message_matrix(self) -> None:
        for name, batch in _error_cases():
            with self.subTest(name=name):
                try:
                    torch.utils.data.default_collate(batch)
                except Exception as expected:
                    with self.assertRaises(type(expected)) as caught:
                        self.loader._collate_batch(batch)
                    self.assertEqual(str(caught.exception), str(expected))
                else:
                    self.fail(f"the torch oracle unexpectedly accepted {name}")

    def test_custom_registration_has_explicit_engine_error(self) -> None:
        def registered_collate(batch, *, collate_fn_map=None):
            return "torch-registration"

        torch_collate.default_collate_fn_map[RegisteredType] = registered_collate
        try:
            self.assertEqual(
                torch.utils.data.default_collate([RegisteredType()]),
                "torch-registration",
            )
            with self.assertRaisesRegex(
                TypeError, "does not support custom default_collate registrations"
            ):
                self.loader._collate_batch([RegisteredType()])
        finally:
            del torch_collate.default_collate_fn_map[RegisteredType]


if __name__ == "__main__":
    unittest.main()
