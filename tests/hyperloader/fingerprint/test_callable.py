"""Callable identity and closure-boundary tests."""

from __future__ import annotations

import unittest

from hyperloader.fingerprint.callable import callable_identity


def _closure(value: int):  # type: ignore[no-untyped-def]
    def add(number: int) -> int:
        return number + value

    return add


def add_one(number: int) -> int:
    """Provide one code object."""
    return number + 1


def double(number: int) -> int:
    """Provide a distinct code object."""
    return number * 2


class CallableFingerprintTest(unittest.TestCase):
    """Prove code identity and the documented closure-value caveat."""

    def test_distinct_code_objects_have_distinct_hashes(self) -> None:
        self.assertNotEqual(
            callable_identity(add_one)["code_sha256"],
            callable_identity(double)["code_sha256"],
        )

    def test_closure_values_are_intentionally_outside_identity(self) -> None:
        self.assertEqual(callable_identity(_closure(1)), callable_identity(_closure(2)))

    def test_builtin_types_have_compact_stable_identity(self) -> None:
        self.assertEqual(
            callable_identity(bytes), {"kind": "type", "qualname": "builtins.bytes"}
        )


if __name__ == "__main__":
    unittest.main()
