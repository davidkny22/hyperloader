"""Differential signature checks against an installed torch reference."""

from __future__ import annotations

import inspect
import os
import unittest
from collections.abc import Callable
from typing import Any

from hyperloader import AUTO, DataLoader


TORCH_PARAMETER_NAMES = (
    "dataset",
    "batch_size",
    "shuffle",
    "sampler",
    "batch_sampler",
    "num_workers",
    "collate_fn",
    "pin_memory",
    "drop_last",
    "timeout",
    "worker_init_fn",
    "multiprocessing_context",
    "generator",
    "prefetch_factor",
    "persistent_workers",
    "pin_memory_device",
    "in_order",
)
HYPERLOADER_PARAMETER_NAMES = (
    "seed",
    "thread_safe",
    "mode",
    "delivery",
    "device",
    "config",
)


def _missing_in_order_surface(
    dataset: Any,
    batch_size: int | None = 1,
    shuffle: bool | None = None,
    sampler: Any = None,
    batch_sampler: Any = None,
    num_workers: int = 0,
    collate_fn: Any = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Any = None,
    multiprocessing_context: Any = None,
    generator: Any = None,
    *,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
) -> None:
    """Model the planted removal of torch's in_order parameter."""
    del dataset


def _candidate_surface() -> Callable[..., Any]:
    if os.environ.get("HYPERLOADER_SURFACE_MUTATION") == "missing-in-order":
        return _missing_in_order_surface
    return DataLoader


def assert_surface_compatible(
    reference: Callable[..., Any], candidate: Callable[..., Any]
) -> None:
    """Raise when the candidate breaks the torch-compatible parameter surface."""
    reference_parameters = inspect.signature(reference).parameters
    candidate_parameters = inspect.signature(candidate).parameters

    expected_names = tuple(reference_parameters)
    if expected_names != TORCH_PARAMETER_NAMES:
        raise AssertionError(
            f"unsupported torch signature changed to {expected_names!r}"
        )
    candidate_names = tuple(candidate_parameters)
    expected_candidate_names = TORCH_PARAMETER_NAMES + HYPERLOADER_PARAMETER_NAMES
    if candidate_names != expected_candidate_names:
        raise AssertionError(
            f"hyperloader parameters are {candidate_names!r}, expected "
            f"{expected_candidate_names!r}"
        )

    for name, reference_parameter in reference_parameters.items():
        candidate_parameter = candidate_parameters[name]
        if candidate_parameter.kind is not reference_parameter.kind:
            raise AssertionError(
                f"parameter {name!r} has kind {candidate_parameter.kind!r}, expected "
                f"{reference_parameter.kind!r}"
            )

    for name in HYPERLOADER_PARAMETER_NAMES:
        if candidate_parameters[name].kind is not inspect.Parameter.KEYWORD_ONLY:
            raise AssertionError(f"hyperloader parameter {name!r} must be keyword-only")


class SurfaceSignatureTest(unittest.TestCase):
    """Compare the installed public surface with the available torch reference."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("torch is unavailable for signature comparison") from error
        cls.reference = torch.utils.data.DataLoader

    def test_parameter_order_and_kinds_match_torch(self) -> None:
        assert_surface_compatible(self.reference, _candidate_surface())

    def test_named_native_default_deviations_are_explicit(self) -> None:
        parameters = inspect.signature(DataLoader).parameters

        self.assertIs(parameters["num_workers"].default, AUTO)
        self.assertIs(parameters["prefetch_factor"].default, AUTO)
        self.assertIs(parameters["persistent_workers"].default, True)

    def test_severed_public_path_is_detected(self) -> None:
        import hyperloader

        self.assertIs(hyperloader.DataLoader, DataLoader)

    def test_missing_torch_parameter_is_a_red_mutation(self) -> None:
        with self.assertRaisesRegex(AssertionError, "hyperloader parameters"):
            assert_surface_compatible(self.reference, _missing_in_order_surface)


if __name__ == "__main__":
    unittest.main()
