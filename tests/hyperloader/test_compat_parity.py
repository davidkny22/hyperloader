"""Installed public-path parity against pinned Torch golden streams."""

from __future__ import annotations

import os
import platform
import unittest
from pathlib import Path
from unittest.mock import patch

import hyperloader
import torch

from benches.compat_candidate_cases import generate_candidate_cases
from benches.compat_golden_model import canonical_system, read_document
from benches.verify_compat_golden import first_difference


class CompatParityGateTest(unittest.TestCase):
    """Compare every supported strict-order stream through the public loader."""

    def test_installed_public_loader_matches_the_platform_minor_oracle(self) -> None:
        _require_expected_install()
        expected = read_document(_artifact_path())
        actual = generate_candidate_cases()
        self.assertEqual(
            actual,
            expected["cases"],
            f"compat stream differs at {first_difference(expected['cases'], actual)}",
        )

    def test_changed_torch_minor_cannot_select_an_oracle(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no pinned Torch golden artifact"):
            _artifact_path(torch_minor="0.0")

    def test_gate_rejects_an_unpinned_product_import(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError, "HYPERLOADER_EXPECTED_INSTALL is required"
            ):
                _require_expected_install()
        with patch.dict(
            os.environ, {"HYPERLOADER_EXPECTED_INSTALL": str(Path.cwd())}
        ):
            with self.assertRaisesRegex(RuntimeError, "expected"):
                _require_expected_install()


def _artifact_path(*, torch_minor: str | None = None) -> Path:
    root = Path(__file__).parents[2]
    system = canonical_system(platform.system())
    minor = torch_minor or ".".join(torch.__version__.split("+")[0].split(".")[:2])
    path = root / "oracles" / "torch-golden" / system / f"torch-{minor}.json"
    if not path.is_file():
        raise RuntimeError(
            f"no pinned Torch golden artifact for {platform.system()} Torch {minor}"
        )
    return path


def _require_expected_install() -> None:
    expected = os.environ.get("HYPERLOADER_EXPECTED_INSTALL")
    if expected is None:
        raise RuntimeError("HYPERLOADER_EXPECTED_INSTALL is required")
    package_root = Path(hyperloader.__file__).resolve().parent.parent
    if package_root != Path(expected).resolve():
        raise RuntimeError(
            f"gate imported hyperloader from {package_root}, expected {Path(expected).resolve()}"
        )


if __name__ == "__main__":
    unittest.main()
