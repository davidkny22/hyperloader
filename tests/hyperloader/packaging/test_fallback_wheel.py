"""Universal fallback wheel contents and reproducibility checks."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[3]
BUILDER_PATH = ROOT / "tools" / "build_fallback_wheel.py"
VERIFIER_PATH = ROOT / "tools" / "verify_wheel.py"
SPEC = importlib.util.spec_from_file_location("fallback_wheel_builder_test", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {BUILDER_PATH}")
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)
VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "wheel_verifier_test", VERIFIER_PATH
)
if VERIFIER_SPEC is None or VERIFIER_SPEC.loader is None:
    raise RuntimeError(f"cannot load {VERIFIER_PATH}")
VERIFIER = importlib.util.module_from_spec(VERIFIER_SPEC)
sys.modules[VERIFIER_SPEC.name] = VERIFIER
VERIFIER_SPEC.loader.exec_module(VERIFIER)


class FallbackWheelTest(unittest.TestCase):
    """Require a deterministic pure wheel with no native or test payload."""

    def test_wheel_is_reproducible_bounded_and_product_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = BUILDER.build_wheel(ROOT, output).read_bytes()
            second = BUILDER.build_wheel(ROOT, output).read_bytes()
            self.assertEqual(first, second)
            self.assertLess(len(first), 5 * 1024 * 1024)
            path = output / "hyperloader-0.1.0-py3-none-any.whl"
            report = VERIFIER.verify_wheel(path, kind="fallback", root=ROOT)
            self.assertEqual(report["native_files"], 0)
            with zipfile.ZipFile(path) as wheel:
                names = wheel.namelist()
                self.assertIn("hyperloader/_hyperloader.py", names)
                self.assertFalse(any("/tests/" in name for name in names))
                self.assertFalse(
                    any(name.endswith((".so", ".pyd", ".dylib")) for name in names)
                )
                self.assertIn("Root-Is-Purelib: true", wheel.read(
                    "hyperloader-0.1.0.dist-info/WHEEL"
                ).decode())

    def test_build_graph_excludes_the_test_tree(self) -> None:
        reachable = VERIFIER.verify_build_graph(ROOT)
        self.assertTrue(reachable)
        self.assertFalse(any(Path(name).parts[0] == "tests" for name in reachable))

    def test_verifier_rejects_a_test_module_in_the_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = BUILDER.build_wheel(ROOT, output)
            with zipfile.ZipFile(path, "a") as wheel:
                wheel.writestr("hyperloader/tests/test_leak.py", b"raise AssertionError\n")
            with self.assertRaises(AssertionError):
                VERIFIER.verify_wheel(path, kind="fallback", root=ROOT)


if __name__ == "__main__":
    unittest.main()
