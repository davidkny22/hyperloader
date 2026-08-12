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
SPEC = importlib.util.spec_from_file_location("fallback_wheel_builder_test", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {BUILDER_PATH}")
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


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
            with zipfile.ZipFile(path) as wheel:
                names = wheel.namelist()
                self.assertIn("hyperloader/_hyperloader.py", names)
                self.assertIn("hyperloader-0.1.0.dist-info/licenses/LICENSE", names)
                self.assertFalse(any("/tests/" in name for name in names))
                self.assertFalse(
                    any(name.endswith((".so", ".pyd", ".dylib")) for name in names)
                )
                self.assertIn("Root-Is-Purelib: true", wheel.read(
                    "hyperloader-0.1.0.dist-info/WHEEL"
                ).decode())


if __name__ == "__main__":
    unittest.main()
