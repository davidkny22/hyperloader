from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
VERIFIER_PATH = ROOT / "tools" / "verify_licenses.py"
SPEC = importlib.util.spec_from_file_location("release_license_verifier_test", VERIFIER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {VERIFIER_PATH}")
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)


def test_license_audit_accepts_reviewed_expressions() -> None:
    report = VERIFIER.audit_metadata(
        {
            "packages": [
                {"name": "hyperloader-engine", "license": "MIT"},
                {"name": "dependency", "license": "MIT OR Apache-2.0"},
            ]
        }
    )
    assert report == {"packages": 2, "license_expressions": 2}


def test_license_audit_rejects_an_unreviewed_inclusion() -> None:
    with pytest.raises(AssertionError, match="dependency"):
        VERIFIER.audit_metadata(
            {
                "packages": [
                    {"name": "hyperloader-engine", "license": "MIT"},
                    {"name": "dependency", "license": "Proprietary"},
                ]
            }
        )
