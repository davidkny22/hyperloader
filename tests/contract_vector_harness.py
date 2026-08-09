"""Verification helpers for the committed deterministic contract vectors."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).parents[1]
VECTOR_DIR = ROOT / "oracles" / "contract-vectors"
VECTOR_PATH = VECTOR_DIR / "vectors.json"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference = _load_module("sealed_contract_reference", VECTOR_DIR / "reference.py")


def load_document() -> dict[str, Any]:
    """Load and validate the artifact, optionally applying the planted mutation."""
    document = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
    reference.validate_document(document)
    if os.environ.get("HYPERLOADER_CONTRACT_MUTATION") == "flip-philox-word":
        document = copy.deepcopy(document)
        document["philox"]["vectors"][0]["words"][0] ^= 1
    return document


def permutation_digest(permutation: list[int]) -> str:
    """Hash a full materialized permutation with fixed-width little-endian words."""
    digest = hashlib.sha256()
    for value in permutation:
        digest.update(value.to_bytes(8, "little"))
    return digest.hexdigest()


def artifact_sha256() -> str:
    """Return the exact committed artifact hash."""
    return hashlib.sha256(VECTOR_PATH.read_bytes()).hexdigest()


def assert_installed_under(module_file: str, expected_root: str) -> None:
    """Reject imports that did not resolve from the isolated installation."""
    module_path = Path(module_file).resolve()
    root = Path(expected_root).resolve()
    if not module_path.is_relative_to(root):
        raise AssertionError(f"{module_path} is outside installed root {root}")
