"""Installed public gate for the three scoped cross-tier parity claims."""

from __future__ import annotations

import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import hyperloader
import torch
from hyperloader import DataLoader, verify
from hyperloader.verify import _bit_equal

from .support import AccessorDataset, AmbientDataset, CONFIG, NestedDataset

SEEDS = (0, 1, 61, 2**63 + 9)


class TierParityGate(unittest.TestCase):
    """Compare decoded values through every declared execution boundary."""

    def test_process_and_severed_fallback_match_admissible_streams(self) -> None:
        native, fallback = _run_process_and_fallback()
        self.assertEqual(list(native), list(fallback))
        mismatches = [
            name for name in native if not _bit_equal(native[name], fallback[name])
        ]
        self.assertEqual(mismatches, [])

    def test_thread_and_process_match_provided_generator_code(self) -> None:
        for seed in SEEDS:
            with self.subTest(seed=seed):
                self.assertEqual(
                    verify(AccessorDataset(), samples=9, seed=seed, num_workers=2),
                    {
                        "bit_exact": True,
                        "compared_samples": 9,
                        "first_mismatch": None,
                    },
                )
                process = DataLoader(
                    AccessorDataset(),
                    batch_size=3,
                    shuffle=True,
                    num_workers=2,
                    seed=seed,
                    config=CONFIG,
                )
                threaded = DataLoader(
                    AccessorDataset(),
                    batch_size=3,
                    shuffle=True,
                    num_workers=2,
                    seed=seed,
                    thread_safe=True,
                    config=CONFIG,
                )
                try:
                    self.assertTrue(_bit_equal(list(process), list(threaded)))
                finally:
                    process.close()
                    threaded.close()

    def test_in_process_matches_process_and_restores_globals(self) -> None:
        import numpy as np

        for seed in SEEDS:
            with self.subTest(seed=seed):
                process = DataLoader(
                    NestedDataset(),
                    batch_size=3,
                    shuffle=True,
                    num_workers=2,
                    seed=seed,
                    config=CONFIG,
                )
                local = DataLoader(
                    NestedDataset(),
                    batch_size=3,
                    shuffle=True,
                    num_workers=0,
                    seed=seed,
                    config=CONFIG,
                )
                random_state = random.getstate()
                numpy_state = np.random.get_state()
                torch_state = torch.default_generator.get_state()
                try:
                    self.assertTrue(_bit_equal(list(process), list(local)))
                    self.assertEqual(random.getstate(), random_state)
                    current_numpy = np.random.get_state()
                    self.assertEqual(current_numpy[0], numpy_state[0])
                    self.assertTrue(np.array_equal(current_numpy[1], numpy_state[1]))
                    self.assertEqual(current_numpy[2:], numpy_state[2:])
                    self.assertTrue(
                        torch.equal(torch.default_generator.get_state(), torch_state)
                    )
                finally:
                    process.close()
                    local.close()

    def test_named_assumptions_fail_closed(self) -> None:
        ambient = verify(AmbientDataset(), samples=6, seed=109, num_workers=2)
        self.assertFalse(ambient["bit_exact"])
        self.assertFalse(_bit_equal((1, 2), [1, 2]))
        view = torch.arange(10)[:3]
        self.assertTrue(_bit_equal(view, view.clone()))
        left = torch.tensor([1.0, 2.0])
        right = left.clone()
        right.view(torch.uint8)[0] ^= 1
        self.assertFalse(_bit_equal(left, right))


def _run_process_and_fallback() -> tuple[object, object]:
    repository = Path(__file__).resolve().parents[3]
    source = repository / "python" / "hyperloader"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fallback_root = root / "fallback"
        shutil.copytree(
            source,
            fallback_root / "hyperloader",
            ignore=shutil.ignore_patterns(
                "_hyperloader*.pyd",
                "_hyperloader*.so",
                "_hyperloader*.dylib",
                "_hyperloader*.pdb",
                "__pycache__",
            ),
        )
        native_path = root / "native.pkl"
        fallback_path = root / "fallback.pkl"
        native_root = Path(hyperloader.__file__).resolve().parent.parent
        _run_tier(repository, native_root, native_path, fallback=False)
        _run_tier(repository, fallback_root, fallback_path, fallback=True)
        return pickle.loads(native_path.read_bytes()), pickle.loads(
            fallback_path.read_bytes()
        )


def _run_tier(
    repository: Path,
    package_root: Path,
    output: Path,
    *,
    fallback: bool,
) -> None:
    environment = os.environ.copy()
    mutation = environment.pop("HYPERLOADER_PARITY_MUTATE", None)
    inherited = environment.get("PYTHONPATH")
    paths = [str(package_root), str(repository)]
    if inherited:
        paths.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    command = [
        sys.executable,
        "-m",
        "tests.hyperloader.parity.runner",
        "--output",
        str(output),
    ]
    if fallback:
        command.append("--fallback")
        if mutation == "1":
            environment["HYPERLOADER_PARITY_MUTATE"] = "1"
    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"fallback={fallback} failed\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


if __name__ == "__main__":
    unittest.main()
