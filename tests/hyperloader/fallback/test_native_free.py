"""Installed-shape public execution with the native module severed."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class NativeFreePublicPathTest(unittest.TestCase):
    """Exercise the copied package without any extension binary."""

    def test_public_loader_runs_process_workers_and_coordinate_resume(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        source = repository / "python" / "hyperloader"
        script = textwrap.dedent(
            """
            import json
            import os

            import torch

            from hyperloader import DataLoader, HyperConfig
            from hyperloader import _hyperloader
            from hyperloader.config import SchedulerConfig
            from native_free_fixture import ArrayDataset, RandomDataset

            def freeze(value):
                if hasattr(value, "tolist"):
                    return value.tolist()
                if isinstance(value, (list, tuple)):
                    return [freeze(item) for item in value]
                if isinstance(value, dict):
                    return {key: freeze(item) for key, item in value.items()}
                return value

            def batches(loader):
                return [freeze(batch) for batch in loader]

            def main():
                assert _hyperloader.IS_FALLBACK is True
                config = HyperConfig(
                    scheduler=SchedulerConfig(profile_cache="off")
                )
                full_loader = DataLoader(
                    range(18),
                    batch_size=3,
                    shuffle=True,
                    num_workers=2,
                    seed=1701,
                    config=config,
                )
                try:
                    full = batches(full_loader)
                    worker_pids = full_loader._process_pool.worker_pids
                    assert len(worker_pids) == 2
                    assert all(pid != os.getpid() for pid in worker_pids)
                finally:
                    full_loader.close()

                cut_loader = DataLoader(
                    range(18),
                    batch_size=3,
                    shuffle=True,
                    num_workers=2,
                    seed=1701,
                    config=config,
                )
                iterator = iter(cut_loader)
                prefix = [next(iterator).tolist(), next(iterator).tolist()]
                state = cut_loader.state_dict()
                cut_loader.close()

                resumed_loader = DataLoader(
                    range(18),
                    batch_size=3,
                    shuffle=True,
                    num_workers=2,
                    seed=1701,
                    config=config,
                )
                resumed_loader.load_state_dict(state)
                try:
                    resumed = batches(resumed_loader)
                finally:
                    resumed_loader.close()
                assert prefix + resumed == full

                rng_two = DataLoader(
                    RandomDataset(12),
                    batch_size=3,
                    num_workers=2,
                    seed=91,
                    config=config,
                )
                rng_three = DataLoader(
                    RandomDataset(12),
                    batch_size=3,
                    num_workers=3,
                    seed=91,
                    config=config,
                )
                try:
                    assert batches(rng_two) == batches(rng_three)
                finally:
                    rng_two.close()
                    rng_three.close()

                array_loader = DataLoader(
                    ArrayDataset(9),
                    batch_size=4,
                    num_workers=2,
                    seed=19,
                    config=config,
                )
                try:
                    assert array_loader._process_pool.batch_size == 4
                    arrays = batches(array_loader)
                    assert [len(batch) for batch in arrays] == [4, 4, 1]
                finally:
                    array_loader.close()

                tensor_loader = DataLoader(
                    torch.arange(8),
                    batch_size=2,
                    num_workers=2,
                    seed=23,
                    config=config,
                )
                try:
                    assert batches(tensor_loader) == [[0, 1], [2, 3], [4, 5], [6, 7]]
                finally:
                    tensor_loader.close()

                sampler_loader = DataLoader(
                    range(10),
                    batch_size=2,
                    sampler=[5, 1, 7, 0],
                    num_workers=2,
                    seed=29,
                    config=config,
                )
                try:
                    assert batches(sampler_loader) == [[5, 1], [7, 0]]
                finally:
                    sampler_loader.close()

                error_loader = DataLoader(
                    range(2),
                    batch_size=1,
                    sampler=[0, 9],
                    num_workers=2,
                    seed=31,
                    config=config,
                )
                error_iterator = iter(error_loader)
                try:
                    assert next(error_iterator).tolist() == [0]
                    try:
                        next(error_iterator)
                    except IndexError as error:
                        assert "Caught IndexError in DataLoader worker process" in str(error)
                    else:
                        raise AssertionError("worker exception did not reach the owner")
                finally:
                    error_loader.close()

                print(json.dumps({"batches": full, "fallback": True}))

            if __name__ == "__main__":
                main()
            """
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "native_free_fixture.py").write_text(
                textwrap.dedent(
                    """
                    import random

                    import numpy as np
                    import torch


                    class ArrayDataset:
                        def __init__(self, length):
                            self.length = length

                        def __len__(self):
                            return self.length

                        def __getitem__(self, index):
                            return np.asarray([index, index + 1], dtype=np.int64)


                    class RandomDataset:
                        def __init__(self, length):
                            self.length = length

                        def __len__(self):
                            return self.length

                        def __getitem__(self, index):
                            return (
                                index,
                                random.random(),
                                int(np.random.randint(0, 1_000_000)),
                                int(torch.randint(0, 1_000_000, ()).item()),
                            )
                    """
                ),
                encoding="utf-8",
            )
            shutil.copytree(
                source,
                root / "hyperloader",
                ignore=shutil.ignore_patterns(
                    "_hyperloader*.pyd",
                    "_hyperloader*.so",
                    "_hyperloader*.dylib",
                    "_hyperloader*.pdb",
                    "__pycache__",
                ),
            )
            environment = os.environ.copy()
            inherited_path = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(root)
                if not inherited_path
                else os.pathsep.join((str(root), inherited_path))
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn('"fallback": true', completed.stdout)


if __name__ == "__main__":
    unittest.main()
