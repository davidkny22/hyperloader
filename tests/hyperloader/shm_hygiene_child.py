"""Crash an owning process while one public loader worker is blocked."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from hyperloader import DataLoader


class BlockingDataset:
    """Keep the second routed worker inside user code until its owner dies."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        if index == 1:
            time.sleep(60)
        return index


def main() -> None:
    """Persist crash context, then bypass every Python cleanup hook."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    arguments = parser.parse_args()

    loader = DataLoader(BlockingDataset(), batch_size=1, num_workers=2, seed=71)
    iterator = iter(loader)
    if int(next(iterator).item()) != 0:
        raise RuntimeError("parent-crash setup did not deliver its first sample")
    time.sleep(0.1)
    registry = (
        Path(os.environ["HYPERLOADER_CACHE_HOME"])
        / "hyperloader"
        / "regions.jsonl"
    )
    arguments.state.write_text(
        json.dumps(
            {
                "parent": os.getpid(),
                "registry": str(registry),
                "workers": loader._process_pool.worker_pids,
            }
        ),
        encoding="utf-8",
    )
    os._exit(23)


if __name__ == "__main__":
    main()
