from __future__ import annotations

import json
import time
import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from hyperloader import DataLoader
from hyperloader.config import HyperConfig, SchedulerConfig


class IntegerDataset:
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> int:
        return index


def timed_collate(rows: list[int], directory: str) -> list[int]:
    started_ns = time.perf_counter_ns()
    time.sleep(0.1)
    ended_ns = time.perf_counter_ns()
    record = {"started_ns": started_ns, "ended_ns": ended_ns}
    Path(directory, f"batch-{rows[0]}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    return rows


class UserCollationTest(unittest.TestCase):
    def test_custom_collation_batches_overlap_across_process_workers(self) -> None:
        with TemporaryDirectory() as directory:
            loader = DataLoader(
                IntegerDataset(),
                batch_size=2,
                num_workers=4,
                collate_fn=partial(timed_collate, directory=directory),
                config=HyperConfig(
                    scheduler=SchedulerConfig(
                        frontier_depth=8,
                        frontier_budget=1 << 20,
                        profile_cache="off",
                    )
                ),
            )
            try:
                self.assertEqual(
                    list(loader), [list(range(i, i + 2)) for i in range(0, 16, 2)]
                )
            finally:
                loader.close()

            intervals = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in Path(directory).glob("batch-*.json")
            ]

        events = sorted(
            [(row["started_ns"], 1) for row in intervals]
            + [(row["ended_ns"], -1) for row in intervals]
        )
        active = 0
        maximum = 0
        for _, change in events:
            active += change
            maximum = max(maximum, active)
        self.assertGreaterEqual(maximum, 2)


if __name__ == "__main__":
    unittest.main()
