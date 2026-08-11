"""Installed public gate for padded-tail collective liveness under gloo."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import hyperloader
import torch
from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import SchedulerConfig
from torch import distributed, multiprocessing


class TailDataset:
    """Expose a map length that leaves a global-batch tail."""

    def __len__(self) -> int:
        return 13

    def __getitem__(self, index: int) -> int:
        return index


def _loader() -> DataLoader:
    return DataLoader(
        TailDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=2,
        thread_safe=True,
        seed=719,
        config=HyperConfig(scheduler=SchedulerConfig(profile_cache="off")),
    )


def _gloo_tail_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    evidence_dir: str,
    mutate_tail: bool,
) -> None:
    distributed.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=8),
    )
    loader: DataLoader | None = None
    try:
        loader = _loader()
        placement = loader._map_placement
        if (placement.rank, placement.world_size) != (rank, world_size):
            raise AssertionError("AUTO topology did not capture the initialized rank")
        if mutate_tail and rank == 0:
            loader._map_placement = replace(placement, drop_last=True)

        local_batches = []
        global_batches = []
        for batch in loader:
            values = [int(value) for value in batch.tolist()]
            local_batches.append(values)
            participants = torch.tensor([1], dtype=torch.int64)
            distributed.all_reduce(participants)
            if int(participants.item()) != world_size:
                raise AssertionError("a tail collective omitted a rank")
            gathered: list[Any] = [None] * world_size
            distributed.all_gather_object(gathered, values)
            if rank == 0:
                global_batches.append(gathered)

        counts: list[Any] = [None] * world_size
        distributed.all_gather_object(counts, len(local_batches))
        expected_batches = (len(TailDataset()) + 2 * world_size - 1) // (2 * world_size)
        if counts != [expected_batches] * world_size:
            raise AssertionError(f"rank batch counts diverged: {counts}")
        if rank == 0:
            Path(evidence_dir, f"world-{world_size}.json").write_text(
                json.dumps(
                    {
                        "global_batch": 2 * world_size,
                        "global_batches": global_batches,
                        "rank_batch_counts": counts,
                        "world_size": world_size,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        if loader is not None:
            loader.close()
        if distributed.is_initialized():
            distributed.destroy_process_group()


def _late_initialization_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    evidence_dir: str,
) -> None:
    loader = _loader()
    try:
        distributed.init_process_group(
            backend="gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=8),
        )
        try:
            iter(loader)
        except RuntimeError as error:
            message = str(error)
            if "captured rank=0, world_size=1" not in message:
                raise AssertionError(message) from error
            if f"current rank={rank}, world_size={world_size}" not in message:
                raise AssertionError(message) from error
            Path(evidence_dir, f"late-rank-{rank}.txt").write_text(
                message + "\n", encoding="utf-8"
            )
        else:
            raise AssertionError("late distributed initialization was accepted")
    finally:
        loader.close()
        if distributed.is_initialized():
            distributed.destroy_process_group()


class DdpTailLivenessGate(unittest.TestCase):
    """Require one collective participation from every rank per padded batch."""

    @unittest.skipUnless(
        distributed.is_available() and distributed.is_gloo_available(),
        "gloo is required",
    )
    def test_padded_tail_keeps_all_ranks_in_collective_sequence(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is not None:
            root = Path(expected_root).resolve()
            self.assertTrue(Path(hyperloader.__file__).resolve().is_relative_to(root))
            self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))

        mutate_tail = (
            os.environ.get("HYPERLOADER_DDP_TAIL_MUTATION") == "drop-padded-tail"
        )
        evidence = []
        with tempfile.TemporaryDirectory() as directory:
            for world_size in (2, 4):
                with self.subTest(world_size=world_size):
                    rendezvous_path = Path(directory, f"gloo-{world_size}").resolve()
                    multiprocessing.spawn(
                        _gloo_tail_worker,
                        args=(
                            world_size,
                            rendezvous_path.as_uri(),
                            directory,
                            mutate_tail,
                        ),
                        nprocs=world_size,
                        join=True,
                    )
                    evidence.append(
                        json.loads(
                            Path(directory, f"world-{world_size}.json").read_text(
                                encoding="utf-8"
                            )
                        )
                    )

        metrics_path = os.environ.get("HYPERLOADER_DDP_TAIL_METRICS")
        if metrics_path is not None:
            Path(metrics_path).write_text(
                json.dumps({"topologies": evidence}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    @unittest.skipUnless(
        distributed.is_available() and distributed.is_gloo_available(),
        "gloo is required",
    )
    def test_late_process_group_initialization_names_the_changed_world(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory, "late-gloo").resolve().as_uri()
            multiprocessing.spawn(
                _late_initialization_worker,
                args=(2, rendezvous, directory),
                nprocs=2,
                join=True,
            )
            messages = [
                Path(directory, f"late-rank-{rank}.txt").read_text(encoding="utf-8")
                for rank in range(2)
            ]
        self.assertTrue(all("world_size=2" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
