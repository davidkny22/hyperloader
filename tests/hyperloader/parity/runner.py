"""Subprocess runner that records one installed tier's public streams."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import torch
from hyperloader import DataLoader, _hyperloader

from .support import (
    CONFIG,
    ArrayDataset,
    FailingDataset,
    FixedBatchSampler,
    FixedSampler,
    NestedDataset,
    ParityIterable,
    build_pipeline,
    collect,
    distributed_config,
    failing_collate,
    seeded_collate,
    single_collate,
)

SEEDS = (0, 1, 61, 2**63 + 9)


def build_report() -> dict[str, Any]:
    """Execute the admissible process-versus-fallback scenario set."""
    report: dict[str, Any] = {
        f"nested:{seed}": collect(
            DataLoader(
                NestedDataset(),
                batch_size=3,
                shuffle=True,
                num_workers=2,
                seed=seed,
                config=CONFIG,
            )
        )
        for seed in SEEDS
    }
    report["array"] = collect(
        DataLoader(
            ArrayDataset(),
            batch_size=4,
            num_workers=2,
            seed=71,
            config=CONFIG,
        )
    )
    report["drop_last"] = collect(
        DataLoader(
            NestedDataset(),
            batch_size=4,
            drop_last=True,
            num_workers=2,
            seed=69,
            config=CONFIG,
        )
    )
    report["unbatched"] = collect(
        DataLoader(
            NestedDataset(),
            batch_size=None,
            num_workers=2,
            seed=70,
            config=CONFIG,
        )
    )
    report["tensor"] = collect(
        DataLoader(
            torch.arange(10),
            batch_size=3,
            num_workers=2,
            seed=73,
            config=CONFIG,
        )
    )
    report["sampler"] = collect(
        DataLoader(
            range(10),
            batch_size=2,
            sampler=FixedSampler(),
            num_workers=2,
            seed=79,
            config=CONFIG,
        )
    )
    report["batch_sampler"] = collect(
        DataLoader(
            range(10),
            batch_sampler=FixedBatchSampler(),
            num_workers=2,
            seed=83,
            config=CONFIG,
        )
    )
    report["pipeline"] = collect(
        DataLoader(
            build_pipeline(),
            batch_size=3,
            num_workers=2,
            seed=89,
            config=CONFIG,
        )
    )
    report["collate"] = collect(
        DataLoader(
            NestedDataset(),
            batch_size=3,
            num_workers=2,
            seed=91,
            collate_fn=seeded_collate,
            config=CONFIG,
        )
    )
    report["single_collate"] = collect(
        DataLoader(
            range(4),
            batch_size=None,
            num_workers=2,
            seed=92,
            collate_fn=single_collate,
            config=CONFIG,
        )
    )
    report["sampler_collate"] = collect(
        DataLoader(
            range(10),
            batch_size=2,
            sampler=FixedSampler(),
            num_workers=2,
            seed=93,
            collate_fn=seeded_collate,
            config=CONFIG,
        )
    )
    report["distributed"] = {
        rank: collect(
            DataLoader(
                NestedDataset(),
                batch_size=2,
                shuffle=True,
                num_workers=2,
                seed=94,
                config=distributed_config(rank),
            )
        )
        for rank in range(3)
    }
    completion = collect(
        DataLoader(
            range(12),
            batch_size=2,
            num_workers=2,
            seed=95,
            in_order=False,
            config=CONFIG,
        )
    )
    report["completion_content"] = sorted(
        tuple(int(value) for value in batch.tolist()) for batch in completion
    )
    report["iterable"] = collect(
        DataLoader(
            ParityIterable(),
            batch_size=2,
            num_workers=2,
            seed=97,
            config=CONFIG,
        )
    )
    report["resume"] = _resume_stream()
    report["exception"] = _exception_record()
    report["collate_exception"] = _collate_exception_record()
    if os.environ.get("HYPERLOADER_PARITY_MUTATE") == "1":
        report["tensor"][0][0] += 1
    return report


def _resume_stream() -> list[Any]:
    source = DataLoader(
        NestedDataset(),
        batch_size=3,
        shuffle=True,
        num_workers=2,
        seed=101,
        config=CONFIG,
    )
    iterator = iter(source)
    prefix = [next(iterator), next(iterator)]
    state = source.state_dict()
    source.close()
    resumed = DataLoader(
        NestedDataset(),
        batch_size=3,
        shuffle=True,
        num_workers=2,
        seed=103,
        config=CONFIG,
    )
    resumed.load_state_dict(state)
    return prefix + collect(resumed)


def _exception_record() -> tuple[str, str]:
    loader = DataLoader(
        FailingDataset(),
        batch_size=2,
        num_workers=2,
        seed=107,
        config=CONFIG,
    )
    try:
        list(loader)
    except BaseException as error:
        return type(error).__name__, "parity sentinel" if "parity sentinel" in str(
            error
        ) else str(error)
    finally:
        loader.close()
    raise AssertionError("exception scenario did not raise")


def _collate_exception_record() -> tuple[str, str]:
    loader = DataLoader(
        range(4),
        batch_size=2,
        num_workers=2,
        seed=108,
        collate_fn=failing_collate,
        config=CONFIG,
    )
    try:
        list(loader)
    except BaseException as error:
        marker = (
            "collate parity sentinel"
            if "collate parity sentinel" in str(error)
            else str(error)
        )
        return type(error).__name__, marker
    finally:
        loader.close()
    raise AssertionError("collation exception scenario did not raise")


def main() -> None:
    """Validate tier selection and serialize decoded public values."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fallback", action="store_true")
    arguments = parser.parse_args()
    actual_fallback = bool(getattr(_hyperloader, "IS_FALLBACK", False))
    if actual_fallback != arguments.fallback:
        raise AssertionError(
            f"expected fallback={arguments.fallback}, resolved {actual_fallback}"
        )
    arguments.output.write_bytes(pickle.dumps(build_report(), protocol=5))


if __name__ == "__main__":
    main()
