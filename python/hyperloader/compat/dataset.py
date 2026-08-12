"""Dataset adapters that expose pre-fetch compat lane state."""

from __future__ import annotations

from typing import Any

import torch

from .protocol import TaggedIndex, TaggedSample
from .worker import capture_worker_state


class MapDatasetAdapter:
    """Tag map-style samples without changing the worker-visible dataset."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, tagged: TaggedIndex) -> TaggedSample:
        if tagged.dummy:
            return TaggedSample(tagged.batch, None, capture_worker_state(), True)
        state = capture_worker_state() if tagged.offset == 0 else None
        return TaggedSample(tagged.batch, self.dataset[tagged.index], state)


class IterableDatasetAdapter(torch.utils.data.IterableDataset):
    """Tag lane-local iterable batches at their pre-fetch RNG boundary."""

    def __init__(self, dataset: Any, batch_size: int | None, workers: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            raise RuntimeError("compat iterable adapter requires a worker process")
        source = iter(self.dataset)
        width = self.batch_size or 1
        sample = 0
        while True:
            offset = sample % width
            local_batch = sample // width
            batch = local_batch * self.workers + info.id
            state = capture_worker_state() if offset == 0 else None
            try:
                value = next(source)
            except StopIteration:
                return
            yield TaggedSample(batch, value, state)
            sample += 1
