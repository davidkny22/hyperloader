"""Strict serial iterator over the persistent black-box process executor."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .pool import ProcessPool


class ProcessIterator(Iterator[Any]):
    """Consume map-style positions while retaining the loader's worker pool."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._epoch = loader._epoch
        self._position = 0
        self._length = len(loader.dataset)
        self._complete = False
        if self._length and loader._process_pool is None:
            loader._process_pool = ProcessPool(
                loader.dataset,
                loader.num_workers,
                loader.root_seed,
                self._epoch,
                0,
                0,
                worker_init_fn=loader.worker_init_fn,
                multiprocessing_context=loader.multiprocessing_context,
                timeout=loader.timeout,
            )

    def __iter__(self) -> ProcessIterator:
        return self

    def __next__(self) -> Any:
        if self._position >= self._length:
            self._finish_epoch()
            raise StopIteration
        batch_size = self._loader.batch_size
        if batch_size is None:
            position = self._position
            self._position += 1
            return self._execute(position)
        stop = min(self._position + batch_size, self._length)
        if self._loader.drop_last and stop - self._position < batch_size:
            self._position = self._length
            self._finish_epoch()
            raise StopIteration
        batch = [self._execute(position) for position in range(self._position, stop)]
        self._position = stop
        return self._loader._collate_batch(batch)

    def _execute(self, position: int) -> Any:
        return self._loader._process_pool.execute(self._epoch, position, position)

    def _finish_epoch(self) -> None:
        if not self._complete:
            self._loader._epoch += 1
            self._complete = True
