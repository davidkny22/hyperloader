"""Worker-side native batch materialization checks."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from hyperloader.process.batching import encode_batch, supports_worker_batch
from hyperloader.process.serialization import ResultDecoder, ResultEncoder


class WorkerBatchTest(unittest.TestCase):
    """Verify one encoded result retains default-collation behavior."""

    def test_numpy_rows_cross_as_one_tensor_storage(self) -> None:
        payload = encode_batch(
            [np.arange(4, dtype=np.int64), np.arange(4, 8, dtype=np.int64)],
            ResultEncoder(),
        )

        batch = ResultDecoder().decode(payload, worker=0)

        self.assertTrue(
            torch.equal(batch, torch.arange(8, dtype=torch.int64).reshape(2, 4))
        )

    def test_probe_accepts_only_exact_contiguous_base_arrays(self) -> None:
        base = np.arange(8, dtype=np.int64)

        self.assertTrue(supports_worker_batch(base))
        self.assertFalse(supports_worker_batch(base[::2]))
        self.assertFalse(supports_worker_batch(torch.arange(8)))

    def test_tensor_rows_preserve_default_collation(self) -> None:
        payload = encode_batch(
            [torch.arange(4), torch.arange(4, 8)], ResultEncoder()
        )

        batch = ResultDecoder().decode(payload, worker=0)

        self.assertTrue(torch.equal(batch, torch.arange(8).reshape(2, 4)))

    def test_incompatible_rows_raise_the_native_collation_error(self) -> None:
        with self.assertRaises(RuntimeError):
            encode_batch(
                [np.arange(4, dtype=np.int64), np.arange(6, dtype=np.int64)],
                ResultEncoder(),
            )


if __name__ == "__main__":
    unittest.main()
