"""Pinned source registration and staging-pool fallback checks."""

from __future__ import annotations

import gc
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from hyperloader.config import FactorConfig, MemoryConfig
from hyperloader.memory.pinned.delivery import PinnedDelivery
from hyperloader.memory.pinned.pool import PinnedTensorPool
from hyperloader.planner import TensorPlan


class _Runtime:
    def __init__(self, register_result: int = 0) -> None:
        self.register_result = register_result
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []

    def cudaHostRegister(self, pointer: int, size: int, flags: int) -> int:
        self.registered.append((pointer, size, flags))
        return self.register_result

    def cudaHostUnregister(self, pointer: int) -> int:
        self.unregistered.append(pointer)
        return 0


class _PinnableBatch:
    def __init__(self) -> None:
        self.calls = 0

    def pin_memory(self) -> str:
        self.calls += 1
        return "pinned-batch"


class PinnedDeliveryTest(unittest.TestCase):
    """Prove selection, write-free identity, refusal fallback, and held buffers."""

    def test_auto_registers_view_source_without_writing_or_changing_identity(
        self,
    ) -> None:
        source = torch.arange(32)
        runtime = _Runtime()
        loader = _loader(source)
        version = source._version
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.cudart", return_value=runtime),
        ):
            delivery = PinnedDelivery(loader)
            batch = source[:8]
            delivered = delivery.stage(batch)

            self.assertIs(delivered, batch)
            self.assertEqual(delivered.untyped_storage().data_ptr(), source.data_ptr())
            self.assertEqual(source._version, version)
            self.assertEqual(delivery.report()["pinned_staged_bytes"], 0)
            delivery.close()

        self.assertEqual(len(runtime.registered), 1)
        self.assertEqual(runtime.unregistered, [source.data_ptr()])

    def test_registration_refusal_uses_one_copy_staging_pool(self) -> None:
        source = torch.arange(16)
        runtime = _Runtime(register_result=1)
        loader = _loader(source)
        empty_strided = torch.empty_strided
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.cudart", return_value=runtime),
            mock.patch(
                "torch.empty_strided",
                side_effect=lambda shape, stride, **options: empty_strided(
                    shape, stride, dtype=options["dtype"]
                ),
            ),
        ):
            delivery = PinnedDelivery(loader)
            delivered = delivery.stage(source[:8])

        self.assertTrue(delivery.stages)
        self.assertTrue(torch.equal(delivered, source[:8]))
        self.assertNotEqual(delivered.data_ptr(), source.data_ptr())
        self.assertEqual(
            delivery.report()["pinned_staged_bytes"], 8 * source.element_size()
        )

    def test_held_staged_outputs_receive_distinct_buffers_then_reuse(self) -> None:
        pool = PinnedTensorPool()
        source = torch.arange(8)
        empty_strided = torch.empty_strided
        with mock.patch(
            "torch.empty_strided",
            side_effect=lambda shape, stride, **options: empty_strided(
                shape, stride, dtype=options["dtype"]
            ),
        ):
            first = pool.stage(source)
            second = pool.stage(source + 1)
            first_pointer = first.data_ptr()
            self.assertNotEqual(first_pointer, second.data_ptr())
            del first
            gc.collect()
            third = pool.stage(source + 2)

        self.assertEqual(third.data_ptr(), first_pointer)

    def test_custom_batch_uses_the_public_pinning_protocol(self) -> None:
        pool = PinnedTensorPool()
        batch = _PinnableBatch()

        delivered = pool.stage(batch)

        self.assertEqual(delivered, "pinned-batch")
        self.assertEqual(batch.calls, 1)

    def test_delivery_stage_bytes_compose_with_unequal_execution_components(
        self,
    ) -> None:
        source = torch.arange(8)
        runtime = _Runtime(register_result=1)
        loader = _loader(source)
        empty_strided = torch.empty_strided
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.cudart", return_value=runtime),
            mock.patch(
                "torch.empty_strided",
                side_effect=lambda shape, stride, **options: empty_strided(
                    shape, stride, dtype=options["dtype"]
                ),
            ),
        ):
            delivery = PinnedDelivery(loader)
            delivery.stage(source)

        memory = {
            "actual_bytes": 30,
            "actual_bytes_per_sample": 15.0,
            "bytes_beyond_irreducible": 0,
            "bytes_beyond_irreducible_per_sample": 0.0,
            "irreducible_bytes": 30,
            "pinned_stage_bytes": 10,
            "produced_samples": 2,
        }
        delivery.compose_memory_report(memory)

        self.assertEqual(memory["pinned_stage_bytes"], 10)
        self.assertEqual(memory["pinned_staged_bytes"], 64)
        self.assertEqual(memory["actual_bytes"], 94)
        self.assertEqual(memory["actual_bytes_per_sample"], 47.0)
        self.assertEqual(memory["bytes_beyond_irreducible"], 64)
        self.assertEqual(memory["bytes_beyond_irreducible_per_sample"], 32.0)

    def test_auto_without_measured_tax_remains_host_delivery(self) -> None:
        loader = _loader(torch.arange(8), staged_copy_tax=None)

        delivery = PinnedDelivery(loader)

        self.assertEqual(delivery.effective_memory, "host")
        self.assertFalse(delivery.stages)


def _loader(
    source: torch.Tensor, staged_copy_tax: object = object()
) -> SimpleNamespace:
    tax = SimpleNamespace(loss_fraction=0.1) if staged_copy_tax is not None else None
    return SimpleNamespace(
        dataset=source,
        delivery_memory=MemoryConfig().delivery_memory,
        _plan=TensorPlan(length=len(source), shuffle=False),
        _execution_dataset=source,
        _calibration=SimpleNamespace(staged_copy_tax=tax),
        config=SimpleNamespace(factors=FactorConfig()),
    )


if __name__ == "__main__":
    unittest.main()
