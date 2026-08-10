"""Typed pipeline construction, validation, and routing tests."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from hyperloader import (
    Collate,
    DataLoader,
    Decode,
    Source,
    StageIO,
    ThreadSafety,
    Transform,
    pipeline,
)
from hyperloader.planner import StagePlan, build_plan


def decode_integer(value: bytes) -> int:
    """Decode one ASCII integer for spawn-safe execution tests."""
    return int(value)


def double_integer(value: int) -> int:
    """Transform one integer for spawn-safe execution tests."""
    return value * 2


def sum_batch(values: list[int]) -> int:
    """Collate one integer batch for spawn-safe execution tests."""
    return sum(values)


def build_integer_pipeline(*, thread_safe: bool = False):  # type: ignore[no-untyped-def]
    """Build the reusable typed pipeline exercised by public-path tests."""
    safety = ThreadSafety.THREAD_SAFE if thread_safe else ThreadSafety.ISOLATED
    return pipeline(
        Source(
            [b"1", b"2", b"3", b"4"],
            output_type=bytes,
            io=StageIO.READ,
            thread_safety=safety,
            cost_hint_ns=11,
        ),
        Decode(
            decode_integer,
            input_type=bytes,
            output_type=int,
            thread_safety=safety,
            cost_hint_ns=13,
        ),
        Transform(
            double_integer,
            input_type=int,
            output_type=int,
            thread_safety=safety,
            cost_hint_ns=17,
        ),
        Collate(sum_batch, input_type=int, output_type=int, cost_hint_ns=19),
    )


class PipelineContractTest(unittest.TestCase):
    """Exercise immutable declarations and composition-time validation."""

    def test_pipeline_executes_source_sample_and_batch_stages(self) -> None:
        dataset = build_integer_pipeline()

        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset[2], 6)
        self.assertEqual(dataset.collate([2, 4]), 6)
        self.assertEqual(len(dataset.stages), 4)

    def test_stage_declarations_are_normalized_and_immutable(self) -> None:
        stage = Source(
            [1],
            output_type=int,
            io="none",
            thread_safety="thread-safe",
            cost_hint_ns=5,
        )

        self.assertIs(stage.io, StageIO.NONE)
        self.assertIs(stage.thread_safety, ThreadSafety.THREAD_SAFE)
        with self.assertRaises(FrozenInstanceError):
            stage.length = 2  # type: ignore[misc]

    def test_callable_source_requires_an_explicit_length(self) -> None:
        with self.assertRaisesRegex(TypeError, "explicit length"):
            Source(lambda index: index, output_type=int)

    def test_invalid_cost_hint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            Transform(double_integer, cost_hint_ns=0)

    def test_incompatible_type_edge_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "bytes -> int"):
            pipeline(
                Source([b"1"], output_type=bytes),
                Transform(double_integer, input_type=int, output_type=int),
                Collate(sum_batch, input_type=int, output_type=int),
            )

    def test_decode_cannot_follow_a_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "Decode must appear"):
            pipeline(
                Source([1], output_type=int),
                Transform(double_integer, input_type=int, output_type=int),
                Decode(str, input_type=int, output_type=str),
                Collate(list, input_type=str, output_type=list),
            )

    def test_terminal_stage_must_be_collate(self) -> None:
        with self.assertRaisesRegex(TypeError, "final pipeline stage"):
            pipeline(Source([1], output_type=int), Transform(double_integer))


class PipelineRoutingTest(unittest.TestCase):
    """Exercise plan selection and installed DataLoader routing behavior."""

    def test_pipeline_selects_stage_plan(self) -> None:
        plan = build_plan(build_integer_pipeline(thread_safe=True), False)

        self.assertIsInstance(plan, StagePlan)
        self.assertTrue(plan.sample_thread_safe)
        self.assertEqual(plan.index(7, 3, 2), 2)

    def test_one_isolated_stage_keeps_the_sample_chain_in_processes(self) -> None:
        dataset = build_integer_pipeline(thread_safe=True)
        isolated = pipeline(
            dataset.source,
            *dataset.sample_stages,
            Transform(double_integer, input_type=int, output_type=int),
            dataset.collate_stage,
        )
        plan = build_plan(isolated, False)

        self.assertIsInstance(plan, StagePlan)
        self.assertFalse(plan.sample_thread_safe)

    def test_process_route_uses_pipeline_collation(self) -> None:
        loader = DataLoader(
            build_integer_pipeline(), batch_size=2, num_workers=2, seed=29
        )
        try:
            self.assertEqual(list(loader), [6, 14])
            self.assertIsNotNone(loader._process_pool)
            self.assertIsNone(loader._thread_pool)
        finally:
            loader.close()

    def test_declared_chain_uses_thread_route(self) -> None:
        loader = DataLoader(
            build_integer_pipeline(thread_safe=True),
            batch_size=2,
            num_workers=2,
            seed=31,
        )
        try:
            self.assertEqual(list(loader), [6, 14])
            self.assertIsNotNone(loader._thread_pool)
            self.assertIsNone(loader._process_pool)
        finally:
            loader.close()

    def test_pipeline_collate_conflicts_with_loader_collate(self) -> None:
        with self.assertRaisesRegex(ValueError, "pipeline Collate"):
            DataLoader(build_integer_pipeline(), collate_fn=list)

    def test_loader_declaration_cannot_override_an_isolated_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "isolated pipeline sample stage"):
            DataLoader(build_integer_pipeline(), thread_safe=True)


if __name__ == "__main__":
    unittest.main()
