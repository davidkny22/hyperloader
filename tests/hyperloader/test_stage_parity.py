"""Installed public gate for typed-stage and black-box contract parity."""

from __future__ import annotations

import inspect
import os
import unittest
import warnings
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from hyperloader import DataLoader, Pipeline, _hyperloader

from .stage_parity_support import (
    AccessorBlackBox,
    ExplodingBlackBox,
    GlobalBlackBox,
    accessor_pipeline,
    assert_contract_equal,
    exploding_pipeline,
    global_pipeline,
    skip_sample_chain,
)


class StageParityGate(unittest.TestCase):
    """Compare observable contracts through installed public entry points."""

    def test_installed_process_stage_stream_matches_black_box(self) -> None:
        self._assert_installed_root()
        for seed in (17, 29, 43):
            black_box = DataLoader(
                GlobalBlackBox(),
                batch_size=3,
                shuffle=True,
                drop_last=True,
                num_workers=2,
                seed=seed,
            )
            staged = DataLoader(
                global_pipeline(),
                batch_size=3,
                shuffle=True,
                drop_last=True,
                num_workers=2,
                seed=seed,
            )
            try:
                assert_contract_equal(self, list(staged), list(black_box))
            finally:
                staged.close()
                black_box.close()

    def test_declared_thread_stage_stream_matches_process_black_box(self) -> None:
        black_box = DataLoader(AccessorBlackBox(), batch_size=2, num_workers=2, seed=53)
        staged = DataLoader(accessor_pipeline(), batch_size=2, num_workers=2, seed=53)
        mutation = (
            mock.patch.object(Pipeline, "__getitem__", skip_sample_chain)
            if os.environ.get("HYPERLOADER_STAGE_PARITY_MUTATION")
            == "skip-sample-chain"
            else nullcontext()
        )
        try:
            expected = list(black_box)
            with mutation:
                actual = list(staged)
            assert_contract_equal(self, actual, expected)
            self.assertIsNotNone(staged._thread_pool)
            self.assertIsNotNone(black_box._process_pool)
        finally:
            staged.close()
            black_box.close()

    def test_exception_type_message_and_position_match_black_box(self) -> None:
        black_box = DataLoader(
            ExplodingBlackBox(), batch_size=None, num_workers=2, seed=61
        )
        staged = DataLoader(
            exploding_pipeline(), batch_size=None, num_workers=2, seed=61
        )
        try:
            black_values, black_error = self._consume_until_error(black_box)
            stage_values, stage_error = self._consume_until_error(staged)
            self.assertEqual(stage_values, black_values)
            self.assertEqual(stage_values, [0, 1, 2, 3])
            self.assertIs(type(stage_error), type(black_error))
            for error in (stage_error, black_error):
                message = str(error)
                self.assertIn("Caught ValueError in DataLoader worker process", message)
                self.assertIn("Original Traceback", message)
                self.assertIn("stage parity sentinel", message)
        finally:
            staged.close()
            black_box.close()

    def test_abandonment_and_explicit_replay_match_black_box(self) -> None:
        black_box = DataLoader(
            GlobalBlackBox(), batch_size=2, shuffle=True, num_workers=2, seed=67
        )
        staged = DataLoader(
            global_pipeline(), batch_size=2, shuffle=True, num_workers=2, seed=67
        )
        try:
            with warnings.catch_warnings(record=True) as black_warnings:
                black_first = next(iter(black_box))
                black_after_abandon = list(black_box)
            with warnings.catch_warnings(record=True) as stage_warnings:
                stage_first = next(iter(staged))
                stage_after_abandon = list(staged)
            assert_contract_equal(self, stage_first, black_first)
            assert_contract_equal(self, stage_after_abandon, black_after_abandon)
            self.assertEqual(len(stage_warnings), len(black_warnings))

            black_box.set_epoch(0)
            staged.set_epoch(0)
            assert_contract_equal(self, list(staged), list(black_box))
        finally:
            staged.close()
            black_box.close()

    def test_root_seed_is_a_contract_input(self) -> None:
        first = DataLoader(global_pipeline(), batch_size=2, num_workers=2, seed=71)
        changed = DataLoader(global_pipeline(), batch_size=2, num_workers=2, seed=72)
        try:
            with self.assertRaises(AssertionError):
                assert_contract_equal(self, list(first), list(changed))
        finally:
            first.close()
            changed.close()

    def _assert_installed_root(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is None:
            return
        root = Path(expected_root).resolve()
        self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))
        self.assertTrue(Path(inspect.getfile(Pipeline)).resolve().is_relative_to(root))

    @staticmethod
    def _consume_until_error(loader: DataLoader) -> tuple[list[int], BaseException]:
        values = []
        iterator = iter(loader)
        while True:
            try:
                values.append(next(iterator))
            except BaseException as error:
                if isinstance(error, StopIteration):
                    raise AssertionError("the expected dataset exception did not occur")
                return values, error


if __name__ == "__main__":
    unittest.main()
