"""Provisional dominance protocol and six-workload checks."""

from __future__ import annotations

import importlib
import gc
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from hyperloader import DataLoader
from hyperloader.config import DeterminismConfig, HyperConfig

BENCHES = Path(__file__).parents[3] / "benches"
sys.path.insert(0, str(BENCHES))

benchmark_protocol = importlib.import_module("benchmark_protocol")
dominance_protocol = importlib.import_module("dominance_protocol")
dominance_workloads = importlib.import_module("dominance_workloads")

EnvironmentMetadata = benchmark_protocol.EnvironmentMetadata
TuningBudget = benchmark_protocol.TuningBudget
DominanceObservation = dominance_protocol.DominanceObservation
DominanceProtocolError = dominance_protocol.DominanceProtocolError
DominanceRun = dominance_protocol.DominanceRun
SelectedConfig = dominance_protocol.SelectedConfig
decide = dominance_protocol.decide
make_workload = dominance_workloads.make_workload
workload_names = importlib.import_module("benchmark_protocol.matrix").workload_names


def _environment() -> object:
    return EnvironmentMetadata(
        captured_at="2026-08-10T00:00:00+00:00",
        machine="spark-test",
        operating_system="Linux",
        kernel="test",
        architecture="aarch64",
        python="3.12",
        commit="abc123",
        cpu_governor="performance",
        gpu_clock="locked-3003MHz",
        cache_regime="warm",
        benchmark_mode=True,
        concurrent_load=False,
    )


def _observations(hyperloader: float, reference: float) -> list[object]:
    environment = _environment()
    tuning = TuningBudget(6, 12.0, ("workers", "prefetch_factor"))
    selected = SelectedConfig(4, 2)
    observations = []
    for ordinal in range(5):
        runs = {
            system: DominanceRun(
                system=system,
                reference="torch",
                workload="fixed-text",
                gpu_regime="compute",
                throughput=throughput,
                duration_seconds=45.0,
                warmed=True,
                selected=selected,
                tuning=tuning,
                environment=environment,
            )
            for system, throughput in (
                ("hyperloader", hyperloader),
                ("torch", reference),
            )
        }
        order = (
            ("hyperloader", "torch") if ordinal % 2 == 0 else ("torch", "hyperloader")
        )
        observations.append(
            DominanceObservation(
                ordinal,
                runs[order[0]],
                runs[order[1]],
                uninterrupted=True,
            )
        )
    return observations


class DominanceHarnessTest(unittest.TestCase):
    """Verify workload equality, decision bands, and protocol rejection."""

    def test_public_import_comes_from_the_expected_install(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is None:
            self.skipTest("an installed package root was not supplied")
        import hyperloader

        package_path = Path(hyperloader.__file__).resolve()
        self.assertTrue(package_path.is_relative_to(Path(expected_root).resolve()))

    def test_six_workloads_match_reference_values_through_public_loader(self) -> None:
        for name in workload_names():
            if name == "numpy-array":
                continue
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    self._assert_public_values(name, Path(directory))
        with tempfile.TemporaryDirectory() as directory:
            self._assert_public_values("numpy-array", Path(directory))
            gc.collect()

    def test_varlen_rows_share_one_fixture_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workload = make_workload("varlen-text", Path(directory), batches=1)
            try:
                storages = {
                    row.untyped_storage().data_ptr()
                    for row in workload.reference_dataset
                }
                self.assertEqual(len(storages), 1)
                self.assertEqual(len(workload.reference_dataset), 64)
            finally:
                workload.close()

    def _assert_public_values(self, name: str, root: Path) -> None:
        workload = make_workload(
            name,
            root,
            batches=1,
            torchvision_version=os.environ.get(
                "HYPERLOADER_DOMINANCE_TORCHVISION_VERSION"
            ),
        )
        loader = DataLoader(
            workload.hyperloader_dataset,
            batch_size=workload.batch_size,
            num_workers=1,
            seed=313,
            config=HyperConfig(
                determinism=(
                    DeterminismConfig(decoder_pins=workload.decoder_pins)
                    if workload.decoder_pins is not None
                    else DeterminismConfig()
                )
            ),
        )
        try:
            if workload.decoder_pins is not None:
                self.assertEqual(
                    loader.decoder_pins[0]["version"],
                    next(iter(workload.decoder_pins.values())).rsplit("@", 1)[1],
                )
            expected_items = [
                workload.reference_dataset[index]
                for index in range(workload.batch_size)
            ]
            expected = workload.normalize(workload.collate_fn(expected_items))
            actual = workload.normalize(next(iter(loader)))
            import torch

            self.assertTrue(torch.equal(actual, expected))
        finally:
            loader.close()
            workload.close()

    def test_bootstrap_decision_distinguishes_win_tie_and_loss(self) -> None:
        self.assertEqual(decide(_observations(102.0, 100.0)).status, "win")
        self.assertEqual(decide(_observations(99.5, 100.0)).status, "tie")
        self.assertEqual(decide(_observations(98.0, 100.0)).status, "loss")

    def test_unequal_tuning_budget_is_rejected(self) -> None:
        observations = _observations(102.0, 100.0)
        first = observations[0]
        changed = replace(
            first.second,
            tuning=TuningBudget(1, 2.0, ("workers",)),
        )
        observations[0] = DominanceObservation(0, first.first, changed, True)
        with self.assertRaisesRegex(DominanceProtocolError, "same counted tuning"):
            decide(observations)

    def test_non_alternating_pair_order_is_rejected(self) -> None:
        observations = _observations(102.0, 100.0)
        second = observations[1]
        observations[1] = DominanceObservation(
            1,
            second.second,
            second.first,
            True,
        )
        with self.assertRaisesRegex(DominanceProtocolError, "alternate"):
            decide(observations)

    def test_selected_configuration_change_is_rejected(self) -> None:
        observations = _observations(102.0, 100.0)
        third = observations[2]
        changed = replace(third.first, selected=SelectedConfig(8, 4))
        observations[2] = DominanceObservation(2, changed, third.second, True)
        with self.assertRaisesRegex(DominanceProtocolError, "configuration changed"):
            decide(observations)

    def test_unknown_workload_and_system_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown dominance workload"):
                make_workload("missing", Path(directory))
        feeders = importlib.import_module("dominance_feeders")
        with self.assertRaisesRegex(ValueError, "unknown dominance system"):
            feeders.build_feeder("missing", object(), SelectedConfig(2, 2))

    def test_loader_slowdown_mutation_exceeds_the_tie_margin(self) -> None:
        mutation = os.environ.get("HYPERLOADER_DOMINANCE_MUTATION")
        hyperloader = 98.0 if mutation == "slow-loader" else 102.0
        self.assertEqual(decide(_observations(hyperloader, 100.0)).status, "win")


if __name__ == "__main__":
    unittest.main()
