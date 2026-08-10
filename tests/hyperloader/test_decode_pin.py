"""Installed public gate for decoder pin identity and byte stability."""

from __future__ import annotations

import inspect
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch
from torchvision.io import encode_png

from hyperloader import DataLoader, HyperConfig, _hyperloader
from hyperloader.config import DeterminismConfig, SchedulerConfig
from hyperloader.fingerprint import require_fingerprint_match
from hyperloader.fingerprint import builder as fingerprint_builder

from .decode_pin_support import image_pipeline


def _encoded_images() -> list[torch.Tensor]:
    images = []
    for offset in range(4):
        image = (torch.arange(60, dtype=torch.uint8) + offset).reshape(3, 4, 5)
        images.append(encode_png(image))
    return images


def _without_decoder_pins(_loader: object) -> list[object]:
    """Plant a contract fingerprint that omits decoder identity."""
    return []


class DecodePinGate(unittest.TestCase):
    """Prove held pins preserve bytes and changed providers invalidate identity."""

    def test_installed_product_is_under_the_declared_root(self) -> None:
        expected_root = os.environ.get("HYPERLOADER_EXPECTED_INSTALL_ROOT")
        if expected_root is None:
            self.skipTest("the installed-artifact root is declared by the gate harness")
        root = Path(expected_root).resolve()
        self.assertTrue(Path(_hyperloader.__file__).resolve().is_relative_to(root))
        self.assertTrue(
            Path(inspect.getfile(DataLoader)).resolve().is_relative_to(root)
        )

    def test_same_pin_repeats_identical_tensor_bytes(self) -> None:
        first = self._loader()
        second = self._loader()
        try:
            first_batches = list(first)
            second_batches = list(second)
        finally:
            first.close()
            second.close()

        self.assertEqual(first.decoder_pins, second.decoder_pins)
        self.assertEqual(len(first_batches), len(second_batches))
        for left, right in zip(first_batches, second_batches, strict=True):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.stride(), right.stride())
            self.assertTrue(torch.equal(left, right))

    def test_backend_change_invalidates_fingerprint_without_changing_png_bytes(
        self,
    ) -> None:
        mutation = (
            mock.patch.object(
                fingerprint_builder,
                "_decoder_pins",
                _without_decoder_pins,
            )
            if os.environ.get("HYPERLOADER_DECODE_PIN_MUTATION") == "omit-pin"
            else nullcontext()
        )
        with mutation:
            baseline = self._loader()
            changed = self._loader(
                decoder_pins={"pipeline-decode-0": "torchvision.io.decode_image@0.26.0"}
            )
        try:
            baseline_batches = list(baseline)
            changed_batches = list(changed)
            self.assertNotEqual(
                baseline._fingerprint.digest, changed._fingerprint.digest
            )
            with self.assertRaisesRegex(
                ValueError, "fingerprint mismatch at decoder_pins"
            ):
                require_fingerprint_match(baseline._fingerprint, changed._fingerprint)
        finally:
            baseline.close()
            changed.close()

        self.assertEqual(len(baseline_batches), len(changed_batches))
        for left, right in zip(baseline_batches, changed_batches, strict=True):
            self.assertTrue(torch.equal(left, right))

    def test_unavailable_selected_provider_fails_through_public_iteration(self) -> None:
        loader = self._loader(decoder_pins={"png": "missing_decoder.decode@1.0"})
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                r"missing_decoder\.decode@1\.0 is not installed",
            ):
                list(loader)
        finally:
            loader.close()

    @staticmethod
    def _loader(decoder_pins: object = None) -> DataLoader:
        determinism = (
            DeterminismConfig()
            if decoder_pins is None
            else DeterminismConfig(decoder_pins=decoder_pins)
        )
        return DataLoader(
            image_pipeline(_encoded_images()),
            batch_size=2,
            num_workers=1,
            seed=211,
            config=HyperConfig(
                scheduler=SchedulerConfig(profile_cache="off"),
                determinism=determinism,
            ),
        )


if __name__ == "__main__":
    unittest.main()
