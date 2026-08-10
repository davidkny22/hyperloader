"""Static platform decoder selection and disclosure tests."""

from __future__ import annotations

import unittest

import torch
from torchvision.io import decode_png, encode_png

from hyperloader import AUTO, Collate, DataLoader, Decode, HyperConfig, Source, pipeline
from hyperloader.config import DeterminismConfig, SchedulerConfig
from hyperloader.decoder import bind_decoder_selections, select_decoder_pins
from hyperloader.decoder.pins import platform_pin


def decode_bytes(value: bytes) -> int:
    """Provide a spawn-safe user decoder identity."""
    return int(value)


def collect(values: list[int]) -> list[int]:
    """Provide one stable collate stage."""
    return values


def forbidden_user_decoder(_value: torch.Tensor) -> torch.Tensor:
    """Fail if an opted-in stage bypasses its selected provider."""
    raise AssertionError("the user decoder must not run after substitution")


def _pipeline(*, codec: str | None = None, substitute: bool = False):
    return pipeline(
        Source([b"1", b"2"], output_type=bytes),
        Decode(
            decode_bytes,
            input_type=bytes,
            output_type=int,
            codec=codec,
            substitute=substitute,
        ),
        Collate(collect, input_type=int, output_type=list),
    )


class DecoderSelectionTest(unittest.TestCase):
    """Prove table selection, opt-in refuge, overrides, and disclosure."""

    def test_release_table_is_complete_for_supported_platforms(self) -> None:
        for platform in ("win32", "linux", "darwin"):
            with self.subTest(platform=platform):
                self.assertEqual(
                    platform_pin(platform, "jpeg"),
                    ("torchvision.io.decode_jpeg", "0.26.0"),
                )
                self.assertEqual(
                    platform_pin(platform, "png"),
                    ("torchvision.io.decode_png", "0.26.0"),
                )
                self.assertEqual(
                    platform_pin(platform, "audio"),
                    ("torchcodec.decoders.AudioDecoder", "0.13.0"),
                )

    def test_user_callable_is_the_non_substituting_refuge(self) -> None:
        loader = DataLoader(
            _pipeline(),
            num_workers=0,
            config=HyperConfig(scheduler=SchedulerConfig(profile_cache="off")),
        )
        try:
            pin = loader.decoder_pins[0]
            values = {
                element.path: element.value for element in loader._fingerprint.elements
            }
            self.assertFalse(pin["substituted"])
            self.assertEqual(pin["source"], "user-callable")
            self.assertIn("decode_bytes", pin["backend"])
            self.assertEqual(values["decoder_pins"], list(loader.decoder_pins))
        finally:
            loader.close()

    def test_substitution_uses_the_static_platform_pin(self) -> None:
        selections = select_decoder_pins(
            _pipeline(codec="jpeg", substitute=True),
            configured=AUTO,
            platform="win32",
        )
        self.assertEqual(len(selections), 1)
        self.assertTrue(selections[0].substituted)
        self.assertEqual(selections[0].source, "platform-table")
        self.assertEqual(selections[0].backend, "torchvision.io.decode_jpeg")
        self.assertEqual(selections[0].version, "0.26.0")

    def test_explicit_override_is_disclosed_and_changes_identity(self) -> None:
        off = SchedulerConfig(profile_cache="off")
        base = DataLoader(
            _pipeline(codec="png", substitute=True),
            num_workers=0,
            config=HyperConfig(scheduler=off),
        )
        changed = DataLoader(
            _pipeline(codec="png", substitute=True),
            num_workers=0,
            config=HyperConfig(
                scheduler=off,
                determinism=DeterminismConfig(
                    decoder_pins={"pipeline-decode-0": "example.decode_png@9.4"}
                ),
            ),
        )
        try:
            self.assertNotEqual(base._fingerprint.digest, changed._fingerprint.digest)
            self.assertEqual(changed.decoder_pins[0]["backend"], "example.decode_png")
            self.assertEqual(changed.decoder_pins[0]["version"], "9.4")
            self.assertEqual(changed.decoder_pins[0]["source"], "configured")
        finally:
            base.close()
            changed.close()

    def test_bound_provider_executes_the_selected_png_decoder(self) -> None:
        image = torch.arange(18, dtype=torch.uint8).reshape(3, 2, 3)
        encoded = encode_png(image)
        declared = pipeline(
            Source([encoded], output_type=torch.Tensor),
            Decode(
                decode_bytes,
                input_type=torch.Tensor,
                output_type=torch.Tensor,
                codec="png",
                substitute=True,
            ),
            Collate(collect, input_type=torch.Tensor, output_type=list),
        )
        selections = select_decoder_pins(declared, AUTO, platform="win32")

        bound = bind_decoder_selections(declared, selections)

        self.assertTrue(torch.equal(bound[0], decode_png(encoded)))

    def test_selected_provider_version_is_enforced_at_first_use(self) -> None:
        declared = _pipeline(codec="png", substitute=True)
        selections = select_decoder_pins(
            declared,
            {"png": "torchvision.io.decode_png@0.0.0"},
            platform="win32",
        )
        bound = bind_decoder_selections(declared, selections)

        with self.assertRaisesRegex(RuntimeError, "requires 0.0.0"):
            bound[0]

    def test_public_process_path_executes_the_selected_provider(self) -> None:
        image = torch.arange(18, dtype=torch.uint8).reshape(3, 2, 3)
        encoded = encode_png(image)
        declared = pipeline(
            Source([encoded], output_type=torch.Tensor),
            Decode(
                forbidden_user_decoder,
                input_type=torch.Tensor,
                output_type=torch.Tensor,
                codec="png",
                substitute=True,
            ),
            Collate(collect, input_type=torch.Tensor, output_type=list),
        )
        loader = DataLoader(
            declared,
            batch_size=1,
            num_workers=1,
            config=HyperConfig(scheduler=SchedulerConfig(profile_cache="off")),
        )
        try:
            batches = list(loader)
        finally:
            loader.close()

        self.assertEqual(len(batches), 1)
        self.assertTrue(torch.equal(batches[0][0], decode_png(encoded)))

    def test_invalid_declarations_and_unmatched_overrides_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit codec"):
            _pipeline(substitute=True)
        with self.assertRaisesRegex(ValueError, "codec must"):
            _pipeline(codec="webp")
        with self.assertRaisesRegex(TypeError, "boolean"):
            _pipeline(codec="jpeg", substitute=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "did not match"):
            select_decoder_pins(
                _pipeline(codec="jpeg", substitute=True),
                {"png": "example.decode@1"},
                platform="linux",
            )


if __name__ == "__main__":
    unittest.main()
