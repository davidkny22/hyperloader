"""Evidence-bounded bottleneck attribution."""

from __future__ import annotations

import unittest

from hyperloader.diagnose.attribution import attribute_cause


def _observation() -> dict[str, object]:
    return {
        "blocking": {"fraction": None, "currently_blocked": False},
        "ceiling_binds": [],
        "saturation": {"occupancy_fraction": None},
    }


class AttributionTest(unittest.TestCase):
    """Prefer direct causes and preserve inconclusive reports."""

    def test_controller_binding_has_priority(self) -> None:
        observed = _observation()
        observed["ceiling_binds"] = ["bandwidth"]
        observed["blocking"] = {"fraction": 0.25}

        attributed = attribute_cause(observed)

        self.assertEqual(attributed["cause"], "user_ceiling")
        self.assertIn("bandwidth", attributed["basis"])

    def test_measured_or_current_wait_names_delivery(self) -> None:
        for blocking in (
            {"fraction": 0.125},
            {"fraction": None, "currently_blocked": True},
        ):
            with self.subTest(blocking=blocking):
                observed = _observation()
                observed["blocking"] = blocking
                self.assertEqual(
                    attribute_cause(observed)["cause"], "delivery_wait"
                )

    def test_low_saturation_and_inconclusive_paths_stay_distinct(self) -> None:
        observed = _observation()
        observed["saturation"] = {"occupancy_fraction": 0.25}
        self.assertEqual(
            attribute_cause(observed)["cause"], "low_ready_saturation"
        )

        self.assertEqual(
            attribute_cause(_observation())["cause"], "not_identified"
        )


if __name__ == "__main__":
    unittest.main()
