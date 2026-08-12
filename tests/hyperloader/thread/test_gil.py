"""Free-threaded GIL restoration instrument tests."""

from __future__ import annotations

import unittest
from unittest import mock

from hyperloader.thread.gil import GilRestorationDetector, free_threaded_build


class Recorder:
    """Count restoration reports from the detector."""

    def __init__(self) -> None:
        self.events = 0

    def record_gil_restore(self) -> None:
        self.events += 1


class GilRestorationDetectorTest(unittest.TestCase):
    """Exercise the process-wide false-to-true transition rule."""

    def test_free_threaded_transition_reports_once(self) -> None:
        recorder = Recorder()
        with (
            mock.patch("hyperloader.thread.gil.free_threaded_build", return_value=True),
            mock.patch("hyperloader.thread.gil.gil_enabled", side_effect=[True, True]),
        ):
            detector = GilRestorationDetector(recorder)
            detector.observe()
            detector.observe()
        self.assertEqual(recorder.events, 1)

    def test_already_enabled_runtime_is_not_a_restoration(self) -> None:
        recorder = Recorder()
        with (
            mock.patch(
                "hyperloader.thread.gil.free_threaded_build", return_value=False
            ),
            mock.patch("hyperloader.thread.gil.gil_enabled", return_value=True),
        ):
            detector = GilRestorationDetector(recorder)
            detector.observe()
        self.assertEqual(recorder.events, 0)

    def test_build_identity_survives_runtime_gil_restoration(self) -> None:
        with mock.patch(
            "hyperloader.thread.gil.sysconfig.get_config_var", return_value=1
        ) as query:
            self.assertTrue(free_threaded_build())
        query.assert_called_once_with("Py_GIL_DISABLED")


if __name__ == "__main__":
    unittest.main()
