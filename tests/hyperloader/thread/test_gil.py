"""Free-threaded GIL restoration instrument tests."""

from __future__ import annotations

import unittest
from unittest import mock

from hyperloader.thread.gil import GilRestorationDetector


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
        with mock.patch(
            "hyperloader.thread.gil.gil_enabled", side_effect=[False, True, True]
        ):
            detector = GilRestorationDetector(recorder)
            detector.observe()
            detector.observe()
        self.assertEqual(recorder.events, 1)

    def test_already_enabled_runtime_is_not_a_restoration(self) -> None:
        recorder = Recorder()
        with mock.patch("hyperloader.thread.gil.gil_enabled", return_value=True):
            detector = GilRestorationDetector(recorder)
            detector.observe()
        self.assertEqual(recorder.events, 0)


if __name__ == "__main__":
    unittest.main()
