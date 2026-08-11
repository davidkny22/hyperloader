"""CPU-idle, interrupt, and wait-attribution helper tests."""

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

BENCHES = Path(__file__).parents[3] / "benches"
sys.path.insert(0, str(BENCHES))
cpu_idle = importlib.import_module("dominance_cpu_idle")
wait_workload = importlib.import_module("dominance_wait_workload")
alu_spinner = importlib.import_module("dominance_alu_spinner")
wake_latency = importlib.import_module("dominance_wake_latency")


class _SequencedEvent:
    def __init__(self, values: list[bool]) -> None:
        self._values = iter(values)

    def query(self) -> bool:
        return next(self._values)


class DominanceWakeLatencyTest(unittest.TestCase):
    def test_only_auto_controls_consumes_the_live_loader(self) -> None:
        self.assertTrue(wake_latency.uses_live_hyperloader("auto-controls"))
        for mode in ("blocking", "event-query", "consumer-warmth"):
            with self.subTest(mode=mode):
                self.assertFalse(wake_latency.uses_live_hyperloader(mode))

    def test_cpuidle_snapshot_and_diff_preserve_every_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cpu in (0, 19):
                for state in (0, 1):
                    path = root / f"cpu{cpu}" / "cpuidle" / f"state{state}"
                    path.mkdir(parents=True)
                    values = {
                        "name": f"LPI-{state}",
                        "desc": f"state {state}",
                        "latency": str(state * 42),
                        "residency": str(state * 1930),
                        "power": "0",
                        "disable": "0",
                        "time": str(100 + cpu + state),
                        "usage": str(10 + state),
                    }
                    for name, value in values.items():
                        (path / name).write_text(value + "\n", encoding="utf-8")
            before = cpu_idle.snapshot_cpuidle(root)
            state = root / "cpu19" / "cpuidle" / "state1"
            (state / "time").write_text("120120\n", encoding="utf-8")
            (state / "usage").write_text("31\n", encoding="utf-8")
            after = cpu_idle.snapshot_cpuidle(root)

        report = cpu_idle.diff_cpuidle(before, after, 1.0)
        row = next(
            item for item in report["rows"] if item["cpu"] == 19 and item["state"] == 1
        )
        self.assertEqual(len(report["rows"]), 4)
        self.assertEqual(row["time_delta_us"], 120000)
        self.assertEqual(row["usage_delta"], 20)
        self.assertEqual(row["window_residency_percent"], 12.0)
        self.assertEqual(row["mean_requested_residency_us"], 6000.0)

    def test_interrupt_parser_and_diff_report_irq_and_cpu_deltas(self) -> None:
        before = cpu_idle.parse_gpu_interrupts(
            "           CPU0 CPU1\n"
            "484: 10 2 GICv3 141 Level nvidia\n"
            "485: 1 0 GICv3 142 Level nvidia-modeset\n"
            "IPI0: 7 8 Rescheduling interrupts\n"
        )
        after = cpu_idle.parse_gpu_interrupts(
            "           CPU0 CPU1\n"
            "484: 13 7 GICv3 141 Level nvidia\n"
            "485: 1 4 GICv3 142 Level nvidia-modeset\n"
        )

        report = cpu_idle.diff_gpu_interrupts(before, after)

        self.assertEqual(report["aggregate_deltas"], {"CPU0": 3, "CPU1": 9})
        self.assertEqual(report["rows"][0]["deltas"], {"CPU0": 3, "CPU1": 5})

    def test_bounded_event_query_returns_exact_attempt_count(self) -> None:
        clock = iter((0.0, 0.1, 0.2)).__next__

        attempts = wait_workload.bounded_event_query(
            _SequencedEvent([False, False, True]), 1.0, clock=clock
        )

        self.assertEqual(attempts, 3)

    def test_bounded_event_query_times_out(self) -> None:
        clock = iter((0.0, 0.1, 0.6)).__next__

        with self.assertRaisesRegex(TimeoutError, "0.500 seconds"):
            wait_workload.bounded_event_query(
                _SequencedEvent([False, False, False]), 0.5, clock=clock
            )

    def test_duty_cycle_rejects_an_invalid_period_before_loading_native_code(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "no longer than the period"):
            alu_spinner.DutyCycleAluSpinner(
                Path("missing"),
                19,
                active_microseconds=1_001,
                period_microseconds=1_000,
            )


if __name__ == "__main__":
    unittest.main()
