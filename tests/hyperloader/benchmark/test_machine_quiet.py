"""Selection checks for Spark machine quieting."""

from __future__ import annotations

import unittest

from benches.spark_machine_quiet import ProcessRecord, gui_processes


class SparkMachineQuietTest(unittest.TestCase):
    """Select only owner-authorized GUI processes from the process table."""

    def test_gui_selection_excludes_ssh_and_the_quieting_harness(self) -> None:
        records = [
            ProcessRecord(1, "root", "sshd", "/usr/sbin/sshd -D"),
            ProcessRecord(2, "david", "gnome-shell", "/usr/bin/gnome-shell"),
            ProcessRecord(
                3,
                "david",
                "python3",
                "python3 benches/spark_machine_quiet.py --evidence run.json",
            ),
            ProcessRecord(
                4,
                "gnome-remote",
                "gnome-remote-de",
                "/usr/libexec/gnome-remote-desktop-daemon --system",
            ),
        ]

        self.assertEqual([record.pid for record in gui_processes(records)], [2, 4])


if __name__ == "__main__":
    unittest.main()
