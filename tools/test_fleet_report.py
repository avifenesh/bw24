#!/usr/bin/env python3
"""CPU-only tests for fleet snapshot accumulation and daily reporting."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
FIXTURE = ROOT / "research/cx-fleet-20260808/restart-fixture.jsonl"


def load_report():
    sys.path.insert(0, str(TOOLS))
    path = TOOLS / "fleet-report.py"
    spec = importlib.util.spec_from_file_location("fleet_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FleetReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = load_report()

    def test_restart_fixture_daily_deltas_and_economics(self) -> None:
        snapshots = self.report.load_snapshots(FIXTURE)
        self.assertTrue(self.report.counters_regressed(snapshots[2], snapshots[3]))

        days = self.report.daily_deltas(snapshots)
        self.assertEqual([day["date"] for day in days], [
            "2026-08-05",
            "2026-08-06",
            "2026-08-07",
        ])
        self.assertEqual(
            (
                days[1]["prompt_tokens_in"],
                days[1]["cached_tokens_in"],
                days[1]["computed_tokens_in"],
                days[1]["restarts"],
            ),
            (1200, 550, 650, 1),
        )

        low = self.report.economics(days[1], 0.25)
        high = self.report.economics(days[1], 1.0)
        self.assertEqual(low["revenue_multiplier"], 1.2115)
        self.assertEqual(high["revenue_multiplier"], 1.8462)
        self.assertEqual(high["lcp_window_64_512_share"], 0.5)

        rendered = self.report.render_report(snapshots, days)
        self.assertIn("45.8%", rendered)
        self.assertIn("+5.83pp", rendered)
        self.assertIn("1.2115x..1.8462x", rendered)

    def test_meter_marks_restart_and_skips_unchanged_snapshot(self) -> None:
        fixture_rows = [
            json.loads(line)
            for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        ]
        before = {
            key: value
            for key, value in fixture_rows[2].items()
            if key not in {"ts", "restart"}
        }
        after = {
            key: value
            for key, value in fixture_rows[3].items()
            if key not in {"ts", "restart"}
        }

        with tempfile.TemporaryDirectory(prefix="memra-fleet-meter-test-") as temp:
            temp_path = Path(temp)
            metrics = temp_path / "metrics.json"
            ledger = temp_path / "fleet.jsonl"
            env = os.environ.copy()
            env.update({
                "FLEET_METRICS_URL": metrics.as_uri(),
                "FLEET_LEDGER": str(ledger),
                "FLEET_TIMEOUT_SECONDS": "2",
            })

            metrics.write_text(json.dumps(before), encoding="utf-8")
            subprocess.run(
                [str(TOOLS / "fleet-meter.sh"), "--once"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            unchanged = subprocess.run(
                [str(TOOLS / "fleet-meter.sh"), "--once"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn("unchanged", unchanged.stdout)

            metrics.write_text(json.dumps(after), encoding="utf-8")
            subprocess.run(
                [str(TOOLS / "fleet-meter.sh"), "--once"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            rows = [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertFalse(rows[0]["restart"])
            self.assertTrue(rows[1]["restart"])

    def test_meter_skips_failed_scrape_without_creating_ledger(self) -> None:
        with tempfile.TemporaryDirectory(prefix="memra-fleet-down-test-") as temp:
            ledger = Path(temp) / "fleet.jsonl"
            env = os.environ.copy()
            env.update({
                "FLEET_METRICS_URL": (Path(temp) / "missing.json").as_uri(),
                "FLEET_LEDGER": str(ledger),
                "FLEET_TIMEOUT_SECONDS": "1",
            })
            result = subprocess.run(
                [str(TOOLS / "fleet-meter.sh"), "--once"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn("skip: scrape failed", result.stderr)
            self.assertFalse(ledger.exists())


if __name__ == "__main__":
    unittest.main()
