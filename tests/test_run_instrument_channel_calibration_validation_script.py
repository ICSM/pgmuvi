"""CLI coverage for maintained calibration-validation execution."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_instrument_channel_calibration_validation.py"


class TestRunInstrumentChannelCalibrationValidationScript(
    unittest.TestCase
):
    def test_help_exposes_dataset_manifest_surface(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("--dataset-manifest", completed.stdout)
        self.assertIn("--result-output", completed.stdout)
        self.assertIn("--report-output", completed.stdout)


if __name__ == "__main__":
    unittest.main()
