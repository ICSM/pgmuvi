"""No-training tests for the toy batch advisory walkthrough script."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "wavelength_advisory_batch_walkthrough.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "wavelength_advisory_batch_walkthrough", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestWavelengthAdvisoryBatchWalkthroughScript(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_prepare_only_writes_two_source_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "walkthrough"
            rc = self.module.main(
                [
                    "--workspace",
                    str(workspace),
                    "--n-points-per-band",
                    "8",
                ]
            )

            self.assertEqual(rc, 0)
            source_list = workspace / "sources.txt"
            manifest_path = workspace / "walkthrough_manifest.json"
            self.assertTrue(source_list.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertTrue((workspace / "inputs/toy_lpv_a.csv").is_file())
            self.assertTrue((workspace / "inputs/toy_lpv_b.csv").is_file())
            self.assertFalse((workspace / "outputs").exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "wavelength_advisory_batch_walkthrough")
            self.assertEqual(manifest["n_sources"], 2)
            self.assertEqual(manifest["bands"], ["J", "H", "K"])
            self.assertIn("do not validate scientific ranking", manifest["scientific_use_warning"])

    def test_generated_csvs_are_positive_and_have_three_bands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "walkthrough"
            self.module.main(
                [
                    "--workspace",
                    str(workspace),
                    "--n-points-per-band",
                    "8",
                ]
            )

            with (workspace / "inputs/toy_lpv_a.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 24)
        self.assertEqual(set(rows[0]), {"time", "wavelength", "flux", "flux_error", "band"})
        self.assertEqual({row["band"] for row in rows}, {"J", "H", "K"})
        self.assertEqual(
            {float(row["wavelength"]) for row in rows},
            {1.25, 1.65, 2.20},
        )
        self.assertTrue(all(float(row["flux"]) > 0.0 for row in rows))
        self.assertTrue(all(float(row["flux_error"]) > 0.0 for row in rows))

    def test_manifest_records_smoke_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "walkthrough"
            self.module.main(
                [
                    "--workspace",
                    str(workspace),
                    "--n-points-per-band",
                    "8",
                    "--training-iter",
                    "7",
                    "--miniter",
                    "3",
                    "--model-kernel-config-limit",
                    "2",
                    "--allow-source-failures",
                ]
            )
            manifest = json.loads(
                (workspace / "walkthrough_manifest.json").read_text(encoding="utf-8")
            )

        command = manifest["batch_command"]
        self.assertIn("run_wavelength_advisory_batch.py", " ".join(command))
        self.assertIn("--source-list", command)
        self.assertIn("--no-check-sampling", command)
        self.assertIn("--no-plots", command)
        self.assertIn("--allow-source-failures", command)
        self.assertEqual(command[command.index("--training-iter") + 1], "7")
        self.assertEqual(command[command.index("--miniter") + 1], "3")
        self.assertEqual(command[command.index("--model-kernel-config-limit") + 1], "2")

    def test_run_batch_delegates_and_returns_child_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "walkthrough"
            with patch.object(self.module, "_run_command", return_value=4) as mocked:
                rc = self.module.main(
                    [
                        "--workspace",
                        str(workspace),
                        "--n-points-per-band",
                        "8",
                        "--run-batch",
                    ]
                )

        self.assertEqual(rc, 4)
        mocked.assert_called_once()
        command = mocked.call_args.args[0]
        self.assertIn("run_wavelength_advisory_batch.py", " ".join(command))

    def test_existing_workspace_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "walkthrough"
            workspace.mkdir()
            with self.assertRaises(SystemExit) as caught:
                self.module.main(["--workspace", str(workspace)])
        self.assertEqual(caught.exception.code, 2)

    def test_argument_validation_rejects_miniter_above_training_iter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "walkthrough"
            with self.assertRaises(SystemExit) as caught:
                self.module.main(
                    [
                        "--workspace",
                        str(workspace),
                        "--training-iter",
                        "2",
                        "--miniter",
                        "3",
                    ]
                )
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
