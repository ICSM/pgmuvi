"""Export contracts for representative observed-LPV D3 reports."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from pgmuvi.wavelength_validation_real_lpv import (
    RepresentativeLPVBatchReport,
    RepresentativeLPVSourceSpecification,
    export_representative_lpv_batch_report,
)


class TestRepresentativeLPVBatchExport(unittest.TestCase):
    def setUp(self):
        self.manifest = (
            RepresentativeLPVSourceSpecification(
                source_id="good/source",
                source_path="inputs/good.csv",
                description="Completed representative LPV.",
                sample_role="validation",
                selection_reason="Exercises completed export.",
                seed=21,
            ),
            RepresentativeLPVSourceSpecification(
                source_id="failed-source",
                source_path="inputs/failed.csv",
                description="Failed representative LPV.",
                sample_role="failure_boundary",
                selection_reason="Exercises failure export.",
                seed=22,
            ),
        )
        self.report = RepresentativeLPVBatchReport(
            batch_id="d3-export-test",
            source_manifest=self.manifest,
            source_results=(
                {
                    "source_id": "good/source",
                    "source_specification": self.manifest[0].to_dict(),
                    "source_output_dir": (
                        "validation_outputs/d3_real_lpv/"
                        "sources/good_source"
                    ),
                    "source_report_path": (
                        "validation_outputs/d3_real_lpv/"
                        "sources/good_source/report.json"
                    ),
                    "status": "completed",
                    "validation_report": {
                        "report_id": "d3:good/source",
                        "advisory_only": True,
                        "automatic_model_selection_applied": False,
                        "selected_model": None,
                    },
                    "failure": None,
                },
                {
                    "source_id": "failed-source",
                    "source_specification": self.manifest[1].to_dict(),
                    "source_output_dir": (
                        "validation_outputs/d3_real_lpv/"
                        "sources/failed-source"
                    ),
                    "source_report_path": (
                        "validation_outputs/d3_real_lpv/"
                        "sources/failed-source/report.json"
                    ),
                    "status": "failed",
                    "validation_report": None,
                    "failure": {
                        "stage": "advisory_workflow",
                        "failure_code": "advisory_workflow_failed",
                        "exception_type": "RuntimeError",
                        "exception_message": "Synthetic failure.",
                    },
                },
            ),
        )

    def test_export_writes_deterministic_artifact_tree(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = export_representative_lpv_batch_report(
                self.report,
                temporary_directory,
            )
            root = Path(temporary_directory)

            expected = (
                root / "batch_report.json",
                root / "source_manifest.json",
                root / "source_summary.csv",
                root / "export_manifest.json",
                root / "sources" / "good_source" / "source_result.json",
                root / "sources" / "good_source" / "report.json",
                (
                    root
                    / "sources"
                    / "failed-source"
                    / "source_result.json"
                ),
                root / "sources" / "failed-source" / "failure.json",
            )
            for path in expected:
                with self.subTest(path=path):
                    self.assertTrue(path.is_file())

            self.assertEqual(manifest["n_sources"], 2)
            self.assertEqual(manifest["n_completed_sources"], 1)
            self.assertEqual(manifest["n_failed_sources"], 1)

    def test_exported_json_is_strict_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = export_representative_lpv_batch_report(
                self.report.to_dict(),
                temporary_directory,
            )

            for path in (
                manifest["batch_report_path"],
                manifest["source_manifest_path"],
                manifest["export_manifest_path"],
                *(
                    item["source_result_path"]
                    for item in manifest["source_artifacts"]
                ),
                *(
                    item["artifact_path"]
                    for item in manifest["source_artifacts"]
                ),
            ):
                with self.subTest(path=path):
                    payload = json.loads(
                        Path(path).read_text(encoding="utf-8")
                    )
                    json.dumps(payload, allow_nan=False)

    def test_summary_csv_records_completed_and_failed_sources(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = export_representative_lpv_batch_report(
                self.report,
                temporary_directory,
            )

            with Path(manifest["source_summary_path"]).open(
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                [row["source_id"] for row in rows],
                ["good/source", "failed-source"],
            )
            self.assertEqual(
                [row["status"] for row in rows],
                ["completed", "failed"],
            )
            self.assertEqual(
                rows[1]["failure_stage"],
                "advisory_workflow",
            )
            self.assertEqual(
                rows[1]["exception_type"],
                "RuntimeError",
            )

    def test_export_layer_remains_nonselecting_and_does_not_run_fits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = export_representative_lpv_batch_report(
                self.report,
                temporary_directory,
            )

            self.assertTrue(manifest["advisory_only"])
            self.assertFalse(
                manifest["automatic_model_selection_applied"]
            )
            self.assertIsNone(manifest["selected_model"])
            self.assertFalse(
                manifest["automatic_constraints_applied"]
            )
            self.assertFalse(
                manifest["automatic_initialization_applied"]
            )
            self.assertTrue(manifest["source_report_writes_only"])
            self.assertFalse(manifest["runs_fits"])

    def test_invalid_report_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(TypeError, "report"):
                export_representative_lpv_batch_report(
                    object(),
                    temporary_directory,
                )


if __name__ == "__main__":
    unittest.main()
