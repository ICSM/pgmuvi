"""Contracts for the D3 representative-LPV command-line runner."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pgmuvi.wavelength_validation_real_lpv import (
    RepresentativeLPVBatchReport,
    RepresentativeLPVSourceSpecification,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT / "scripts" / "run_representative_lpv_validation.py"
)


def _load_script():
    specification = importlib.util.spec_from_file_location(
        "run_representative_lpv_validation",
        SCRIPT_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load D3 runner script.")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


SCRIPT = _load_script()


class TestRepresentativeLPVManifestLoader(unittest.TestCase):
    def test_example_manifest_resolves_public_source_path(self):
        manifest_path = (
            ROOT
            / "examples"
            / "validation"
            / "d3_representative_lpv_manifest.json"
        )
        specifications = (
            SCRIPT.load_representative_lpv_manifest(
                manifest_path
            )
        )

        self.assertEqual(len(specifications), 1)
        self.assertEqual(
            specifications[0].source_id,
            "10131+3049",
        )
        self.assertEqual(
            Path(specifications[0].source_path),
            (
                ROOT
                / "examples"
                / "data"
                / "10131+3049.csv"
            ).resolve(),
        )
        self.assertEqual(
            specifications[0].metadata["manifest_source_path"],
            "../data/10131+3049.csv",
        )

    def test_object_manifest_requires_sources_key(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = (
                Path(temporary_directory) / "manifest.json"
            )
            manifest_path.write_text(
                json.dumps({"not_sources": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "sources"):
                SCRIPT.load_representative_lpv_manifest(
                    manifest_path
                )

    def test_workflow_configuration_must_be_mapping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            configuration_path = (
                Path(temporary_directory) / "workflow.json"
            )
            configuration_path.write_text(
                json.dumps(["not", "a", "mapping"]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "object"):
                SCRIPT.load_representative_lpv_workflow_configuration(
                    configuration_path
                )


class TestRepresentativeLPVCLI(unittest.TestCase):
    def setUp(self):
        self.manifest_path = (
            ROOT
            / "examples"
            / "validation"
            / "d3_representative_lpv_manifest.json"
        )
        self.workflow_path = (
            ROOT
            / "examples"
            / "validation"
            / "d3_representative_lpv_workflow.json"
        )

    def test_validate_only_runs_no_fits_and_writes_no_outputs(self):
        batch_calls = []
        export_calls = []
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            status = SCRIPT.main(
                [
                    "--manifest",
                    str(self.manifest_path),
                    "--validate-manifest-only",
                ],
                batch_runner=lambda *args, **kwargs: (
                    batch_calls.append((args, kwargs))
                ),
                exporter=lambda *args, **kwargs: (
                    export_calls.append((args, kwargs))
                ),
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(batch_calls, [])
        self.assertEqual(export_calls, [])
        self.assertTrue(payload["valid"])
        self.assertFalse(payload["runs_fits"])
        self.assertFalse(payload["writes_outputs"])
        self.assertEqual(
            payload["source_ids"],
            ["10131+3049"],
        )

    def test_run_passes_manifest_configuration_and_exports(self):
        captured = {}
        stdout = io.StringIO()

        def fake_batch_runner(
            manifest,
            *,
            workflow_kwargs,
            output_root,
            batch_id,
        ):
            captured["manifest"] = manifest
            captured["workflow_kwargs"] = workflow_kwargs
            captured["output_root"] = output_root
            captured["batch_id"] = batch_id

            source = manifest[0]
            return RepresentativeLPVBatchReport(
                batch_id=batch_id,
                source_manifest=manifest,
                source_results=(
                    {
                        "source_id": source.source_id,
                        "source_specification": source.to_dict(),
                        "source_output_dir": (
                            f"{output_root}/sources/10131_3049"
                        ),
                        "source_report_path": (
                            f"{output_root}/sources/"
                            "10131_3049/report.json"
                        ),
                        "status": "completed",
                        "execution_seed": source.seed,
                        "seed_applied": True,
                        "seed_scope": (
                            "source_loading_and_advisory_workflow"
                        ),
                        "validation_report": {
                            "report_id": "d3:10131+3049",
                            "advisory_only": True,
                            "automatic_model_selection_applied": (
                                False
                            ),
                            "selected_model": None,
                        },
                        "failure": None,
                    },
                ),
                output_root=output_root,
                workflow_configuration=workflow_kwargs,
                runtime_environment={
                    "python_version": "test",
                },
            )

        def fake_exporter(report, output_root):
            captured["export_report"] = report
            captured["export_output_root"] = output_root
            return {
                "kind": (
                    "representative_lpv_batch_report_export"
                ),
                "output_dir": output_root,
            }

        with redirect_stdout(stdout):
            status = SCRIPT.main(
                [
                    "--manifest",
                    str(self.manifest_path),
                    "--workflow-config",
                    str(self.workflow_path),
                    "--output-root",
                    "validation_outputs/test_d3",
                    "--batch-id",
                    "test-d3-batch",
                ],
                batch_runner=fake_batch_runner,
                exporter=fake_exporter,
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(
            captured["manifest"][0].source_id,
            "10131+3049",
        )
        self.assertEqual(
            captured["workflow_kwargs"][
                "base_fit_kwargs"
            ]["fit_strategy"],
            "consensus",
        )
        self.assertEqual(
            captured["workflow_kwargs"][
                "base_fit_kwargs"
            ]["time_kernel_type"],
            "quasi_periodic",
        )
        self.assertTrue(
            captured["workflow_kwargs"][
                "base_fit_kwargs"
            ]["learn_additional_noise"]
        )
        self.assertEqual(
            captured["output_root"],
            "validation_outputs/test_d3",
        )
        self.assertEqual(
            captured["batch_id"],
            "test-d3-batch",
        )
        self.assertTrue(payload["exported"])
        self.assertEqual(payload["n_failed_sources"], 0)
        self.assertFalse(
            payload["automatic_model_selection_applied"]
        )
        self.assertIsNone(payload["selected_model"])

    def test_fail_on_source_failure_returns_two_after_reporting(self):
        source = RepresentativeLPVSourceSpecification(
            source_id="failed",
            source_path="inputs/failed.csv",
            description="Synthetic failed source.",
            sample_role="failure_boundary",
            selection_reason="Exercise CLI exit status.",
            seed=3,
        )

        def fake_batch_runner(*args, **kwargs):
            del args
            output_root = kwargs["output_root"]
            return RepresentativeLPVBatchReport(
                batch_id=kwargs["batch_id"],
                source_manifest=(source,),
                source_results=(
                    {
                        "source_id": source.source_id,
                        "source_specification": source.to_dict(),
                        "source_output_dir": (
                            f"{output_root}/sources/failed"
                        ),
                        "source_report_path": (
                            f"{output_root}/sources/"
                            "failed/report.json"
                        ),
                        "status": "failed",
                        "execution_seed": 3,
                        "seed_applied": True,
                        "seed_scope": (
                            "source_loading_and_advisory_workflow"
                        ),
                        "validation_report": None,
                        "failure": {
                            "stage": "source_loading",
                            "failure_code": (
                                "source_loading_failed"
                            ),
                            "exception_type": (
                                "FileNotFoundError"
                            ),
                            "exception_message": "Missing.",
                        },
                    },
                ),
                output_root=output_root,
            )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = SCRIPT.main(
                [
                    "--manifest",
                    str(self.manifest_path),
                    "--no-export",
                    "--fail-on-source-failure",
                ],
                batch_runner=fake_batch_runner,
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(status, 2)
        self.assertFalse(payload["exported"])
        self.assertEqual(payload["n_failed_sources"], 1)


if __name__ == "__main__":
    unittest.main()
