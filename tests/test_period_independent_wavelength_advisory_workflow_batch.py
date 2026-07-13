import json
import tempfile
import unittest
from pathlib import Path

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    run_period_independent_wavelength_advisory_workflow_batch,
)


class _FakeLightcurve:
    def __init__(self, model="2DDustMean", score=10.0, fail=False):
        self.model = model
        self.score = score
        self.fail = fail
        self.workflow_calls = []
        self.export_calls = []

    def run_period_independent_wavelength_advisory_workflow(self, **kwargs):
        self.workflow_calls.append(dict(kwargs))
        if self.fail:
            raise RuntimeError("synthetic workflow failure")
        return {
            "kind": "period_independent_wavelength_advisory_workflow",
            "advisory_only": True,
            "runs_fits": True,
            "candidate_fit_state_isolated": True,
            "mutates_input_lightcurve": False,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "automatic_constraints_applied": False,
            "automatic_initialization_applied": False,
            "top_ranked_model": self.model,
            "top_ranked_fit_quality_score": self.score,
            "score_kind": "training_residual_fit_quality",
            "ranked_results": [
                {"model": self.model, "fit_success": True, "fit_quality_score": self.score}
            ],
            "text_report": f"Workflow report for {self.model}",
            "comparison_text_report": f"Comparison report for {self.model}",
            "figures": {},
        }

    def export_period_independent_wavelength_advisory_workflow(
        self, workflow, output_dir, prefix, **kwargs
    ):
        self.export_calls.append(
            {"workflow": workflow, "output_dir": output_dir, "prefix": prefix, **kwargs}
        )
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        json_path = outdir / f"{prefix}.json"
        text_path = outdir / f"{prefix}.txt"
        json_path.write_text(json.dumps({"kind": workflow["kind"]}))
        text_path.write_text(workflow.get("text_report") or "")
        return {
            "kind": "period_independent_wavelength_advisory_workflow_export",
            "json_path": str(json_path),
            "text_report_path": str(text_path),
            "exported_files": [str(json_path), str(text_path)],
        }


class TestPeriodIndependentWavelengthAdvisoryWorkflowBatch(unittest.TestCase):
    def test_runs_multiple_preconstructed_lightcurves(self):
        sources = [
            {"source_id": "src_a", "lightcurve": _FakeLightcurve("2DDustMean", 12.0)},
            {"source_id": "src_b", "lightcurve": _FakeLightcurve("2DWavelengthDependent", 9.0)},
        ]
        report = run_period_independent_wavelength_advisory_workflow_batch(
            sources,
            workflow_kwargs={"include_2d_baseline": True},
            export=False,
        )

        self.assertEqual(report["kind"], "period_independent_wavelength_advisory_workflow_batch")
        self.assertEqual(report["n_sources"], 2)
        self.assertEqual(report["n_succeeded"], 2)
        self.assertEqual(report["n_failed"], 0)
        self.assertEqual(report["source_results"][0]["top_ranked_model"], "2DDustMean")
        self.assertEqual(
            report["source_results"][1]["top_ranked_model"],
            "2DWavelengthDependent",
        )
        self.assertEqual(sources[0]["lightcurve"].workflow_calls[0]["include_2d_baseline"], True)

    def test_report_contract_is_advisory_and_nonselecting(self):
        report = run_period_independent_wavelength_advisory_workflow_batch(
            [{"source_id": "src", "lightcurve": _FakeLightcurve()}],
            export=False,
        )
        self.assertTrue(report["advisory_only"])
        self.assertTrue(report["runs_fits"])
        self.assertTrue(report["applies_to_fit"])
        self.assertTrue(report["candidate_fit_state_isolated"])
        self.assertFalse(report["mutates_input_lightcurve"])
        self.assertFalse(report["automatic_model_selection_applied"])
        self.assertIsNone(report["selected_model"])
        self.assertFalse(report["automatic_constraints_applied"])
        self.assertFalse(report["automatic_initialization_applied"])

    def test_failures_are_recorded_without_stopping(self):
        sources = [
            {"source_id": "good", "lightcurve": _FakeLightcurve("2DDustMean")},
            {"source_id": "bad", "lightcurve": _FakeLightcurve(fail=True)},
        ]
        report = run_period_independent_wavelength_advisory_workflow_batch(
            sources, export=False
        )
        self.assertEqual(report["n_succeeded"], 1)
        self.assertEqual(report["n_failed"], 1)
        failed = report["source_results"][1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["exception_type"], "RuntimeError")
        self.assertIn("synthetic workflow failure", failed["exception_message"])

    def test_stop_on_error_reraises(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic workflow failure"):
            run_period_independent_wavelength_advisory_workflow_batch(
                [{"source_id": "bad", "lightcurve": _FakeLightcurve(fail=True)}],
                stop_on_error=True,
                export=False,
            )

    def test_export_writes_per_source_and_batch_summaries(self):
        sources = [
            {"source_id": "src a", "lightcurve": _FakeLightcurve("2DDustMean")},
            {"source_id": "src/b", "lightcurve": _FakeLightcurve("2DPowerLawMean")},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = run_period_independent_wavelength_advisory_workflow_batch(
                sources,
                output_dir=tmp,
                export=True,
                batch_prefix="batch_test",
            )
            self.assertTrue(Path(report["batch_json_path"]).exists())
            self.assertTrue(Path(report["batch_csv_path"]).exists())
            self.assertGreaterEqual(len(report["exported_files"]), 6)
            for row in report["source_results"]:
                self.assertIsNotNone(row["export_json_path"])
                self.assertTrue(Path(row["export_json_path"]).exists())
                self.assertTrue(Path(row["export_text_report_path"]).exists())

    def test_counts_nested_pr63_workflow_candidate_results(self):
        class NestedWorkflowLightcurve(_FakeLightcurve):
            def run_period_independent_wavelength_advisory_workflow(self, **kwargs):
                return {
                    "kind": "period_independent_wavelength_advisory_workflow",
                    "advisory_only": True,
                    "runs_fits": True,
                    "candidate_fit_state_isolated": True,
                    "mutates_input_lightcurve": False,
                    "automatic_model_selection_applied": False,
                    "selected_model": None,
                    "automatic_constraints_applied": False,
                    "automatic_initialization_applied": False,
                    "top_ranked_model": "2DDustMean",
                    "top_ranked_fit_quality_score": 42.0,
                    "score_kind": "training_residual_fit_quality",
                    "run_report": {
                        "kind": "period_independent_wavelength_fit_candidate_results",
                        "candidate_results": [
                            {"model": "2DDustMean", "fit_success": True},
                            {"model": "2DWavelengthDependent", "fit_success": True},
                            {"model": "2D", "fit_success": False},
                        ],
                    },
                    "quality_report": {
                        "kind": "period_independent_wavelength_fit_candidate_quality_scores",
                        "ranked_results": [
                            {"model": "2DDustMean", "fit_success": True},
                            {"model": "2DWavelengthDependent", "fit_success": True},
                            {"model": "2D", "fit_success": False},
                        ],
                    },
                }

        report = run_period_independent_wavelength_advisory_workflow_batch(
            [{"source_id": "nested", "lightcurve": NestedWorkflowLightcurve()}],
            export=False,
        )
        row = report["source_results"][0]
        self.assertEqual(row["n_model_kernel_configs"], 3)
        self.assertEqual(row["n_successful_model_kernel_configs"], 2)
        self.assertEqual(row["n_failed_model_kernel_configs"], 1)
        # Backward-compatible aliases are retained but should not be preferred in new code.
        self.assertEqual(row["n_candidates"], 3)
        self.assertEqual(row["n_passed_candidates"], 2)

    def test_lightcurve_static_method_delegates(self):
        report = Lightcurve.run_period_independent_wavelength_advisory_workflow_batch(
            [{"source_id": "src", "lightcurve": _FakeLightcurve()}],
            export=False,
        )
        self.assertEqual(report["n_succeeded"], 1)

    def test_rejects_empty_sources(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            run_period_independent_wavelength_advisory_workflow_batch([])

    def test_rejects_invalid_source_dict(self):
        report = run_period_independent_wavelength_advisory_workflow_batch(
            [{"source_id": "bad"}], export=False
        )
        self.assertEqual(report["n_failed"], 1)
        self.assertIn("lightcurve", report["source_results"][0]["exception_message"])


if __name__ == "__main__":
    unittest.main()
