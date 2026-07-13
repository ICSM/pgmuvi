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
            "model_kernel_config_state_isolated": True,
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
        self.assertTrue(report["model_kernel_config_state_isolated"])
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

    def test_counts_nested_pr63_workflow_model_kernel_config_results(self):
        class NestedWorkflowLightcurve(_FakeLightcurve):
            def run_period_independent_wavelength_advisory_workflow(self, **kwargs):
                return {
                    "kind": "period_independent_wavelength_advisory_workflow",
                    "advisory_only": True,
                    "runs_fits": True,
                    "model_kernel_config_state_isolated": True,
                    "mutates_input_lightcurve": False,
                    "automatic_model_selection_applied": False,
                    "selected_model": None,
                    "automatic_constraints_applied": False,
                    "automatic_initialization_applied": False,
                    "top_ranked_model": "2DDustMean",
                    "top_ranked_fit_quality_score": 42.0,
                    "score_kind": "training_residual_fit_quality",
                    "run_report": {
                        "kind": "period_independent_wavelength_model_kernel_config_results",
                        "model_kernel_config_results": [
                            {"model": "2DDustMean", "fit_success": True},
                            {"model": "2DWavelengthDependent", "fit_success": True},
                            {"model": "2D", "fit_success": False},
                        ],
                    },
                    "quality_report": {
                        "kind": "period_independent_wavelength_model_kernel_config_quality_scores",
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


    def test_writes_long_form_model_kernel_config_csv(self):
        nested_workflow = {
            "kind": "period_independent_wavelength_advisory_workflow",
            "top_ranked_model": "2DDustMean",
            "top_ranked_fit_quality_score": 42.0,
            "score_kind": "training_residual_fit_quality",
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "run_report": {
                "kind": "period_independent_wavelength_model_kernel_config_results",
                "model_kernel_config_results": [
                    {
                        "model_kernel_config_id": "rank1_2DWavelengthDependent",
                        "rank": 1,
                        "model": "2DWavelengthDependent",
                        "status": "passed",
                        "fit_success": True,
                        "fit_kwargs": {
                            "fit_strategy": "consensus",
                            "time_kernel_type": "quasi_periodic",
                            "training_iter": 7,
                            "miniter": 3,
                            "learn_additional_noise": True,
                        },
                        "consensus_period": 603.0,
                        "training_nrmse_by_target_scale": 1.3,
                    },
                    {
                        "model_kernel_config_id": "rank2_2DDustMean",
                        "rank": 2,
                        "model": "2DDustMean",
                        "status": "passed",
                        "fit_success": True,
                        "fit_kwargs": {
                            "fit_strategy": "consensus",
                            "time_kernel_type": "quasi_periodic",
                            "training_iter": 7,
                            "miniter": 3,
                            "learn_additional_noise": True,
                        },
                        "consensus_period": 603.0,
                        "training_nrmse_by_target_scale": 1.2,
                    },
                ],
            },
            "quality_report": {
                "kind": "period_independent_wavelength_model_kernel_config_quality_scores",
                "score_kind": "training_residual_fit_quality",
                # Real workflow quality rows may omit model_kernel_config_id and
                # model-kernel-config rank.  They must merge by model name into
                # the execution rows rather than becoming duplicate CSV rows.
                "ranked_results": [
                    {
                        "quality_rank": 1,
                        "is_top_ranked": True,
                        "model": "2DDustMean",
                        "fit_success": True,
                        "fit_quality_score": 42.0,
                        "training_reduced_chi2": 0.5,
                    },
                    {
                        "quality_rank": 2,
                        "is_top_ranked": False,
                        "model": "2DWavelengthDependent",
                        "fit_success": True,
                        "fit_quality_score": 41.0,
                        "training_reduced_chi2": 0.6,
                    },
                ],
            },
        }

        class NestedFake(_FakeLightcurve):
            def run_period_independent_wavelength_advisory_workflow(self, **kwargs):
                return nested_workflow

        with tempfile.TemporaryDirectory() as tmp:
            report = run_period_independent_wavelength_advisory_workflow_batch(
                [{"source_id": "src", "lightcurve": NestedFake()}],
                output_dir=tmp,
                export=False,
                batch_prefix="batch",
            )
            detail_path = Path(report["batch_model_kernel_config_csv_path"])
            self.assertTrue(detail_path.exists())

            import csv

            with detail_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(len(report["model_kernel_config_results"]), 2)
        self.assertEqual(len(csv_rows), 2)

        by_model = {
            row["model"]: row for row in report["model_kernel_config_results"]
        }
        dust = by_model["2DDustMean"]
        self.assertEqual(dust["model_kernel_config_id"], "rank2_2DDustMean")
        self.assertEqual(dust["model_kernel_config_rank"], 2)
        self.assertEqual(dust["quality_rank"], 1)
        self.assertEqual(dust["time_kernel_type"], "quasi_periodic")
        self.assertEqual(dust["fit_quality_score"], 42.0)

        csv_by_model = {row["model"]: row for row in csv_rows}
        self.assertEqual(csv_by_model["2DDustMean"]["model_kernel_config_id"], "rank2_2DDustMean")
        self.assertEqual(csv_by_model["2DDustMean"]["quality_rank"], "1")
        self.assertEqual(csv_by_model["2DWavelengthDependent"]["quality_rank"], "2")

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


    def test_writes_aggregate_model_kernel_config_summary_csv(self):
        workflow_template = {
            "kind": "period_independent_wavelength_advisory_workflow",
            "advisory_only": True,
            "runs_fits": True,
            "model_kernel_config_state_isolated": True,
            "mutates_input_lightcurve": False,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "top_ranked_model": "2DDustMean",
            "top_ranked_fit_quality_score": 42.0,
            "score_kind": "training_residual_fit_quality",
            "run_report": {
                "kind": "period_independent_wavelength_model_kernel_config_results",
                "model_kernel_config_results": [
                    {
                        "model_kernel_config_id": "rank1_2DWavelengthDependent",
                        "rank": 1,
                        "model": "2DWavelengthDependent",
                        "status": "passed",
                        "fit_success": True,
                        "fit_kwargs": {
                            "fit_strategy": "consensus",
                            "time_kernel_type": "quasi_periodic",
                            "learn_additional_noise": True,
                        },
                        "training_nrmse_by_target_scale": 2.0,
                    },
                    {
                        "model_kernel_config_id": "rank2_2DDustMean",
                        "rank": 2,
                        "model": "2DDustMean",
                        "status": "passed",
                        "fit_success": True,
                        "fit_kwargs": {
                            "fit_strategy": "consensus",
                            "time_kernel_type": "quasi_periodic",
                            "learn_additional_noise": True,
                        },
                        "training_nrmse_by_target_scale": 1.0,
                    },
                ],
            },
            "quality_report": {
                "kind": "period_independent_wavelength_model_kernel_config_quality_scores",
                "score_kind": "training_residual_fit_quality",
                "ranked_results": [
                    {
                        "quality_rank": 1,
                        "is_top_ranked": True,
                        "model": "2DDustMean",
                        "fit_success": True,
                        "fit_quality_score": 42.0,
                    },
                    {
                        "quality_rank": 2,
                        "is_top_ranked": False,
                        "model": "2DWavelengthDependent",
                        "fit_success": True,
                        "fit_quality_score": 40.0,
                    },
                ],
            },
        }

        class NestedFake(_FakeLightcurve):
            def run_period_independent_wavelength_advisory_workflow(self, **kwargs):
                return workflow_template

        with tempfile.TemporaryDirectory() as tmp:
            report = run_period_independent_wavelength_advisory_workflow_batch(
                [
                    {"source_id": "src-a", "lightcurve": NestedFake()},
                    {"source_id": "src-b", "lightcurve": NestedFake()},
                ],
                output_dir=tmp,
                export=False,
                batch_prefix="batch",
            )
            summary_path = Path(report["batch_model_kernel_config_summary_csv_path"])
            self.assertTrue(summary_path.exists())
            text = summary_path.read_text(encoding="utf-8")

        summary = report["model_kernel_config_summary"]
        self.assertEqual(len(summary), 2)
        by_model = {row["model"]: row for row in summary}
        dust = by_model["2DDustMean"]
        self.assertEqual(dust["n_sources_evaluated"], 2)
        self.assertEqual(dust["n_successful_sources"], 2)
        self.assertEqual(dust["n_top_ranked_sources"], 2)
        self.assertEqual(dust["top_ranked_fraction"], 1.0)
        self.assertEqual(dust["median_fit_quality_score"], 42.0)
        self.assertIn("n_top_ranked_sources", text)
        self.assertIn("2DDustMean", text)



if __name__ == "__main__":
    unittest.main()
