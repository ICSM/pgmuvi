"""No-training tests for examples/interpret_pgmuvi_outputs.py."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "interpret_pgmuvi_outputs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("interpret_pgmuvi_outputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestInterpretPgmuviOutputsScript(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()

    def test_period_summary_flags_primary_largest_area_difference(self):
        payload = {
            "method": "summed_psd_peaks",
            "model_name": "2D",
            "kernel_family": "spectral_mixture",
            "time_kernel_family": "spectral_mixture",
            "dominant_period": 100.0,
            "dominant_frequency": 0.01,
            "period_interval": [95.0, 106.0],
            "interval_definition": "peak_centered_68pct_mass_interval",
            "q_factor": 12.0,
            "primary_peak_rank": 1,
            "largest_area_peak_rank": 2,
            "largest_area_period": 200.0,
            "n_significant_peaks": 2,
            "peaks": [
                {"rank": 1, "period": 100.0},
                {"rank": 2, "period": 200.0},
            ],
            "component_diagnostics": {"component_periods": [99.0, 202.0]},
        }
        summary = self.module.interpret_payload(payload)
        self.assertEqual(summary["report_type"], "period_summary")
        self.assertEqual(summary["dominant_period"], 100.0)
        self.assertTrue(
            any("differs from the largest-area" in item for item in summary["warnings"])
        )
        self.assertTrue(
            any("not independent final periods" in item for item in summary["warnings"])
        )

    def test_advisory_summary_surfaces_training_score_ard_and_nonselection(self):
        payload = {
            "kind": "period_independent_wavelength_advisory_workflow",
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "fit_quality_ranking_status": "available",
            "fit_quality_ranking_available": True,
            "only_valid_model": None,
            "top_ranked_model": "2DDustMean",
            "top_ranked_fit_quality_score": 82.5,
            "score_kind": "training_residual_fit_quality",
            "quality_report": {
                "ranked_results": [
                    {
                        "model": "2DDustMean",
                        "training_nrmse_by_target_scale": 0.2,
                        "training_median_abs_standardized_residual": 0.7,
                        "training_outlier_fraction_3sigma": 0.01,
                        "training_reduced_chi2": 1.1,
                    }
                ]
            },
            "run_report": {
                "model_kernel_config_results": [
                    {
                        "model": "2D",
                        "model_kernel_config_id": "rank4_2D_baseline",
                        "status": "passed",
                        "n_constrained_sm_ard_components": 1,
                        "constrained_sm_ard_dimension_counts": {
                            "wavelength_frequency": 1
                        },
                    }
                ]
            },
            "fallback_report": {"available": False},
        }
        summary = self.module.interpret_payload(payload)
        self.assertEqual(summary["fit_quality_ranking_status"], "available")
        self.assertTrue(summary["fit_quality_ranking_available"])
        self.assertEqual(summary["top_ranked_model"], "2DDustMean")
        self.assertIsNone(summary["only_valid_model"])
        self.assertIsNone(summary["selected_model"])
        self.assertEqual(len(summary["constrained_sm_ard_rows"]), 1)
        text = self.module.format_interpretation(summary)
        self.assertIn("training_residual_fit_quality", text)
        self.assertIn("constrained_sm_ard_rows: 1", text)


    def test_advisory_summary_clears_stale_top_when_ranking_unavailable(self):
        payload = {
            "kind": "period_independent_wavelength_advisory_workflow",
            "fit_quality_ranking_status": "unavailable",
            "fit_quality_ranking_available": True,
            "top_ranked_model": "stale-model",
            "top_ranked_fit_quality_score": -1.0e9,
            "quality_report": {
                "ranking_status": "unavailable",
                "fit_quality_ranking_available": True,
                "top_ranked_model": "stale-model",
                "ranked_results": [],
            },
            "fallback_report": {
                "available": True,
                "reason": "no_model_kernel_config_fit_quality_available",
            },
        }

        summary = self.module.interpret_payload(payload)

        self.assertEqual(summary["fit_quality_ranking_status"], "unavailable")
        self.assertFalse(summary["fit_quality_ranking_available"])
        self.assertIsNone(summary["top_ranked_model"])
        self.assertIsNone(summary["top_ranked_fit_quality_score"])

    def test_advisory_summary_reports_single_valid_candidate_separately(self):
        payload = {
            "kind": "period_independent_wavelength_advisory_workflow",
            "fit_quality_ranking_status": "single_valid_candidate",
            "fit_quality_ranking_available": False,
            "only_valid_model": "2DDustMean",
            "top_ranked_model": None,
            "quality_report": {
                "ranking_status": "single_valid_candidate",
                "fit_quality_ranking_available": False,
                "only_valid_model": "2DDustMean",
                "ranked_results": [],
            },
            "fallback_report": {
                "available": True,
                "reason": "only_one_model_kernel_config_has_fit_quality",
            },
        }

        summary = self.module.interpret_payload(payload)

        self.assertEqual(
            summary["fit_quality_ranking_status"], "single_valid_candidate"
        )
        self.assertEqual(summary["only_valid_model"], "2DDustMean")
        self.assertIsNone(summary["top_ranked_model"])

    def test_advisory_fallback_surfaces_failure_counts(self):
        payload = {
            "kind": "period_independent_wavelength_advisory_workflow",
            "run_report": {
                "outcomes": [
                    {
                        "model": "2D",
                        "status": "failed",
                        "fit_failed": True,
                        "failure_stage": "consensus",
                    }
                ]
            },
            "quality_report": {"ranked_results": []},
            "fallback_report": {
                "available": True,
                "reason": "all_model_kernel_config_fits_failed",
                "failure_stage_counts": {"consensus": 1},
                "exception_type_counts": {"ConsensusFitError": 1},
            },
        }
        summary = self.module.interpret_payload(payload)
        self.assertTrue(summary["fallback_available"])
        self.assertEqual(summary["failure_stage_counts"], {"consensus": 1})
        self.assertTrue(
            any(
                "Comparative fit-quality ranking is unavailable" in item
                for item in summary["warnings"]
            )
        )

    def test_batch_summary_counts_failures_and_ard_hits(self):
        payload = {
            "kind": "period_independent_wavelength_advisory_workflow_batch",
            "n_sources": 2,
            "n_succeeded": 1,
            "n_failed": 1,
            "selected_model": None,
            "source_results": [
                {"source_id": "a", "status": "passed"},
                {"source_id": "b", "status": "failed"},
            ],
            "model_kernel_config_results": [
                {
                    "source_id": "a",
                    "model": "2D",
                    "status": "passed",
                    "n_constrained_sm_ard_components": 2,
                    "constrained_sm_ard_dimension_counts": {
                        "time_frequency": 1,
                        "wavelength_frequency": 1,
                    },
                },
                {"source_id": "b", "model": "2DDustMean", "status": "failed"},
            ],
        }
        summary = self.module.interpret_payload(payload)
        self.assertEqual(summary["n_source_failure_rows"], 1)
        self.assertEqual(summary["n_model_kernel_config_failures"], 1)
        self.assertEqual(len(summary["ard_ceiling_hits"]), 1)

    def test_cli_writes_normalized_json(self):
        payload = {
            "method": "explicit_period_parameter",
            "model_name": "1DQuasiPeriodic",
            "dominant_period": 120.0,
            "dominant_frequency": 1.0 / 120.0,
            "primary_peak_rank": 1,
            "largest_area_peak_rank": 1,
            "peaks": [{"rank": 1, "period": 120.0}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "period.json"
            output_path = Path(tmpdir) / "interpretation.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            rc = self.module.main(
                [str(input_path), "--json-output", str(output_path)]
            )
            written = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(rc, 0)
        self.assertEqual(written["report_type"], "period_summary")
        self.assertEqual(written["dominant_period"], 120.0)

    def test_unsupported_payload_returns_status_two(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "unknown.json"
            input_path.write_text(json.dumps({"kind": "unknown"}), encoding="utf-8")
            rc = self.module.main([str(input_path)])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
