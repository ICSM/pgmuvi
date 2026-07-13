"""Tests for scoring period-independent wavelength model/kernel-config runs."""

import unittest

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    score_period_independent_wavelength_model_kernel_config_runs,
)


def _run_report():
    return {
        "kind": "period_independent_wavelength_model_kernel_config_results",
        "outcomes": [
            {
                "rank": 1,
                "model": "2DWavelengthDependent",
                "status": "passed",
                "fit_success": True,
                "consensus_success": True,
                "consensus_period": 600.0,
                "consensus_time_kernel_constraint_mode": "period_length",
                "n_accepted_bands": 4,
                "n_rejected_bands": 1,
            },
            {
                "rank": 2,
                "model": "2DDustMean",
                "status": "passed",
                "fit_success": True,
                "consensus_success": True,
                "consensus_period": 600.0,
                "consensus_time_kernel_constraint_mode": "period_length",
                "n_accepted_bands": 5,
                "n_rejected_bands": 0,
            },
            {
                "rank": 3,
                "model": "2D",
                "status": "failed",
                "fit_success": False,
                "consensus_success": None,
                "exception_type": "RuntimeError",
                "exception_message": "synthetic failure",
            },
        ],
    }


class TestPeriodIndependentWavelengthModelKernelConfigScoring(unittest.TestCase):
    def test_score_report_contract_is_advisory_and_nonselecting(self):
        report = score_period_independent_wavelength_model_kernel_config_runs(_run_report())

        self.assertEqual(
            report["kind"], "period_independent_wavelength_model_kernel_config_scores"
        )
        self.assertTrue(report["scores_completed_fits"])
        self.assertEqual(report["score_kind"], "completion_viability")
        self.assertFalse(report["scores_fit_quality"])
        self.assertEqual(report["fit_quality_metrics_used"], [])
        self.assertIn("viability score", report["score_interpretation"])
        self.assertFalse(report["runs_fits"])
        self.assertFalse(report["applies_to_fit"])
        self.assertTrue(report["advisory_only"])
        self.assertFalse(report["automatic_model_selection_applied"])
        self.assertIsNone(report["selected_model"])
        self.assertFalse(report["automatic_constraints_applied"])
        self.assertFalse(report["automatic_initialization_applied"])
        self.assertFalse(report["parameter_suggestions_applied"])

    def test_ranks_successful_candidate_with_more_accepted_bands_first(self):
        report = score_period_independent_wavelength_model_kernel_config_runs(_run_report())
        ranked = report["ranked_results"]

        self.assertEqual(ranked[0]["model"], "2DDustMean")
        self.assertTrue(ranked[0]["is_top_ranked"])
        self.assertEqual(ranked[0]["score_rank"], 1)
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])
        self.assertEqual(report["top_ranked_model"], "2DDustMean")
        self.assertEqual(report["top_ranked_viability_score"], ranked[0]["viability_score"])
        self.assertEqual(ranked[0]["score_kind"], "completion_viability")
        self.assertFalse(ranked[0]["score_is_fit_quality_metric"])
        self.assertEqual(ranked[0]["fit_quality_metrics_used"], [])
        self.assertIsNone(report["selected_model"])

    def test_failures_are_scored_below_passes(self):
        report = score_period_independent_wavelength_model_kernel_config_runs(_run_report())
        ranked = report["ranked_results"]

        self.assertEqual(ranked[-1]["model"], "2D")
        self.assertFalse(ranked[-1]["fit_success"])
        self.assertIn("RuntimeError", ranked[-1]["score_components"])

    def test_model_kernel_config_results_alias_is_accepted(self):
        run_report = _run_report()
        run_report["model_kernel_config_results"] = run_report.pop("outcomes")

        report = score_period_independent_wavelength_model_kernel_config_runs(run_report)

        self.assertEqual(report["n_scored"], 3)
        self.assertEqual(report["ranked_results"], report["scored_model_kernel_configs"])

    def test_unknown_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown scoring weight"):
            score_period_independent_wavelength_model_kernel_config_runs(
                _run_report(), weights={"nonsense": 1.0}
            )

    def test_custom_weights_can_change_ranking(self):
        # Make rejected-band penalty large enough that the rank-1 candidate with
        # one rejected band falls behind the otherwise similar rank-2 candidate.
        report = score_period_independent_wavelength_model_kernel_config_runs(
            _run_report(), weights={"rejected_band": 100.0}
        )

        self.assertEqual(report["ranked_results"][0]["model"], "2DDustMean")
        self.assertLess(
            next(r for r in report["ranked_results"] if r["model"] == "2DWavelengthDependent")["score"],
            report["ranked_results"][0]["score"],
        )

    def test_rejects_non_run_report(self):
        with self.assertRaisesRegex(ValueError, "model/kernel-config results report"):
            score_period_independent_wavelength_model_kernel_config_runs(
                {"kind": "period_independent_wavelength_model_kernel_configs"}
            )

    def test_method_matches_module_function(self):
        lc = Lightcurve([0, 1, 2], [1, 2, 3])
        via_method = lc.score_period_independent_wavelength_model_kernel_config_runs(_run_report())
        via_function = score_period_independent_wavelength_model_kernel_config_runs(_run_report())
        self.assertEqual(via_method, via_function)

    def test_missing_fit_success_falls_back_to_status(self):
        run_report = _run_report()
        del run_report["outcomes"][0]["fit_success"]

        report = score_period_independent_wavelength_model_kernel_config_runs(run_report)
        first_model = next(
            item for item in report["ranked_results"] if item["model"] == "2DWavelengthDependent"
        )
        self.assertTrue(first_model["fit_success"])


if __name__ == "__main__":
    unittest.main()
