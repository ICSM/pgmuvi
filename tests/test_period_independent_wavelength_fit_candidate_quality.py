"""Tests for training-residual wavelength fit-quality diagnostics."""

import unittest

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    score_period_independent_wavelength_fit_candidate_quality,
)


def _quality_run_report():
    return {
        "kind": "period_independent_wavelength_fit_candidate_results",
        "candidate_results": [
            {
                "rank": 1,
                "model": "2DWavelengthDependent",
                "status": "passed",
                "fit_success": True,
                "consensus_success": True,
                "consensus_period": 600.0,
                "consensus_time_kernel_constraint_mode": "period_length",
                "fit_quality_available": True,
                "fit_quality": {
                    "available": True,
                    "rmse": 0.30,
                    "mae": 0.20,
                    "normalized_rmse_by_target_scale": 0.30,
                    "normalized_rmse": 1.0,
                    "median_abs_standardized_residual": 0.70,
                    "outlier_fraction_3sigma": 0.02,
                    "reduced_chi2": 1.2,
                    "log_marginal_likelihood": -10.0,
                },
            },
            {
                "rank": 2,
                "model": "2DDustMean",
                "status": "passed",
                "fit_success": True,
                "consensus_success": True,
                "consensus_period": 600.0,
                "consensus_time_kernel_constraint_mode": "period_length",
                "fit_quality_available": True,
                "fit_quality": {
                    "available": True,
                    "rmse": 0.10,
                    "mae": 0.08,
                    "normalized_rmse_by_target_scale": 0.10,
                    "normalized_rmse": 0.8,
                    "median_abs_standardized_residual": 0.50,
                    "outlier_fraction_3sigma": 0.0,
                    "reduced_chi2": 0.9,
                    "log_marginal_likelihood": -5.0,
                },
            },
            {
                "rank": 3,
                "model": "2D",
                "status": "failed",
                "fit_success": False,
                "consensus_success": None,
                "fit_quality": {"available": False, "reason": "failed"},
            },
        ],
    }


class TestPeriodIndependentWavelengthFitCandidateQuality(unittest.TestCase):
    def test_quality_score_report_is_advisory_and_nonselecting(self):
        report = score_period_independent_wavelength_fit_candidate_quality(
            _quality_run_report()
        )

        self.assertEqual(
            report["kind"], "period_independent_wavelength_fit_candidate_quality_scores"
        )
        self.assertEqual(report["score_kind"], "training_residual_fit_quality")
        self.assertTrue(report["scores_fit_quality"])
        self.assertFalse(report["runs_fits"])
        self.assertFalse(report["applies_to_fit"])
        self.assertTrue(report["advisory_only"])
        self.assertFalse(report["automatic_model_selection_applied"])
        self.assertIsNone(report["selected_model"])
        self.assertFalse(report["automatic_constraints_applied"])
        self.assertFalse(report["automatic_initialization_applied"])

    def test_lower_residual_metrics_rank_higher_than_advisory_order(self):
        report = score_period_independent_wavelength_fit_candidate_quality(
            _quality_run_report()
        )
        ranked = report["ranked_results"]

        self.assertEqual(ranked[0]["model"], "2DDustMean")
        self.assertEqual(ranked[0]["quality_rank"], 1)
        self.assertTrue(ranked[0]["is_top_ranked"])
        self.assertGreater(
            ranked[0]["fit_quality_score"], ranked[1]["fit_quality_score"]
        )
        self.assertEqual(report["top_ranked_model"], "2DDustMean")
        self.assertIsNone(report["selected_model"])

    def test_failed_or_unavailable_quality_scores_last(self):
        report = score_period_independent_wavelength_fit_candidate_quality(
            _quality_run_report()
        )
        ranked = report["ranked_results"]

        self.assertEqual(ranked[-1]["model"], "2D")
        self.assertFalse(ranked[-1]["fit_quality_available"])
        self.assertLess(ranked[-1]["fit_quality_score"], -1e8)

    def test_report_aliases_are_present(self):
        report = score_period_independent_wavelength_fit_candidate_quality(
            _quality_run_report()
        )
        self.assertEqual(report["ranked_results"], report["quality_ranked_results"])
        self.assertEqual(report["n_with_fit_quality"], 2)
        self.assertIn("training_nrmse_by_target_scale", report["fit_quality_metrics_used"])

    def test_rejects_non_run_report(self):
        with self.assertRaisesRegex(ValueError, "fit-candidate results report"):
            score_period_independent_wavelength_fit_candidate_quality(
                {"kind": "period_independent_wavelength_fit_candidate_scores"}
            )

    def test_method_matches_module_function(self):
        lc = Lightcurve([0, 1, 2], [1, 2, 3])
        via_method = lc.score_period_independent_wavelength_fit_candidate_quality(
            _quality_run_report()
        )
        via_function = score_period_independent_wavelength_fit_candidate_quality(
            _quality_run_report()
        )
        self.assertEqual(via_method, via_function)


if __name__ == "__main__":
    unittest.main()
