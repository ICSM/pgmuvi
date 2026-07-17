"""Tests for training-residual wavelength fit-quality diagnostics."""

import unittest

import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    _piwd_compute_training_fit_quality,
    score_period_independent_wavelength_model_kernel_config_quality,
)


def _quality_run_report():
    return {
        "kind": "period_independent_wavelength_model_kernel_config_results",
        "model_kernel_config_results": [
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


class _Prediction:
    def __init__(self, mean, variance=None):
        self.mean = torch.as_tensor(mean, dtype=torch.float64)
        if variance is not None:
            self.variance = torch.as_tensor(variance, dtype=torch.float64)


class _Model:
    def __call__(self, _x):
        return object()


class _Likelihood:
    def __init__(self, prediction):
        self.prediction = prediction

    def __call__(self, _output):
        return self.prediction


class _FittedLightcurve:
    def __init__(self, *, prediction, y, yerr):
        self.model = _Model()
        self.likelihood = _Likelihood(prediction)
        self._xdata_transformed = torch.arange(len(y), dtype=torch.float64)
        self._ydata_transformed = torch.as_tensor(y, dtype=torch.float64)
        self._yerr_transformed = torch.as_tensor(yerr, dtype=torch.float64)

    def _eval(self):
        return None


class TestTrainingFitQualityVarianceSemantics(unittest.TestCase):
    def test_observed_predictive_variance_is_not_combined_with_yerr_again(self):
        fitted = _FittedLightcurve(
            prediction=_Prediction(mean=[0.0, 0.0], variance=[4.0, 4.0]),
            y=[1.0, 2.0],
            yerr=[3.0, 3.0],
        )

        report = _piwd_compute_training_fit_quality(fitted)

        self.assertTrue(report["available"])
        self.assertEqual(report["predictive_variance_kind"], "observed")
        self.assertEqual(
            report["standardization_sigma_source"],
            "observed_predictive_standard_deviation",
        )
        self.assertFalse(report["measurement_uncertainty_added_separately"])
        self.assertAlmostEqual(report["normalized_rmse"], (0.625) ** 0.5)
        self.assertAlmostEqual(report["reduced_chi2"], 1.25)

    def test_yerr_is_used_only_when_predictive_variance_is_unavailable(self):
        fitted = _FittedLightcurve(
            prediction=_Prediction(mean=[0.0, 0.0]),
            y=[1.0, 2.0],
            yerr=[2.0, 2.0],
        )

        report = _piwd_compute_training_fit_quality(fitted)

        self.assertEqual(
            report["standardization_sigma_source"],
            "transformed_measurement_uncertainty_fallback",
        )
        self.assertIsNone(report["predictive_variance_kind"])
        self.assertAlmostEqual(report["normalized_rmse"], (0.625) ** 0.5)
        self.assertAlmostEqual(report["reduced_chi2"], 1.25)

    def test_invalid_predictive_variance_uses_pointwise_yerr_fallback(self):
        fitted = _FittedLightcurve(
            prediction=_Prediction(mean=[0.0, 0.0], variance=[4.0, float("nan")]),
            y=[1.0, 2.0],
            yerr=[3.0, 2.0],
        )

        report = _piwd_compute_training_fit_quality(fitted)

        self.assertEqual(
            report["standardization_sigma_source"],
            "observed_predictive_standard_deviation_with_transformed_yerr_fallback",
        )
        self.assertAlmostEqual(report["normalized_rmse"], (0.625) ** 0.5)
        self.assertAlmostEqual(report["reduced_chi2"], 1.25)


class TestPeriodIndependentWavelengthModelKernelConfigQuality(unittest.TestCase):
    def test_quality_score_report_is_advisory_and_nonselecting(self):
        report = score_period_independent_wavelength_model_kernel_config_quality(
            _quality_run_report()
        )

        self.assertEqual(
            report["kind"], "period_independent_wavelength_model_kernel_config_quality_scores"
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
        report = score_period_independent_wavelength_model_kernel_config_quality(
            _quality_run_report()
        )
        ranked = report["ranked_results"]

        self.assertEqual(ranked[0]["model"], "2DDustMean")
        self.assertEqual(ranked[0]["quality_rank"], 1)
        self.assertTrue(ranked[0]["is_top_ranked"])
        self.assertGreater(
            ranked[0]["fit_quality_score"], ranked[1]["fit_quality_score"]
        )
        self.assertEqual(report["ranking_status"], "available")
        self.assertTrue(report["fit_quality_ranking_available"])
        self.assertEqual(report["top_ranked_model"], "2DDustMean")
        self.assertIsNone(report["only_valid_model"])
        self.assertIsNone(report["selected_model"])

    def test_failed_or_unavailable_quality_scores_last(self):
        report = score_period_independent_wavelength_model_kernel_config_quality(
            _quality_run_report()
        )
        ranked = report["ranked_results"]

        self.assertEqual(ranked[-1]["model"], "2D")
        self.assertFalse(ranked[-1]["fit_quality_available"])
        self.assertIsNone(ranked[-1]["fit_quality_score"])
        self.assertIsNone(ranked[-1]["quality_rank"])
        self.assertFalse(ranked[-1]["is_top_ranked"])


    def test_all_failed_candidates_do_not_produce_a_top_rank(self):
        report = score_period_independent_wavelength_model_kernel_config_quality(
            {
                "kind": "period_independent_wavelength_model_kernel_config_results",
                "outcomes": [
                    {
                        "rank": 1,
                        "model": "2DDustMean",
                        "status": "failed",
                        "fit_success": False,
                        "fit_quality": {"available": False, "reason": "failed"},
                    },
                    {
                        "rank": 2,
                        "model": "2D",
                        "status": "failed",
                        "fit_success": False,
                        "fit_quality": {"available": False, "reason": "failed"},
                    },
                ],
            }
        )

        self.assertEqual(report["ranking_status"], "unavailable")
        self.assertFalse(report["fit_quality_ranking_available"])
        self.assertIsNone(report["top_ranked_model"])
        self.assertIsNone(report["top_ranked_fit_quality_score"])
        self.assertIsNone(report["only_valid_model"])
        self.assertEqual(report["n_scored"], 0)
        self.assertEqual(report["n_unscored"], 2)
        self.assertTrue(
            all(not row["is_top_ranked"] for row in report["ranked_results"])
        )
        self.assertTrue(
            all(row["fit_quality_score"] is None for row in report["ranked_results"])
        )

    def test_completed_but_unavailable_diagnostics_do_not_produce_a_top_rank(self):
        report = score_period_independent_wavelength_model_kernel_config_quality(
            {
                "kind": "period_independent_wavelength_model_kernel_config_results",
                "outcomes": [
                    {
                        "rank": 1,
                        "model": "2DDustMean",
                        "status": "passed",
                        "fit_success": True,
                        "fit_quality": {
                            "available": False,
                            "reason": "diagnostics unavailable",
                        },
                    },
                    {
                        "rank": 2,
                        "model": "2D",
                        "status": "passed",
                        "fit_success": True,
                        "fit_quality": {
                            "available": False,
                            "reason": "diagnostics unavailable",
                        },
                    },
                ],
            }
        )

        self.assertEqual(report["ranking_status"], "unavailable")
        self.assertIsNone(report["top_ranked_model"])
        self.assertEqual(report["n_with_fit_quality"], 0)

    def test_single_valid_candidate_is_not_a_comparative_top_rank(self):
        run_report = _quality_run_report()
        run_report["model_kernel_config_results"] = [
            run_report["model_kernel_config_results"][1],
            run_report["model_kernel_config_results"][2],
        ]

        report = score_period_independent_wavelength_model_kernel_config_quality(
            run_report
        )

        self.assertEqual(report["ranking_status"], "single_valid_candidate")
        self.assertFalse(report["fit_quality_ranking_available"])
        self.assertTrue(report["single_valid_candidate"])
        self.assertIsNone(report["top_ranked_model"])
        self.assertEqual(report["only_valid_model"], "2DDustMean")
        self.assertIsNotNone(report["only_valid_fit_quality_score"])
        valid = report["ranked_results"][0]
        self.assertEqual(valid["quality_rank"], 1)
        self.assertFalse(valid["is_top_ranked"])

    def test_comparison_ineligible_attempt_remains_unscored(self):
        run_report = _quality_run_report()
        run_report["model_kernel_config_results"][0][
            "comparison_eligibility"
        ] = "ineligible"
        run_report["model_kernel_config_results"][0][
            "technical_outcome"
        ] = "initialized_only"

        report = score_period_independent_wavelength_model_kernel_config_quality(
            run_report
        )
        row = next(
            item
            for item in report["ranked_results"]
            if item["model"] == "2DWavelengthDependent"
        )

        self.assertFalse(row["fit_quality_available"])
        self.assertIsNone(row["fit_quality_score"])
        self.assertEqual(row["comparison_eligibility"], "ineligible")
        self.assertEqual(row["technical_outcome"], "initialized_only")
        self.assertIn("comparison-ineligible", row["score_components"][0])

    def test_report_aliases_are_present(self):
        report = score_period_independent_wavelength_model_kernel_config_quality(
            _quality_run_report()
        )
        self.assertEqual(report["ranked_results"], report["quality_ranked_results"])
        self.assertEqual(report["n_with_fit_quality"], 2)
        self.assertIn("training_nrmse_by_target_scale", report["fit_quality_metrics_used"])

    def test_rejects_non_run_report(self):
        with self.assertRaisesRegex(ValueError, "model/kernel-config results report"):
            score_period_independent_wavelength_model_kernel_config_quality(
                {"kind": "period_independent_wavelength_model_kernel_config_scores"}
            )

    def test_method_matches_module_function(self):
        lc = Lightcurve([0, 1, 2], [1, 2, 3])
        via_method = lc.score_period_independent_wavelength_model_kernel_config_quality(
            _quality_run_report()
        )
        via_function = score_period_independent_wavelength_model_kernel_config_quality(
            _quality_run_report()
        )
        self.assertEqual(via_method, via_function)


if __name__ == "__main__":
    unittest.main()
