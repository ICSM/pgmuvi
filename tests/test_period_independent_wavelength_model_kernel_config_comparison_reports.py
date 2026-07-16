"""Tests for wavelength candidate comparison report/plot helpers."""

import unittest

import matplotlib
matplotlib.use("Agg")

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    format_period_independent_wavelength_model_kernel_config_comparison_report,
    plot_period_independent_wavelength_model_kernel_config_comparison,
)


def _quality_score_report():
    return {
        "kind": "period_independent_wavelength_model_kernel_config_quality_scores",
        "score_kind": "training_residual_fit_quality",
        "scores_fit_quality": True,
        "score_interpretation": "Training-residual fit-quality score.",
        "runs_fits": False,
        "applies_to_fit": False,
        "advisory_only": True,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "ranking_status": "available",
        "fit_quality_ranking_available": True,
        "n_with_fit_quality": 3,
        "top_ranked_model": "2DDustMean",
        "ranked_results": [
            {
                "quality_rank": 1,
                "rank": 2,
                "model": "2DDustMean",
                "fit_quality_available": True,
                "fit_quality_score": 64.3,
                "score": 64.3,
                "fit_success": True,
                "consensus_success": True,
                "consensus_period": 603.0,
                "consensus_time_kernel_constraint_mode": "period_length",
                "training_nrmse_by_target_scale": 1.4,
                "training_median_abs_standardized_residual": 0.05,
                "training_reduced_chi2": 0.03,
                "training_outlier_fraction_3sigma": 0.0,
            },
            {
                "quality_rank": 2,
                "rank": 1,
                "model": "2DWavelengthDependent",
                "fit_quality_available": True,
                "fit_quality_score": 64.2,
                "score": 64.2,
                "fit_success": True,
                "consensus_success": True,
                "consensus_period": 603.0,
                "consensus_time_kernel_constraint_mode": "period_length",
                "training_nrmse_by_target_scale": 1.42,
                "training_median_abs_standardized_residual": 0.06,
                "training_reduced_chi2": 0.031,
                "training_outlier_fraction_3sigma": 0.0,
            },
            {
                "quality_rank": 3,
                "rank": 4,
                "model": "2D",
                "fit_quality_available": True,
                "fit_quality_score": 18.5,
                "score": 18.5,
                "fit_success": True,
                "consensus_success": True,
                "consensus_period": 603.0,
                "consensus_time_kernel_constraint_mode": "spectral_mixture",
                "training_nrmse_by_target_scale": 3.2,
                "training_median_abs_standardized_residual": 0.09,
                "training_reduced_chi2": 0.16,
                "training_outlier_fraction_3sigma": 0.0,
            },
        ],
    }


class TestPeriodIndependentWavelengthModelKernelConfigComparisonReports(unittest.TestCase):
    def test_text_report_contains_contract_and_ranked_models(self):
        text = format_period_independent_wavelength_model_kernel_config_comparison_report(
            _quality_score_report()
        )

        self.assertIn("Period-independent wavelength model/kernel-config comparison", text)
        self.assertIn("score_kind: training_residual_fit_quality", text)
        self.assertIn("advisory_only: True", text)
        self.assertIn("automatic_model_selection_applied: False", text)
        self.assertIn("selected_model: None", text)
        self.assertIn("2DDustMean", text)
        self.assertIn("2DWavelengthDependent", text)
        self.assertIn("2D", text)

    def test_text_report_max_rows_limits_output(self):
        text = format_period_independent_wavelength_model_kernel_config_comparison_report(
            _quality_score_report(),
            max_rows=1,
        )
        self.assertIn("2DDustMean", text)
        self.assertNotIn("2DWavelengthDependent", text)
        self.assertNotIn("spectral_mixture", text)

    def test_text_report_rejects_invalid_kind(self):
        with self.assertRaisesRegex(ValueError, "score or quality-score report"):
            format_period_independent_wavelength_model_kernel_config_comparison_report(
                {"kind": "period_independent_wavelength_model_kernel_configs"}
            )

    def test_plot_helper_returns_metric_figures(self):
        figures = plot_period_independent_wavelength_model_kernel_config_comparison(
            _quality_score_report()
        )

        self.assertIn("score", figures)
        self.assertIn("training_nrmse_by_target_scale", figures)
        self.assertIn("training_median_abs_standardized_residual", figures)
        for fig in figures.values():
            self.assertGreaterEqual(len(fig.axes), 1)
            fig.clf()

    def test_plot_helper_max_rows_limits_number_of_bars(self):
        figures = plot_period_independent_wavelength_model_kernel_config_comparison(
            _quality_score_report(),
            max_rows=2,
        )
        score_ax = figures["score"].axes[0]
        self.assertEqual(len(score_ax.patches), 2)
        for fig in figures.values():
            fig.clf()

    def test_lightcurve_methods_delegate_to_module_functions(self):
        lc = Lightcurve.__new__(Lightcurve)
        report = _quality_score_report()

        via_method = lc.format_period_independent_wavelength_model_kernel_config_comparison_report(
            report
        )
        via_function = format_period_independent_wavelength_model_kernel_config_comparison_report(
            report
        )
        self.assertEqual(via_method, via_function)

        figures = lc.plot_period_independent_wavelength_model_kernel_config_comparison(report)
        self.assertIn("score", figures)
        for fig in figures.values():
            fig.clf()

    def test_max_rows_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "max_rows"):
            format_period_independent_wavelength_model_kernel_config_comparison_report(
                _quality_score_report(),
                max_rows=0,
            )
        with self.assertRaisesRegex(ValueError, "max_rows"):
            plot_period_independent_wavelength_model_kernel_config_comparison(
                _quality_score_report(),
                max_rows=0,
            )


if __name__ == "__main__":
    unittest.main()
