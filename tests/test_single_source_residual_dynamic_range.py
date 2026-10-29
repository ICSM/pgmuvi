"""Tests for scale-aware single-source residual diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from pgmuvi.single_source_analysis import (
    build_phase_folded_prediction_summary,
)
from pgmuvi.wavelength_constraint_tutorial import (
    summarize_observational_channel_residuals,
)


def _scaled_predictions():
    time = np.asarray([0.0, 1.0, 2.0, 3.0] * 2)
    channel = np.asarray(["low"] * 4 + ["high"] * 4)
    wavelength = np.asarray([0.5] * 4 + [0.6] * 4)
    observed_low = np.asarray([1.0, 2.0, 3.0, 4.0])
    observed_high = 1000.0 * observed_low
    residual_low = np.asarray([0.1, -0.1, 0.1, -0.1])
    residual_high = 1000.0 * residual_low
    standardized = np.asarray([1.0, -1.0, 1.0, -1.0] * 2)
    observed = np.concatenate([observed_low, observed_high])
    residual = np.concatenate([residual_low, residual_high])
    return {
        "time": time,
        "observational_channel": channel,
        "physical_wavelength": wavelength,
        "observed": observed,
        "predictive_mean": observed - residual,
        "predictive_standard_deviation": np.abs(residual),
        "residual": residual,
        "standardized_residual": standardized,
    }


class TestSingleSourceResidualDynamicRange(unittest.TestCase):
    def test_dimensionless_channel_metrics_are_scale_invariant(self):
        rows = summarize_observational_channel_residuals(
            _scaled_predictions()
        )
        self.assertEqual(len(rows), 2)
        low, high = rows
        self.assertAlmostEqual(high["rmse"] / low["rmse"], 1000.0)
        self.assertAlmostEqual(
            low["standardized_residual_rms"],
            high["standardized_residual_rms"],
        )
        self.assertAlmostEqual(
            low["fractional_rmse_over_abs_median_flux"],
            high["fractional_rmse_over_abs_median_flux"],
        )
        self.assertAlmostEqual(
            low["normalized_rmse_over_robust_amplitude"],
            high["normalized_rmse_over_robust_amplitude"],
        )
        self.assertEqual(low["empirical_95_percent_coverage"], 1.0)
        self.assertEqual(high["empirical_95_percent_coverage"], 1.0)

    def test_phase_summary_preserves_standardized_residuals(self):
        predictions = _scaled_predictions()
        summary = build_phase_folded_prediction_summary(
            predictions,
            period=4.0,
        )
        np.testing.assert_allclose(
            summary["standardized_residual"],
            predictions["standardized_residual"],
        )
        self.assertEqual(
            summary["combined_signed_residual_metric"],
            "standardized_residual",
        )
        self.assertEqual(
            summary["raw_signed_residual_presentation"],
            "per_channel_independent_y_ranges",
        )
        for row in summary["observational_channel_rows"]:
            self.assertAlmostEqual(
                row["standardized_residual_rms"],
                1.0,
            )
            self.assertEqual(
                row["empirical_95_percent_coverage"],
                1.0,
            )


    def test_phase_summary_derives_standardized_residuals_when_possible(self):
        predictions = _scaled_predictions()
        expected = predictions.pop("standardized_residual")
        predictions["predictive_standard_deviation"] = np.abs(
            predictions["residual"] / expected
        )
        summary = build_phase_folded_prediction_summary(
            predictions,
            period=4.0,
        )
        np.testing.assert_allclose(
            summary["standardized_residual"],
            expected,
        )
        self.assertEqual(
            summary["standardized_residual_source"],
            "derived_from_predictive_standard_deviation",
        )

    def test_phase_summary_preserves_legacy_raw_only_input(self):
        predictions = _scaled_predictions()
        predictions.pop("standardized_residual")
        predictions.pop("predictive_standard_deviation")
        summary = build_phase_folded_prediction_summary(
            predictions,
            period=4.0,
        )
        self.assertEqual(
            summary["standardized_residual_source"],
            "unavailable",
        )
        self.assertTrue(
            all(
                row["standardized_residual_rms"] is None
                for row in summary["observational_channel_rows"]
            )
        )

    def test_channel_summary_preserves_legacy_input_without_observed_flux(self):
        predictions = _scaled_predictions()
        predictions.pop("observed")
        predictions.pop("predictive_mean")
        rows = summarize_observational_channel_residuals(predictions)
        self.assertAlmostEqual(rows[0]["rmse"], 0.1)
        self.assertAlmostEqual(rows[1]["rmse"], 100.0)
        self.assertTrue(
            all(row["observed_flux_source"] == "unavailable" for row in rows)
        )
        self.assertTrue(
            all(
                row["fractional_rmse_over_abs_median_flux"] is None
                for row in rows
            )
        )

    def test_phase_summary_requires_row_aligned_standardized_residuals(self):
        predictions = _scaled_predictions()
        predictions["standardized_residual"] = np.asarray([1.0])
        with self.assertRaisesRegex(ValueError, "equal row counts"):
            build_phase_folded_prediction_summary(
                predictions,
                period=4.0,
            )


if __name__ == "__main__":
    unittest.main()
