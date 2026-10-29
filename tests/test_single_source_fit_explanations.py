"""Tests for per-channel and pairwise fit explanations."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pgmuvi.single_source_analysis import (
    summarize_single_source_fit_explanations,
)


def _predictions(shape_b):
    phase = np.linspace(0.0, 1.0, 24, endpoint=False)
    time_a = 100.0 + 600.0 * phase
    time_b = 1300.0 + 600.0 * phase
    shape_a = np.sin(2.0 * np.pi * phase)
    observed_a = 1.0 + 0.2 * shape_a
    observed_b = 10.0 + 2.0 * shape_b
    observed = np.concatenate([observed_a, observed_b])
    predictive_mean = observed.copy()
    predictive_sd = np.full(observed.shape, 0.1)
    residual = observed - predictive_mean
    return {
        "time": np.concatenate([time_a, time_b]),
        "physical_wavelength": np.concatenate(
            [np.full(phase.shape, 0.48), np.full(phase.shape, 0.51)]
        ),
        "observational_channel": np.asarray(
            ["ZTF/g"] * phase.size + ["Gaia/GBP"] * phase.size
        ),
        "observed": observed,
        "predictive_mean": predictive_mean,
        "predictive_standard_deviation": predictive_sd,
        "measurement_standard_deviation": np.full(
            observed.shape, 0.02
        ),
        "residual": residual,
        "standardized_residual": residual / predictive_sd,
    }


class _RBFLikeKernel:
    def __init__(self, active_dimension, lengthscale):
        self.active_dimension = active_dimension
        self.lengthscale = float(lengthscale)

    def __call__(self, left, right=None):
        if right is None:
            right = left
        left_values = left[:, self.active_dimension].reshape(-1, 1)
        right_values = right[:, self.active_dimension].reshape(1, -1)
        distance = (left_values - right_values) / self.lengthscale
        return torch.exp(-0.5 * distance.square())


class _DummyCovarianceModule:
    def __init__(self):
        self.kernels = (
            _RBFLikeKernel(0, 600.0),
            _RBFLikeKernel(1, 0.1),
        )


class _DummyModel:
    def __init__(self):
        self.covar_module = _DummyCovarianceModule()


class _DummyLightcurve:
    def __init__(self):
        self.model = _DummyModel()
        self.xdata = torch.zeros((1, 2), dtype=torch.float64)

    @staticmethod
    def transform_x(values):
        return values


class TestSingleSourceFitExplanations(unittest.TestCase):
    def test_similar_shapes_and_one_dex_offset_are_not_called_calibration_proof(self):
        phase = np.linspace(0.0, 1.0, 24, endpoint=False)
        diagnostics = summarize_single_source_fit_explanations(
            _predictions(np.sin(2.0 * np.pi * phase)),
            period=600.0,
        )
        pair = diagnostics["pair_rows"][0]
        self.assertGreater(
            pair["normalized_phase_shape_correlation"],
            0.99,
        )
        self.assertAlmostEqual(
            pair["absolute_median_flux_difference_dex"],
            1.0,
            places=6,
        )
        self.assertEqual(
            pair["interpretation"],
            "passband_sed_or_calibration_incompatibility_candidate",
        )
        self.assertIn(
            "cannot distinguish full passband/SED effects",
            diagnostics["warning_messages"][0],
        )

    def test_different_phase_shapes_are_not_labelled_flux_scale_candidate(self):
        phase = np.linspace(0.0, 1.0, 24, endpoint=False)
        diagnostics = summarize_single_source_fit_explanations(
            _predictions(np.sin(4.0 * np.pi * phase)),
            period=600.0,
        )
        pair = diagnostics["pair_rows"][0]
        self.assertEqual(
            pair["interpretation"],
            "possible_chromatic_shape_or_model_family_mismatch",
        )

    def test_channel_sampling_phase_and_predictive_metrics_are_reported(self):
        phase = np.linspace(0.0, 1.0, 24, endpoint=False)
        diagnostics = summarize_single_source_fit_explanations(
            _predictions(np.sin(2.0 * np.pi * phase)),
            period=600.0,
        )
        row = diagnostics["observational_channel_rows"][0]
        for key in (
            "time_baseline",
            "median_positive_cadence",
            "maximum_gap_fraction",
            "phase_bin_coverage_fraction",
            "circular_phase_coverage_fraction",
            "predictive_mean_bias",
            "rmse",
            "standardized_residual_rms",
            "median_measurement_standard_deviation",
            "median_predictive_standard_deviation",
            "empirical_95_percent_coverage",
            "nearest_pair_interpretation",
            "fit_interpretation",
            "final_diagnostic_interpretation",
        ):
            with self.subTest(key=key):
                self.assertIn(key, row)
        self.assertEqual(row["empirical_95_percent_coverage"], 1.0)
        self.assertEqual(
            row["fit_interpretation"],
            "adequate_under_current_training_diagnostics",
        )

    def test_fitted_kernel_support_is_reported_when_available(self):
        phase = np.linspace(0.0, 1.0, 24, endpoint=False)
        diagnostics = summarize_single_source_fit_explanations(
            _predictions(np.sin(2.0 * np.pi * phase)),
            period=600.0,
            lightcurve=_DummyLightcurve(),
        )
        pair = diagnostics["pair_rows"][0]
        self.assertTrue(pair["kernel_support_available"])
        self.assertIsNotNone(
            pair["fitted_wavelength_kernel_correlation"]
        )
        self.assertIsNotNone(
            pair[
                "phase_aligned_temporal_kernel_"
                "median_absolute_correlation"
            ]
        )
        self.assertIsNotNone(
            pair["phase_aligned_total_kernel_support"]
        )

    def test_invalid_period_is_rejected(self):
        phase = np.linspace(0.0, 1.0, 24, endpoint=False)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            summarize_single_source_fit_explanations(
                _predictions(np.sin(2.0 * np.pi * phase)),
                period=0.0,
            )


if __name__ == "__main__":
    unittest.main()
