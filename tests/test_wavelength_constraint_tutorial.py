"""Tests for reusable real-data wavelength-constraint tutorial helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pgmuvi.wavelength_constraint_tutorial import (
    build_tutorial_fit_summary,
    build_wavelength_constraint_position_rows,
    deterministic_time_stratified_observational_channel_indices,
    load_wavelength_constraint_tutorial_lightcurve,
    summarize_observational_channel_residuals,
)


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_CSV = ROOT / "examples/data/10131+3049.csv"


class _FakeConstraint:
    def __init__(self, lower, upper):
        import torch

        self.lower_bound = torch.tensor(lower, dtype=torch.float64)
        self.upper_bound = torch.tensor(upper, dtype=torch.float64)


class _FakeBaseKernel:
    def __init__(self):
        self.raw_lengthscale_constraint = _FakeConstraint([0.2], [2.0])


class _FakeScaleKernel:
    def __init__(self):
        self.base_kernel = _FakeBaseKernel()


class _FakeProductKernel:
    def __init__(self):
        self.kernels = [object(), _FakeScaleKernel()]


class _FakeModel:
    def __init__(self):
        self.covar_module = _FakeProductKernel()


class _FakeLightcurve:
    def __init__(self):
        import torch

        self.model = _FakeModel()
        self.is_fitted = True
        self._parameters = {
            "covar_module.kernels.1.base_kernel.lengthscale": torch.tensor(
                [1.4], dtype=torch.float64
            )
        }

    def get_parameters(self, raw=False, transform=False):
        return dict(self._parameters)

    def get_parameter_workflow_report(self):
        return {
            "available": True,
            "applied": [
                {
                    "parameter": (
                        "covar_module.kernels.1.base_kernel.lengthscale"
                    ),
                    "value_applied": True,
                    "constraint_applied": True,
                    "constraint_action": "tightened",
                    "wavelength_estimate_provenance": {
                        "estimated_value": 0.8,
                        "estimated_constraint": [0.2, 2.0],
                        "effective_constraint": [0.2, 2.0],
                        "diagnostics": {
                            "coordinate_basis": "physical_wavelength"
                        },
                    },
                }
            ],
            "skipped": [],
        }


class TestObservationalChannelTutorialSampling(unittest.TestCase):
    def test_sampling_is_deterministic_and_preserves_time_endpoints(self):
        times = np.concatenate((np.arange(20.0), np.array([3.0, 2.0, 1.0])))
        channels = np.array(["A"] * 20 + ["B"] * 3)

        first = deterministic_time_stratified_observational_channel_indices(
            times,
            channels,
            max_samples_per_observational_channel=5,
        )
        second = deterministic_time_stratified_observational_channel_indices(
            times,
            channels,
            max_samples_per_observational_channel=5,
        )

        np.testing.assert_array_equal(first, second)
        retained_a = first[channels[first] == "A"]
        retained_b = first[channels[first] == "B"]
        self.assertEqual(retained_a.size, 5)
        self.assertEqual(retained_b.size, 3)
        self.assertIn(0, retained_a)
        self.assertIn(19, retained_a)
        np.testing.assert_array_equal(retained_b, np.array([20, 21, 22]))

    def test_representative_csv_retains_every_channel_and_wavelength(self):
        lightcurve, summary = load_wavelength_constraint_tutorial_lightcurve(
            REPRESENTATIVE_CSV,
            max_samples_per_observational_channel=100,
        )

        self.assertEqual(summary["n_rows_original"], 10815)
        self.assertEqual(summary["n_rows_retained"], 789)
        self.assertEqual(summary["n_observational_channels_original"], 17)
        self.assertEqual(summary["n_observational_channels_retained"], 17)
        self.assertEqual(summary["n_physical_wavelengths_original"], 16)
        self.assertEqual(summary["n_physical_wavelengths_retained"], 16)
        self.assertEqual(lightcurve.xdata.shape[0], 789)
        self.assertEqual(len(lightcurve.observational_channel_labels), 789)
        self.assertFalse(summary["check_sampling"])
        self.assertEqual(
            summary["n_rows_before_sampling_quality_filter"],
            789,
        )
        self.assertEqual(
            summary["n_rows_removed_by_sampling_quality_filter"],
            0,
        )
        self.assertTrue(summary["instrument_calibration_tbd"])
        shared = summary["observational_channels_by_shared_wavelength"]
        self.assertEqual(len(shared), 1)
        self.assertEqual(len(next(iter(shared.values()))), 2)

    def test_sampling_quality_filter_is_forwarded_and_reported(self):
        rows = ["time,flux,flux_error,wavelength,band"]
        rows.extend(
            f"{time},10.0,1.0,1.0,A"
            for time in range(20)
        )
        rows.extend(
            f"{time},10.0,1.0,2.0,B"
            for time in range(4)
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "sampling.csv"
            source_path.write_text(
                "\n".join(rows) + "\n",
                encoding="utf-8",
            )
            with self.assertWarnsRegex(
                UserWarning,
                "Skipping band",
            ):
                lightcurve, summary = (
                    load_wavelength_constraint_tutorial_lightcurve(
                        source_path,
                        max_samples_per_observational_channel=100,
                        check_sampling=True,
                    )
                )

        self.assertTrue(summary["check_sampling"])
        self.assertEqual(summary["sampling_kwargs"], {})
        self.assertEqual(
            summary["n_rows_before_sampling_quality_filter"],
            24,
        )
        self.assertEqual(
            summary["n_rows_removed_by_sampling_quality_filter"],
            4,
        )
        self.assertEqual(summary["n_rows_retained"], 20)
        self.assertEqual(
            summary[
                "n_physical_wavelengths_before_sampling_quality_filter"
            ],
            2,
        )
        self.assertEqual(
            summary["n_physical_wavelengths_retained"],
            1,
        )
        self.assertEqual(
            summary[
                "n_observational_channels_before_sampling_quality_filter"
            ],
            2,
        )
        self.assertEqual(
            summary["n_observational_channels_retained"],
            1,
        )
        self.assertEqual(lightcurve.xdata.shape[0], 20)
        self.assertEqual(
            set(lightcurve.observational_channel_labels),
            {"A"},
        )

    def test_quota_must_be_an_integer_at_least_two(self):
        with self.assertRaises(TypeError):
            deterministic_time_stratified_observational_channel_indices(
                [0.0, 1.0],
                ["A", "A"],
                max_samples_per_observational_channel=2.5,
            )
        with self.assertRaises(ValueError):
            deterministic_time_stratified_observational_channel_indices(
                [0.0, 1.0],
                ["A", "A"],
                max_samples_per_observational_channel=1,
            )


class TestTutorialFitDiagnostics(unittest.TestCase):
    def test_channel_residual_summary_does_not_merge_shared_wavelength(self):
        predictions = {
            "time": np.array([0.0, 1.0, 0.5, 1.5]),
            "physical_wavelength": np.array([0.65, 0.65, 0.65, 0.65]),
            "observational_channel": np.array(["A", "A", "B", "B"]),
            "residual": np.array([1.0, -1.0, 2.0, -2.0]),
            "standardized_residual": np.array([0.5, -0.5, 1.0, -1.0]),
        }

        rows = summarize_observational_channel_residuals(predictions)

        self.assertEqual([row["observational_channel"] for row in rows], ["A", "B"])
        self.assertEqual([row["n_points"] for row in rows], [2, 2])
        self.assertEqual([row["physical_wavelength"] for row in rows], [0.65, 0.65])
        self.assertAlmostEqual(rows[0]["rmse"], 1.0)
        self.assertAlmostEqual(rows[1]["rmse"], 2.0)

    def test_constraint_rows_report_registration_order_and_boundary_distance(self):
        lightcurve = _FakeLightcurve()
        parameter = "covar_module.kernels.1.base_kernel.lengthscale"
        rows = build_wavelength_constraint_position_rows(
            lightcurve,
            {parameter: [np.array([0.8]), np.array([1.4])]},
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["applies_to"], "covariance")
        self.assertEqual(row["application_order"], "constraint_then_value")
        self.assertTrue(row["constraint_registered"])
        self.assertTrue(row["initial_inside_constraint"])
        self.assertTrue(row["fitted_inside_constraint"])
        self.assertAlmostEqual(row["fractional_position_within_bounds"], 2.0 / 3.0)
        self.assertAlmostEqual(row["minimum_distance_to_bound"], 0.6)

    def test_compact_fit_summary_reports_finite_quality_fields(self):
        lightcurve = _FakeLightcurve()
        parameter = "covar_module.kernels.1.base_kernel.lengthscale"
        constraint_rows = build_wavelength_constraint_position_rows(
            lightcurve,
            {parameter: [np.array([0.8]), np.array([1.4])]},
        )
        summary = build_tutorial_fit_summary(
            model_name="2DWavelengthDependent",
            lightcurve=lightcurve,
            fit_result={
                "loss": [3.0, 2.0, 1.5],
                parameter: [np.array([0.8]), np.array([1.4])],
                "training_recovered_from_failure": False,
            },
            sampling_summary={
                "n_rows_retained": 20,
                "n_observational_channels_retained": 3,
                "n_physical_wavelengths_retained": 2,
            },
            constraint_rows=constraint_rows,
            predictions={
                "residual": np.array([1.0, -1.0]),
                "predictive_mean": np.array([4.0, 5.0]),
                "predictive_variance": np.array([0.5, 0.5]),
                "observational_channel": np.array(["A", "B"]),
                "physical_wavelength": np.array([0.55, 0.65]),
            },
            warning_messages=["example warning"],
        )

        self.assertTrue(summary["fit_executed"])
        self.assertTrue(summary["objective_improved"])
        self.assertEqual(summary["n_iterations"], 3)
        self.assertEqual(summary["n_observations"], 2)
        self.assertEqual(summary["n_observational_channels"], 2)
        self.assertEqual(summary["n_physical_wavelengths"], 2)
        self.assertEqual(summary["gp_training_scope"]["n_observations"], 2)
        self.assertEqual(
            summary["consensus_input_scope"]["n_observations"],
            20,
        )
        self.assertEqual(summary["nonfinite_prediction_count"], 0)
        self.assertEqual(summary["negative_variance_count"], 0)
        self.assertAlmostEqual(summary["residual_rmse"], 1.0)
        self.assertEqual(summary["warning_count"], 1)


if __name__ == "__main__":
    unittest.main()
