import unittest

import gpytorch
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.spectral_mixture_ard_diagnostics import (
    SPECTRAL_MIXTURE_ARD_DIAGNOSTIC_SCHEMA_VERSION,
    diagnose_spectral_mixture_ard,
)


class _FakeKernel:
    def __init__(self, means, scales, mean_bounds, scale_bounds):
        self.mixture_means = torch.as_tensor(means, dtype=torch.float64)
        self.mixture_scales = torch.as_tensor(scales, dtype=torch.float64)
        self.raw_mixture_means = torch.zeros_like(self.mixture_means)
        self.raw_mixture_scales = torch.zeros_like(self.mixture_scales)
        self.raw_mixture_means_constraint = gpytorch.constraints.Interval(
            torch.as_tensor(mean_bounds[0], dtype=torch.float64),
            torch.as_tensor(mean_bounds[1], dtype=torch.float64),
        )
        self.raw_mixture_scales_constraint = gpytorch.constraints.Interval(
            torch.as_tensor(scale_bounds[0], dtype=torch.float64),
            torch.as_tensor(scale_bounds[1], dtype=torch.float64),
        )


class _FakeModel:
    def __init__(self, kernel):
        self.covar_module = kernel


class _FakeLightcurve:
    def __init__(self, kernel, *, factors=(100.0, 4.0)):
        self.model = _FakeModel(kernel)
        self.parameter_workflow_result = {}
        for parameter_name in ("mixture_means", "mixture_scales"):
            values = getattr(kernel, parameter_name)
            raw_initial = values.detach().clone()
            raw_initial[..., 0] /= factors[0]
            raw_initial[..., 1] /= factors[1]
            self.parameter_workflow_result[f"covar_module.{parameter_name}"] = {
                "spectral_mixture_ard_provenance": {
                    "diagnostics": {
                        "raw_initial_value": raw_initial.tolist(),
                        "model_initial_value": values.tolist(),
                    }
                }
            }


def _diagnostic_lightcurve():
    kernel = _FakeKernel(
        means=[[[1.10, 19.60]], [[5.00, 10.20]]],
        scales=[[[0.96, 3.00]], [[0.50, 1.10]]],
        mean_bounds=(
            [[[1.0, 10.0]]],
            [[[9.0, 20.0]]],
        ),
        scale_bounds=(
            [[[0.1, 1.0]]],
            [[[1.0, 5.0]]],
        ),
    )
    return _FakeLightcurve(kernel)


class TestSpectralMixtureArdDiagnostics(unittest.TestCase):
    def test_reports_parameter_dimension_and_bound_side(self):
        diagnostics = diagnose_spectral_mixture_ard(
            _diagnostic_lightcurve(),
            boundary_tolerance_fraction=0.05,
            requested_num_mixtures=2,
        )

        self.assertEqual(
            diagnostics["schema_version"],
            SPECTRAL_MIXTURE_ARD_DIAGNOSTIC_SCHEMA_VERSION,
        )
        self.assertTrue(diagnostics["available"])
        self.assertEqual(
            diagnostics["coordinate_order"],
            ["temporal_frequency", "wavelength_frequency"],
        )
        self.assertEqual(diagnostics["boundary_pressure_scope"], "both")
        self.assertEqual(
            diagnostics["boundary_hit_counts_by_parameter"],
            {"mixture_means": 3, "mixture_scales": 2},
        )
        self.assertEqual(
            diagnostics["boundary_hit_counts_by_side"],
            {"lower": 3, "upper": 2},
        )
        self.assertEqual(
            diagnostics["boundary_component_counts_by_dimension"],
            {"temporal_frequency": 1, "wavelength_frequency": 2},
        )

        identities = {
            (
                hit["parameter"],
                hit["component_index"],
                hit["dimension_name"],
                hit["bound_side"],
            )
            for hit in diagnostics["boundary_hits"]
        }
        self.assertEqual(
            identities,
            {
                ("mixture_means", 0, "temporal_frequency", "lower"),
                ("mixture_means", 0, "wavelength_frequency", "upper"),
                ("mixture_means", 1, "wavelength_frequency", "lower"),
                ("mixture_scales", 0, "temporal_frequency", "upper"),
                ("mixture_scales", 1, "wavelength_frequency", "lower"),
            },
        )

    def test_reports_model_raw_input_and_gpytorch_raw_values(self):
        diagnostics = diagnose_spectral_mixture_ard(_diagnostic_lightcurve())
        means = diagnostics["parameters"]["mixture_means"]

        self.assertTrue(means["raw_input_coordinate_available"])
        self.assertEqual(means["coordinate_transform_factors"], [100.0, 4.0])
        self.assertAlmostEqual(means["raw_input_coordinate_values"][0][0], 0.011)
        self.assertAlmostEqual(means["raw_input_coordinate_values"][0][1], 4.9)
        self.assertAlmostEqual(
            means["raw_input_coordinate_lower_bounds"][0][0],
            0.01,
        )
        self.assertEqual(means["gpytorch_raw_parameter_values"][0][0], 0.0)

    def test_distinguishes_actual_one_component_from_explicit_fixed_one(self):
        kernel = _FakeKernel(
            means=[[[2.0, 3.0]]],
            scales=[[[0.5, 1.5]]],
            mean_bounds=([[[1.0, 1.0]]], [[[5.0, 5.0]]]),
            scale_bounds=([[[0.1, 0.5]]], [[[1.0, 2.0]]]),
        )
        lc = _FakeLightcurve(kernel)

        inferred = diagnose_spectral_mixture_ard(lc)
        fixed = diagnose_spectral_mixture_ard(
            lc,
            requested_num_mixtures=1,
        )

        self.assertTrue(inferred["num_mixtures_is_one"])
        self.assertFalse(inferred["num_mixtures_fixed_at_one"])
        self.assertTrue(fixed["num_mixtures_is_one"])
        self.assertTrue(fixed["num_mixtures_fixed_at_one"])
        self.assertEqual(fixed["num_mixtures_request_source"], "explicit_fit_kwargs")

    def test_one_dimensional_spectral_mixture_kernel_is_out_of_scope(self):
        kernel = _FakeKernel(
            means=[[[2.0]], [[3.0]]],
            scales=[[[0.5]], [[0.6]]],
            mean_bounds=([[[1.0]]], [[[5.0]]]),
            scale_bounds=([[[0.1]]], [[[1.0]]]),
        )
        fitted = type("Fitted", (), {"model": _FakeModel(kernel)})()
        diagnostics = diagnose_spectral_mixture_ard(fitted)
        self.assertFalse(diagnostics["available"])
        self.assertIn("does not expose", diagnostics["reason"])

    def test_unavailable_for_non_spectral_mixture_object(self):
        diagnostics = diagnose_spectral_mixture_ard(object())
        self.assertFalse(diagnostics["available"])
        self.assertEqual(diagnostics["n_boundary_hits"], 0)
        self.assertEqual(diagnostics["boundary_pressure_scope"], "none")

    def test_rejects_invalid_tolerances(self):
        with self.assertRaises(ValueError):
            diagnose_spectral_mixture_ard(
                _diagnostic_lightcurve(),
                boundary_tolerance_fraction=0.5,
            )
        with self.assertRaises(ValueError):
            diagnose_spectral_mixture_ard(
                _diagnostic_lightcurve(),
                boundary_tolerance_fraction=0.01,
                at_bound_tolerance_fraction=0.02,
            )

    def test_real_2d_model_exposes_registered_dimension_specific_bounds(self):
        rows = []
        fluxes = []
        for wavelength in (1.0, 2.0, 5.0):
            for time in (0.0, 1.0, 2.0, 100.0):
                rows.append([time, wavelength])
                fluxes.append(wavelength + 0.01 * time)
        lc = Lightcurve(
            torch.tensor(rows),
            torch.tensor(fluxes),
            xtransform="minmax",
        )
        lc.set_model("2D", num_mixtures=2)
        lc.set_default_constraints()
        lc._apply_parameter_workflow_estimates()

        diagnostics = diagnose_spectral_mixture_ard(
            lc,
            requested_num_mixtures=2,
        )
        for parameter_name in ("mixture_means", "mixture_scales"):
            record = diagnostics["parameters"][parameter_name]
            self.assertTrue(record["constraint_registered"])
            self.assertEqual(len(record["model_coordinate_values"]), 2)
            self.assertEqual(len(record["model_coordinate_values"][0]), 2)
            self.assertTrue(record["raw_input_coordinate_available"])


if __name__ == "__main__":
    unittest.main()
