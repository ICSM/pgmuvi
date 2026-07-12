"""Tests for period-independent wavelength parameter planning."""

import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    build_period_independent_wavelength_parameter_plan,
    diagnose_period_independent_wavelength_structure,
)


def _make_multiband_lightcurve(
    *,
    n_per_band=80,
    wavelengths=(1.0, 2.0, 4.0),
    medians=(10.0, 20.0, 40.0),
    amplitudes=(0.4, 0.8, 1.6),
):
    times = []
    wls = []
    ys = []
    yerrs = []
    t = np.linspace(0.0, 100.0, n_per_band)
    for wl, median, amp in zip(wavelengths, medians, amplitudes, strict=True):
        phase = 2.0 * np.pi * t / 25.0
        y = median + amp * np.sin(phase) + 0.05 * amp * np.cos(2.0 * phase)
        times.append(t)
        wls.append(np.full_like(t, wl))
        ys.append(y)
        yerrs.append(np.full_like(t, 0.05))

    x = np.column_stack([np.concatenate(times), np.concatenate(wls)])
    y = np.concatenate(ys)
    yerr = np.concatenate(yerrs)
    return Lightcurve(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        yerr=torch.as_tensor(yerr, dtype=torch.float64),
        max_samples=None,
    )


class TestPeriodIndependentWavelengthParameterPlan(unittest.TestCase):
    def test_plan_is_advisory_and_period_independent(self):
        lc = _make_multiband_lightcurve()

        plan = lc.build_period_independent_wavelength_parameter_plan()

        self.assertEqual(plan["kind"], "period_independent_wavelength_parameter_plan")
        self.assertEqual(plan["stage"], "prefit_period_independent_advisory")
        self.assertTrue(plan["is_period_independent"])
        self.assertFalse(plan["uses_temporal_consensus"])
        self.assertFalse(plan["uses_period_or_frequency"])
        self.assertFalse(plan["applies_to_fit"])

    def test_module_function_matches_lightcurve_method(self):
        lc = _make_multiband_lightcurve()

        via_method = lc.build_period_independent_wavelength_parameter_plan()
        via_function = build_period_independent_wavelength_parameter_plan(lc)

        self.assertEqual(via_method, via_function)

    def test_can_reuse_precomputed_diagnostics_report(self):
        lc = _make_multiband_lightcurve()
        report = diagnose_period_independent_wavelength_structure(lc)

        from_report = build_period_independent_wavelength_parameter_plan(report)
        from_kwarg = lc.build_period_independent_wavelength_parameter_plan(
            diagnostics_report=report
        )

        self.assertEqual(from_report, from_kwarg)
        self.assertEqual(
            from_report["source_report_kind"],
            "period_independent_wavelength_diagnostics",
        )

    def test_increasing_median_flux_prefers_dust_mean_then_power_law(self):
        lc = _make_multiband_lightcurve(medians=(10.0, 20.0, 40.0))

        plan = lc.build_period_independent_wavelength_parameter_plan()
        ranked = [entry["model"] for entry in plan["recommended_models"]]

        self.assertEqual(ranked[0], "2DDustMean")
        self.assertIn("2DPowerLawMean", ranked)
        self.assertIn("2DWavelengthDependent", ranked)

    def test_non_monotonic_median_flux_prefers_flexible_model(self):
        lc = _make_multiband_lightcurve(medians=(10.0, 40.0, 20.0))

        plan = lc.build_period_independent_wavelength_parameter_plan()

        self.assertEqual(plan["primary_recommended_model"], "2DWavelengthDependent")

    def test_power_law_exponent_tracks_loglog_slope(self):
        lc = _make_multiband_lightcurve(
            wavelengths=(1.0, 2.0, 4.0),
            medians=(3.0, 12.0, 48.0),
            amplitudes=(0.2, 0.4, 0.8),
        )

        plan = lc.build_period_independent_wavelength_parameter_plan()
        slope = plan["summary"]["median_flux_loglog_slope"]
        exponent = plan["model_parameter_suggestions"]["2DPowerLawMean"][
            "initial_values"
        ]["mean_module.exponent"]

        self.assertAlmostEqual(exponent, slope)
        self.assertGreater(exponent, 1.5)
        self.assertLess(exponent, 2.5)

    def test_dust_mean_suggestions_are_log_parameterized_and_finite(self):
        lc = _make_multiband_lightcurve()

        plan = lc.build_period_independent_wavelength_parameter_plan()
        dust = plan["model_parameter_suggestions"]["2DDustMean"]
        init = dust["initial_values"]
        physical = dust["physical_initial_values"]

        self.assertGreater(physical["amplitude"], 0.0)
        self.assertGreater(physical["alpha"], 0.0)
        self.assertTrue(np.isfinite(init["mean_module.log_amplitude"]))
        self.assertTrue(np.isfinite(init["mean_module.log_tau"]))
        self.assertTrue(np.isfinite(init["mean_module.log_alpha"]))

    def test_constraints_enclose_power_law_initial_values(self):
        lc = _make_multiband_lightcurve()

        plan = lc.build_period_independent_wavelength_parameter_plan()
        power = plan["model_parameter_suggestions"]["2DPowerLawMean"]
        for name, value in power["initial_values"].items():
            lower, upper = power["constraints"][name]
            self.assertLessEqual(lower, value)
            self.assertGreaterEqual(upper, value)

    def test_does_not_mutate_consensus_diagnostics(self):
        lc = _make_multiband_lightcurve()
        lc.consensus_diagnostics = {"sentinel": True}

        lc.build_period_independent_wavelength_parameter_plan()

        self.assertEqual(lc.consensus_diagnostics, {"sentinel": True})

    def test_explicit_advisory_contract_fields_are_present(self):
        lc = _make_multiband_lightcurve()

        plan = lc.build_period_independent_wavelength_parameter_plan()

        self.assertTrue(plan["advisory_only"])
        self.assertFalse(plan["hard_model_exclusions"])
        self.assertFalse(plan["automatic_constraints_applied"])
        self.assertFalse(plan["automatic_initialization_applied"])
        self.assertEqual(
            [entry["model"] for entry in plan["ranked_candidates"]],
            [entry["model"] for entry in plan["recommended_models"]],
        )
        self.assertTrue(
            all(entry["hard_exclusion"] is False for entry in plan["ranked_candidates"])
        )

    def test_rejects_non_period_independent_report(self):
        with self.assertRaisesRegex(ValueError, "period-independent"):
            build_period_independent_wavelength_parameter_plan({"kind": "other"})


if __name__ == "__main__":
    unittest.main()
