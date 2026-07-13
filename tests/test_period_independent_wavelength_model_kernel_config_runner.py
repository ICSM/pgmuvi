"""Tests for running period-independent wavelength model/kernel configs."""

import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    run_period_independent_wavelength_model_kernel_configs,
)


def _make_multiband_lightcurve(
    *,
    n_per_band=40,
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


def _model_kernel_config_report():
    return {
        "kind": "period_independent_wavelength_model_kernel_configs",
        "model_kernel_configs": [
            {
                "model_kernel_config_id": "rank1_2DDustMean",
                "rank": 1,
                "model": "2DDustMean",
                "source": "parameter_plan",
                "recommendation_strength": "advisory",
                "hard_exclusion": False,
                "fit_kwargs": {
                    "model": "2DDustMean",
                    "fit_strategy": "consensus",
                    "time_kernel_type": "quasi_periodic",
                    "training_iter": 2,
                },
                "parameter_suggestions_applied": False,
                "applies_constraints": False,
            },
            {
                "model_kernel_config_id": "rank2_2D_baseline",
                "rank": 2,
                "model": "2D",
                "source": "baseline_comparison",
                "recommendation_strength": "baseline_comparison",
                "hard_exclusion": False,
                "fit_kwargs": {
                    "model": "2D",
                    "fit_strategy": "consensus",
                    "training_iter": 2,
                },
                "parameter_suggestions_applied": False,
                "applies_constraints": False,
            },
        ],
    }


class TestPeriodIndependentWavelengthModelKernelConfigRunner(unittest.TestCase):
    def test_runner_report_contract_is_advisory_and_nonselecting(self):
        lc = _make_multiband_lightcurve()

        def runner(candidate_lc, fit_kwargs, candidate):
            candidate_lc.consensus_diagnostics = {
                "consensus_success": True,
                "consensus_period": 25.0,
                "consensus_time_kernel_constraint_mode": "period_length",
                "n_accepted_bands": 3,
                "n_rejected_bands": 0,
            }
            return {"model": fit_kwargs["model"]}

        report = lc.run_period_independent_wavelength_model_kernel_configs(
            model_kernel_config_report=_model_kernel_config_report(),
            fit_runner=runner,
        )

        self.assertEqual(
            report["kind"], "period_independent_wavelength_model_kernel_config_results"
        )
        self.assertTrue(report["runs_fits"])
        self.assertTrue(report["applies_to_fit"])
        self.assertTrue(report["model_kernel_config_state_isolated"])
        self.assertFalse(report["mutates_input_lightcurve"])
        self.assertTrue(report["advisory_only"])
        self.assertFalse(report["automatic_model_selection_applied"])
        self.assertIsNone(report["selected_model"])
        self.assertFalse(report["automatic_constraints_applied"])
        self.assertFalse(report["automatic_initialization_applied"])
        self.assertEqual(report["n_passed"], 2)
        self.assertEqual(report["n_failed"], 0)

    def test_module_function_matches_lightcurve_method(self):
        lc = _make_multiband_lightcurve()

        def runner(candidate_lc, fit_kwargs, candidate):
            candidate_lc.consensus_diagnostics = {"consensus_success": True}
            return fit_kwargs["model"]

        via_method = lc.run_period_independent_wavelength_model_kernel_configs(
            model_kernel_config_report=_model_kernel_config_report(), fit_runner=runner
        )
        via_function = run_period_independent_wavelength_model_kernel_configs(
            lc, model_kernel_config_report=_model_kernel_config_report(), fit_runner=runner
        )
        self.assertEqual(via_method, via_function)

    def test_model_kernel_config_results_alias_matches_outcomes(self):
        lc = _make_multiband_lightcurve()

        def runner(candidate_lc, fit_kwargs, candidate):
            candidate_lc.consensus_diagnostics = {"consensus_success": True}
            return None

        report = lc.run_period_independent_wavelength_model_kernel_configs(
            model_kernel_config_report=_model_kernel_config_report(), fit_runner=runner
        )

        self.assertIn("outcomes", report)
        self.assertIn("model_kernel_config_results", report)
        self.assertEqual(report["model_kernel_config_results"], report["outcomes"])

    def test_fit_kwargs_are_passed_to_runner(self):
        lc = _make_multiband_lightcurve()
        seen = []

        def runner(candidate_lc, fit_kwargs, candidate):
            seen.append((candidate["model"], dict(fit_kwargs)))
            candidate_lc.consensus_diagnostics = {"consensus_success": True}
            return None

        lc.run_period_independent_wavelength_model_kernel_configs(
            model_kernel_config_report=_model_kernel_config_report(), fit_runner=runner
        )

        self.assertEqual(seen[0][0], "2DDustMean")
        self.assertEqual(seen[0][1]["time_kernel_type"], "quasi_periodic")
        self.assertEqual(seen[1][0], "2D")
        self.assertNotIn("time_kernel_type", seen[1][1])

    def test_failures_are_recorded_without_stopping(self):
        lc = _make_multiband_lightcurve()

        def runner(candidate_lc, fit_kwargs, candidate):
            if fit_kwargs["model"] == "2D":
                raise RuntimeError("synthetic failure")
            candidate_lc.consensus_diagnostics = {"consensus_success": True}
            return None

        report = lc.run_period_independent_wavelength_model_kernel_configs(
            model_kernel_config_report=_model_kernel_config_report(), fit_runner=runner
        )

        self.assertEqual(report["n_passed"], 1)
        self.assertEqual(report["n_failed"], 1)
        failed = [o for o in report["outcomes"] if o["status"] == "failed"][0]
        self.assertEqual(failed["model"], "2D")
        self.assertEqual(failed["exception_type"], "RuntimeError")
        self.assertIn("synthetic failure", failed["exception_message"])

    def test_fit_success_alias_tracks_pass_and_failure_status(self):
        lc = _make_multiband_lightcurve()

        def runner(candidate_lc, fit_kwargs, candidate):
            if fit_kwargs["model"] == "2D":
                raise RuntimeError("synthetic failure")
            candidate_lc.consensus_diagnostics = {"consensus_success": True}
            return None

        report = lc.run_period_independent_wavelength_model_kernel_configs(
            model_kernel_config_report=_model_kernel_config_report(), fit_runner=runner
        )

        passed = [o for o in report["outcomes"] if o["status"] == "passed"][0]
        failed = [o for o in report["outcomes"] if o["status"] == "failed"][0]

        self.assertTrue(passed["fit_success"])
        self.assertFalse(passed["fit_failed"])
        self.assertFalse(failed["fit_success"])
        self.assertTrue(failed["fit_failed"])

    def test_stop_on_error_reraises(self):
        lc = _make_multiband_lightcurve()

        def runner(candidate_lc, fit_kwargs, candidate):
            raise RuntimeError("stop here")

        with self.assertRaisesRegex(RuntimeError, "stop here"):
            lc.run_period_independent_wavelength_model_kernel_configs(
                model_kernel_config_report=_model_kernel_config_report(),
                fit_runner=runner,
                stop_on_error=True,
            )

    def test_model_kernel_config_limit_limits_attempts(self):
        lc = _make_multiband_lightcurve()
        seen = []

        def runner(candidate_lc, fit_kwargs, candidate):
            seen.append(fit_kwargs["model"])
            candidate_lc.consensus_diagnostics = {"consensus_success": True}
            return None

        report = lc.run_period_independent_wavelength_model_kernel_configs(
            model_kernel_config_report=_model_kernel_config_report(),
            fit_runner=runner,
            model_kernel_config_limit=1,
        )

        self.assertEqual(seen, ["2DDustMean"])
        self.assertEqual(report["n_attempted"], 1)

    def test_rejects_non_model_kernel_config_report(self):
        lc = _make_multiband_lightcurve()

        with self.assertRaisesRegex(ValueError, "model/kernel-config report"):
            lc.run_period_independent_wavelength_model_kernel_configs(
                model_kernel_config_report={"kind": "period_independent_wavelength_parameter_plan"}
            )

    def test_model_kernel_config_limit_must_be_positive(self):
        lc = _make_multiband_lightcurve()

        with self.assertRaisesRegex(ValueError, "model_kernel_config_limit"):
            lc.run_period_independent_wavelength_model_kernel_configs(
                model_kernel_config_report=_model_kernel_config_report(),
                model_kernel_config_limit=0,
            )

    def test_original_lightcurve_is_not_mutated_by_default(self):
        lc = _make_multiband_lightcurve()
        lc.consensus_diagnostics = {"sentinel": True}
        lc.gp_model = "original"

        def runner(candidate_lc, fit_kwargs, candidate):
            candidate_lc.consensus_diagnostics = {"consensus_success": True}
            candidate_lc.gp_model = fit_kwargs["model"]
            return None

        report = lc.run_period_independent_wavelength_model_kernel_configs(
            model_kernel_config_report=_model_kernel_config_report(), fit_runner=runner
        )

        self.assertEqual(lc.consensus_diagnostics, {"sentinel": True})
        self.assertEqual(lc.gp_model, "original")
        self.assertTrue(report["model_kernel_config_state_isolated"])

    def test_can_build_model_kernel_config_report_when_not_supplied(self):
        lc = _make_multiband_lightcurve()
        seen = []

        def runner(candidate_lc, fit_kwargs, candidate):
            seen.append(fit_kwargs["model"])
            candidate_lc.consensus_diagnostics = {"consensus_success": True}
            return None

        report = lc.run_period_independent_wavelength_model_kernel_configs(
            fit_runner=runner,
            model_kernel_config_limit=1,
            include_2d_baseline=False,
            base_fit_kwargs={"training_iter": 2},
        )

        self.assertEqual(report["n_attempted"], 1)
        self.assertEqual(len(seen), 1)
        self.assertIn(seen[0], {"2DDustMean", "2DPowerLawMean", "2DWavelengthDependent"})


if __name__ == "__main__":
    unittest.main()
