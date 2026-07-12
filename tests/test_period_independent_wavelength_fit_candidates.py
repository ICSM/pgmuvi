"""Tests for period-independent wavelength fit-candidate configs."""

import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    build_period_independent_wavelength_fit_candidates,
    build_period_independent_wavelength_parameter_plan,
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


class TestPeriodIndependentWavelengthFitCandidates(unittest.TestCase):
    def test_candidate_report_is_advisory_and_does_not_run_fits(self):
        lc = _make_multiband_lightcurve()

        report = lc.build_period_independent_wavelength_fit_candidates()

        self.assertEqual(report["kind"], "period_independent_wavelength_fit_candidates")
        self.assertEqual(report["stage"], "prefit_period_independent_candidate_config")
        self.assertTrue(report["is_period_independent"])
        self.assertFalse(report["uses_temporal_consensus"])
        self.assertFalse(report["uses_period_or_frequency"])
        self.assertFalse(report["applies_to_fit"])
        self.assertTrue(report["advisory_only"])
        self.assertFalse(report["runs_fits"])
        self.assertFalse(report["automatic_constraints_applied"])
        self.assertFalse(report["automatic_initialization_applied"])

    def test_module_function_matches_lightcurve_method(self):
        lc = _make_multiband_lightcurve()

        via_method = lc.build_period_independent_wavelength_fit_candidates()
        via_function = build_period_independent_wavelength_fit_candidates(lc)

        self.assertEqual(via_method, via_function)

    def test_can_reuse_precomputed_parameter_plan(self):
        lc = _make_multiband_lightcurve()
        plan = lc.build_period_independent_wavelength_parameter_plan()

        via_kwarg = lc.build_period_independent_wavelength_fit_candidates(
            parameter_plan=plan
        )
        via_direct = build_period_independent_wavelength_fit_candidates(plan)

        self.assertEqual(via_kwarg, via_direct)
        self.assertEqual(
            via_kwarg["source_plan_kind"],
            "period_independent_wavelength_parameter_plan",
        )

    def test_candidate_order_follows_parameter_plan_ranking(self):
        lc = _make_multiband_lightcurve(medians=(10.0, 40.0, 20.0))

        plan = lc.build_period_independent_wavelength_parameter_plan()
        report = lc.build_period_independent_wavelength_fit_candidates(
            parameter_plan=plan,
            include_2d_baseline=False,
        )

        plan_models = [entry["model"] for entry in plan["ranked_candidates"]]
        candidate_models = [entry["model"] for entry in report["fit_candidates"]]
        self.assertEqual(candidate_models, plan_models)
        self.assertEqual(candidate_models[0], "2DWavelengthDependent")

    def test_lpv_candidates_use_quasi_periodic_consensus_kwargs(self):
        lc = _make_multiband_lightcurve()

        report = lc.build_period_independent_wavelength_fit_candidates(
            include_2d_baseline=False
        )

        for candidate in report["fit_candidates"]:
            kwargs = candidate["fit_kwargs"]
            self.assertEqual(kwargs["model"], candidate["model"])
            self.assertEqual(kwargs["fit_strategy"], "consensus")
            self.assertTrue(kwargs["learn_additional_noise"])
            self.assertEqual(kwargs["time_kernel_type"], "quasi_periodic")

    def test_2d_baseline_is_optional_and_keeps_default_time_kernel(self):
        lc = _make_multiband_lightcurve()

        with_baseline = lc.build_period_independent_wavelength_fit_candidates(
            include_2d_baseline=True
        )
        without_baseline = lc.build_period_independent_wavelength_fit_candidates(
            include_2d_baseline=False
        )

        self.assertIn("2D", [entry["model"] for entry in with_baseline["fit_candidates"]])
        self.assertNotIn("2D", [entry["model"] for entry in without_baseline["fit_candidates"]])
        baseline = [
            entry for entry in with_baseline["fit_candidates"] if entry["model"] == "2D"
        ][0]
        self.assertNotIn("time_kernel_type", baseline["fit_kwargs"])
        self.assertEqual(baseline["recommendation_strength"], "baseline_comparison")

    def test_base_fit_kwargs_are_preserved_without_overriding_model(self):
        lc = _make_multiband_lightcurve()

        report = lc.build_period_independent_wavelength_fit_candidates(
            include_2d_baseline=False,
            base_fit_kwargs={"training_iter": 25, "miniter": 5, "model": "WRONG"},
        )

        for candidate in report["fit_candidates"]:
            self.assertEqual(candidate["fit_kwargs"]["training_iter"], 25)
            self.assertEqual(candidate["fit_kwargs"]["miniter"], 5)
            self.assertEqual(candidate["fit_kwargs"]["model"], candidate["model"])

    def test_parameter_suggestions_are_metadata_not_applied_kwargs(self):
        lc = _make_multiband_lightcurve()

        report = lc.build_period_independent_wavelength_fit_candidates(
            include_2d_baseline=False
        )

        for candidate in report["fit_candidates"]:
            self.assertFalse(candidate["applies_parameter_suggestions"])
            self.assertFalse(candidate["applies_constraints"])
            self.assertIsInstance(candidate["parameter_suggestions"], dict)
            forbidden = set(candidate["parameter_suggestions"].get("initial_values", {}))
            self.assertTrue(forbidden.isdisjoint(candidate["fit_kwargs"]))

    def test_candidate_metadata_aliases_are_explicit(self):
        lc = _make_multiband_lightcurve()

        report = lc.build_period_independent_wavelength_fit_candidates(
            include_2d_baseline=True
        )

        sources = [candidate["source"] for candidate in report["fit_candidates"]]
        self.assertIn("parameter_plan", sources)
        self.assertIn("baseline_comparison", sources)
        for candidate in report["fit_candidates"]:
            self.assertIn("parameter_suggestions_applied", candidate)
            self.assertFalse(candidate["parameter_suggestions_applied"])

    def test_include_models_filters_candidate_list(self):
        lc = _make_multiband_lightcurve()

        report = lc.build_period_independent_wavelength_fit_candidates(
            include_models=["2DDustMean", "2D"],
            include_2d_baseline=True,
        )

        self.assertEqual(
            [entry["model"] for entry in report["fit_candidates"]],
            ["2DDustMean", "2D"],
        )

    def test_candidate_limit_truncates_candidates(self):
        lc = _make_multiband_lightcurve()

        report = lc.build_period_independent_wavelength_fit_candidates(candidate_limit=2)

        self.assertEqual(report["n_candidates"], 2)
        self.assertEqual(len(report["fit_candidates"]), 2)
        self.assertEqual([entry["rank"] for entry in report["fit_candidates"]], [1, 2])

    def test_does_not_mutate_consensus_diagnostics(self):
        lc = _make_multiband_lightcurve()
        lc.consensus_diagnostics = {"sentinel": True}

        lc.build_period_independent_wavelength_fit_candidates()

        self.assertEqual(lc.consensus_diagnostics, {"sentinel": True})

    def test_rejects_non_parameter_plan(self):
        with self.assertRaisesRegex(ValueError, "parameter plan"):
            build_period_independent_wavelength_fit_candidates(
                {"kind": "period_independent_wavelength_diagnostics"}
            )

    def test_candidate_limit_must_be_positive(self):
        lc = _make_multiband_lightcurve()

        with self.assertRaisesRegex(ValueError, "candidate_limit"):
            lc.build_period_independent_wavelength_fit_candidates(candidate_limit=0)


if __name__ == "__main__":
    unittest.main()
