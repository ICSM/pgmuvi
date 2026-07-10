"""Tests for pre-fit wavelength-dependence diagnostics."""

import unittest
from unittest import mock

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    compare_wavelength_candidate_models,
    compute_wavelength_residual_diagnostics,
    diagnose_wavelength_dependence_prefit,
    interpret_wavelength_model_comparison,
)


def _make_multiband_lightcurve(
    *,
    n_per_band=40,
    wavelengths=(1.0, 2.0, 4.0),
    amplitudes=(0.2, 0.3, 0.5),
    period=30.0,
    lags=None,
    yerr_value=0.03,
    include_yerr=True,
    band_labels=None,
):
    times = []
    wls = []
    ys = []
    yerrs = []
    labels = []
    t = np.linspace(0.0, 90.0, n_per_band)
    if lags is None:
        lags = tuple(0.0 for _ in wavelengths)
    for i, (wl, amp, lag) in enumerate(
        zip(wavelengths, amplitudes, lags, strict=True)
    ):
        y = 1.0 + amp * np.cos(2.0 * np.pi * (t - lag) / period)
        times.append(t)
        wls.append(np.full_like(t, wl))
        ys.append(y)
        yerrs.append(np.full_like(t, yerr_value))
        if band_labels is not None:
            labels.extend([band_labels[i]] * n_per_band)

    x = np.column_stack([np.concatenate(times), np.concatenate(wls)])
    y = np.concatenate(ys)
    yerr = np.concatenate(yerrs) if include_yerr else None
    band = np.asarray(labels, dtype=np.str_) if band_labels is not None else None

    return Lightcurve(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        yerr=torch.as_tensor(yerr, dtype=torch.float64) if yerr is not None else None,
        band=band,
        max_samples=None,
    )


class _FakeFittedModel:
    pass


class _FakeLikelihood:
    @property
    def noise(self):
        return np.asarray([0.02])

    def named_parameters(self):
        return [("second_noise_covar.raw_noise", np.asarray([0.03]))]


class _FakeLightcurveForComparison:
    def __init__(self, *, fail_models=None, prediction_offsets=None):
        self.fail_models = dict(fail_models or {})
        self.prediction_offsets = dict(prediction_offsets or {})
        self.calls = []
        self.fit_history = []
        self.model = None
        self.likelihood = None
        t = np.tile(np.linspace(0.0, 20.0, 8), 2)
        wl = np.repeat([1.0, 2.0], 8)
        self._xdata_raw = torch.as_tensor(
            np.column_stack([t, wl]), dtype=torch.float64
        )
        self._xdata_transformed = self._xdata_raw
        self._ydata_transformed = torch.as_tensor(
            1.0 + 0.2 * np.cos(2.0 * np.pi * t / 10.0), dtype=torch.float64
        )
        self.band = np.asarray(["A"] * 8 + ["B"] * 8, dtype=np.str_)
        self._prediction_mean = self._ydata_transformed.detach().cpu().numpy()
        self._prediction_variance = np.full(self._prediction_mean.shape, 0.01)

    def training_predictions_for_wavelength_diagnostics(self):
        return {
            "mean": self._prediction_mean,
            "variance": self._prediction_variance,
        }

    def fit(self, **kwargs):
        self.calls.append(dict(kwargs))
        model = kwargs.get("model")
        if model in self.fail_models:
            exc = self.fail_models[model]
            self.fit_history.append(
                {
                    "model_class": model,
                    "success": False,
                    "failed": True,
                    "exception_type": exc.__class__.__name__,
                    "exception_message": str(exc),
                }
            )
            raise exc
        offset = float(self.prediction_offsets.get(model, 0.0))
        self._prediction_mean = (
            self._ydata_transformed.detach().cpu().numpy() + offset
        )
        self.model = _FakeFittedModel()
        self.likelihood = _FakeLikelihood()
        self.fit_history.append(
            {
                "model_class": model,
                "fit_strategy": kwargs.get("fit_strategy"),
                "success": True,
                "failed": False,
                "training_iter": kwargs.get("training_iter"),
            }
        )
        return {"model": model}

    def get_fit_history(self):
        return [dict(item) for item in self.fit_history]

    def clear_fit_history(self):
        self.fit_history.clear()


class TestDiagnoseWavelengthDependencePrefit(unittest.TestCase):
    def test_report_contains_one_row_per_wavelength(self):
        lc = _make_multiband_lightcurve(band_labels=["J", "H", "K"])

        report = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10, "max_gap_fraction": 0.2},
            variability_kwargs={"min_points": 10, "fvar_min": 0.01},
        )

        self.assertEqual(report["kind"], "wavelength_dependence_prefit_diagnostics")
        self.assertEqual(report["stage"], "prefit")
        self.assertEqual(len(report["band_table"]), 3)
        self.assertEqual(report["summary"]["n_bands"], 3)
        self.assertEqual(report["summary"]["n_sampling_pass"], 3)
        self.assertEqual(report["summary"]["n_variable"], 3)
        self.assertEqual(report["summary"]["n_usable_for_wavelength_diagnostics"], 3)

        first = report["band_table"][0]
        self.assertEqual(first["band_labels"], ["J"])
        self.assertIn("sampling_metrics", first)
        self.assertIn("variability_metrics", first)
        self.assertIn("flux_summary", first)
        self.assertNotIn("fixed_frequency_diagnostics", first)
        self.assertTrue(first["sampling_pass"])
        self.assertTrue(first["variable"])
        self.assertTrue(first["usable_for_wavelength_diagnostics"])
        self.assertGreater(first["flux_summary"]["robust_amplitude_5_95"], 0.0)

    def test_module_function_matches_lightcurve_method(self):
        lc = _make_multiband_lightcurve()

        via_method = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10, "fvar_min": 0.01},
            period=30.0,
        )
        via_function = diagnose_wavelength_dependence_prefit(
            lc,
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10, "fvar_min": 0.01},
            period=30.0,
        )

        self.assertEqual(via_method["summary"], via_function["summary"])
        self.assertEqual(via_method["band_table"], via_function["band_table"])
        self.assertEqual(
            via_method["amplitude_phase_summary"],
            via_function["amplitude_phase_summary"],
        )

    def test_single_band_report_refuses_wavelength_claim(self):
        lc = _make_multiband_lightcurve(
            wavelengths=(2.0,),
            amplitudes=(0.3,),
            band_labels=["H"],
        )

        report = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10, "fvar_min": 0.01},
        )

        self.assertEqual(report["summary"]["n_bands"], 1)
        self.assertEqual(report["summary"]["n_usable_for_wavelength_diagnostics"], 1)
        self.assertTrue(
            any("Only one wavelength" in warning for warning in report["warnings"])
        )
        self.assertTrue(
            any("Fewer than two bands" in warning for warning in report["warnings"])
        )

    def test_no_yerr_leaves_variability_unavailable(self):
        lc = _make_multiband_lightcurve(include_yerr=False)

        report = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10},
        )

        self.assertFalse(report["summary"]["has_yerr"])
        self.assertEqual(report["summary"]["n_variable"], 0)
        self.assertTrue(any("No yerr" in warning for warning in report["warnings"]))
        for row in report["band_table"]:
            self.assertFalse(row["variability_available"])
            self.assertIsNone(row["variable"])
            self.assertIn("UNAVAILABLE", row["variability_decision"])

    def test_fixed_period_amplitudes_recover_wavelength_trend(self):
        lc = _make_multiband_lightcurve(
            n_per_band=80,
            amplitudes=(0.1, 0.2, 0.4),
            yerr_value=0.01,
        )

        report = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10, "fvar_min": 0.001},
            period=30.0,
        )

        rows = report["band_table"]
        recovered = [row["fixed_frequency_diagnostics"]["amplitude"] for row in rows]
        np.testing.assert_allclose(recovered, [0.1, 0.2, 0.4], rtol=0.03, atol=0.01)
        self.assertEqual(
            report["amplitude_phase_summary"]["n_bands_with_fixed_frequency_fit"], 3
        )
        self.assertGreater(
            report["amplitude_phase_summary"]["amplitude_ratio_max_to_min"], 3.5
        )
        self.assertAlmostEqual(
            report["amplitude_phase_summary"]["amplitude_loglog_slope"],
            1.0,
            delta=0.08,
        )

    def test_fixed_period_phase_lags_recover_band_lag_order(self):
        lc = _make_multiband_lightcurve(
            n_per_band=90,
            amplitudes=(0.4, 0.4, 0.4),
            lags=(0.0, 2.0, 4.0),
            yerr_value=0.01,
        )

        report = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10, "fvar_min": 0.001},
            frequency=1.0 / 30.0,
            amplitude_phase_kwargs={"reference_time": 0.0},
        )

        lags = [
            row["fixed_frequency_diagnostics"]["lag"]
            for row in report["band_table"]
        ]
        np.testing.assert_allclose(lags, [0.0, 2.0, 4.0], atol=0.05)
        self.assertGreater(report["amplitude_phase_summary"]["lag_span"], 3.9)

    def test_period_argument_matches_frequency_argument(self):
        lc = _make_multiband_lightcurve(n_per_band=80, amplitudes=(0.2, 0.2, 0.2))

        via_period = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            period=30.0,
        )
        via_frequency = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            frequency=1.0 / 30.0,
        )

        self.assertAlmostEqual(
            via_period["fixed_frequency"],
            via_frequency["fixed_frequency"],
        )
        self.assertEqual(
            via_period["amplitude_phase_summary"],
            via_frequency["amplitude_phase_summary"],
        )

    def test_frequency_period_conflict_raises(self):
        lc = _make_multiband_lightcurve()

        with self.assertRaisesRegex(ValueError, "Specify only one"):
            lc.diagnose_wavelength_dependence(frequency=1.0 / 30.0, period=30.0)

    def test_raises_for_1d_lightcurve(self):
        t = torch.linspace(0.0, 10.0, 20)
        y = torch.sin(t)
        lc = Lightcurve(t, y, yerr=torch.full_like(y, 0.1), max_samples=None)

        with self.assertRaisesRegex(ValueError, "requires 2-D multiband data"):
            lc.diagnose_wavelength_dependence()


    def test_classification_recommends_achromatic_candidate_for_constant_amp(self):
        lc = _make_multiband_lightcurve(
            n_per_band=80,
            amplitudes=(0.25, 0.26, 0.24),
            yerr_value=0.01,
        )

        report = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10, "fvar_min": 0.001},
            period=30.0,
        )

        classification = report["classification"]
        self.assertTrue(classification["available"])
        self.assertEqual(
            classification["amplitude_class"],
            "consistent_with_constant_amplitude",
        )
        self.assertEqual(
            classification["primary_class"],
            "achromatic_shared_variability_candidate",
        )
        models = [item["model"] for item in report["recommended_candidate_models"]]
        self.assertIn("2D", models)
        self.assertIn("2DAchromatic", models)

    def test_classification_recommends_power_law_candidate_for_smooth_trend(self):
        lc = _make_multiband_lightcurve(
            n_per_band=80,
            wavelengths=(1.0, 2.0, 4.0),
            amplitudes=(0.1, 0.2, 0.4),
            yerr_value=0.01,
        )

        report = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10, "fvar_min": 0.001},
            period=30.0,
        )

        classification = report["classification"]
        self.assertEqual(
            classification["amplitude_class"],
            "power_law_like_amplitude_trend",
        )
        self.assertEqual(
            classification["primary_class"],
            "wavelength_modulated_shared_variability_candidate",
        )
        models = [item["model"] for item in report["recommended_candidate_models"]]
        self.assertIn("2DWavelengthDependent", models)
        self.assertIn("2DPowerLawMean", models)

    def test_classification_flags_possible_phase_lag_without_specific_lag_model(self):
        lc = _make_multiband_lightcurve(
            n_per_band=90,
            amplitudes=(0.4, 0.4, 0.4),
            lags=(0.0, 2.0, 4.0),
            yerr_value=0.01,
        )

        report = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10, "fvar_min": 0.001},
            frequency=1.0 / 30.0,
            amplitude_phase_kwargs={"reference_time": 0.0},
        )

        classification = report["classification"]
        self.assertEqual(
            classification["phase_lag_class"],
            "possible_monotonic_wavelength_lag",
        )
        self.assertEqual(
            classification["primary_class"],
            "possible_wavelength_dependent_lag",
        )
        self.assertTrue(
            any(
                "do not explicitly parameterize" in warning
                for warning in classification["warnings"]
            )
        )

    def test_classification_without_fixed_frequency_requests_consensus_step(self):
        lc = _make_multiband_lightcurve()

        report = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10, "fvar_min": 0.001},
        )

        classification = report["classification"]
        self.assertTrue(classification["available"])
        self.assertEqual(classification["primary_class"], "prefit_table_only")
        names = [item["name"] for item in report["recommended_candidate_models"]]
        self.assertIn("robust_2d_consensus_baseline", names)
        self.assertIn("next_step_consensus_period_diagnostics", names)

    def test_classification_suppressed_for_single_band(self):
        lc = _make_multiband_lightcurve(
            wavelengths=(2.0,),
            amplitudes=(0.3,),
            band_labels=["H"],
        )

        report = lc.diagnose_wavelength_dependence(
            sampling_kwargs={"min_points": 10},
            variability_kwargs={"min_points": 10, "fvar_min": 0.001},
            period=30.0,
        )

        classification = report["classification"]
        self.assertFalse(classification["available"])
        self.assertEqual(classification["primary_class"], "insufficient_data")
        self.assertEqual(report["recommended_candidate_models"], [])

    def test_compare_wavelength_models_runs_fit_candidates_and_skips_next_steps(self):
        fake_lc = _FakeLightcurveForComparison()
        candidates = [
            {
                "name": "baseline",
                "model": "2D",
                "fit_strategy": "consensus",
                "priority": "baseline",
                "reason": "baseline fit",
                "options": {"learn_additional_noise": True},
            },
            {
                "name": "next_step_consensus_period_diagnostics",
                "model": None,
                "fit_strategy": "consensus",
                "priority": "next_step",
                "reason": "not a fit candidate",
            },
        ]

        report = compare_wavelength_candidate_models(
            fake_lc,
            candidates=candidates,
            base_fit_kwargs={"training_iter": 0, "miniter": 0},
            copy_lightcurve=False,
        )

        self.assertEqual(report["kind"], "wavelength_model_comparison")
        self.assertEqual(report["summary"]["n_candidates"], 2)
        self.assertEqual(report["summary"]["n_fit_candidates"], 1)
        self.assertEqual(report["summary"]["n_successful"], 1)
        self.assertEqual(report["summary"]["n_skipped"], 1)
        self.assertEqual(fake_lc.calls[0]["model"], "2D")
        self.assertEqual(fake_lc.calls[0]["fit_strategy"], "consensus")
        self.assertTrue(fake_lc.calls[0]["learn_additional_noise"])
        self.assertEqual(fake_lc.calls[0]["training_iter"], 0)

        success = report["results"][0]
        self.assertTrue(success["success"])
        self.assertEqual(success["resolved_model_class"], "_FakeFittedModel")
        self.assertTrue(success["likelihood_noise_summary"]["available"])

        skipped = report["results"][1]
        self.assertTrue(skipped["skipped"])
        self.assertIn("no model", skipped["skip_reason"])

    def test_compare_wavelength_models_records_failures_without_stopping(self):
        fake_lc = _FakeLightcurveForComparison(
            fail_models={"2DAchromatic": FloatingPointError("non-finite loss")}
        )

        report = compare_wavelength_candidate_models(
            fake_lc,
            candidates=["2DAchromatic", "2DWavelengthDependent"],
            base_fit_kwargs={"training_iter": 0},
            copy_lightcurve=False,
        )

        self.assertEqual(report["summary"]["n_fit_candidates"], 2)
        self.assertEqual(report["summary"]["n_failed"], 1)
        self.assertEqual(report["summary"]["n_successful"], 1)
        failed = report["results"][0]
        self.assertTrue(failed["failed"])
        self.assertEqual(failed["exception_type"], "FloatingPointError")
        self.assertEqual(failed["failure_category"], "numerical_failure")
        self.assertEqual(report["results"][1]["status"], "success")

    def test_compare_wavelength_models_uses_diagnostic_recommendations(self):
        fake_lc = _FakeLightcurveForComparison()
        diagnostic_report = {
            "classification": {"primary_class": "prefit_table_only"},
            "recommended_candidate_models": [
                {
                    "name": "baseline",
                    "model": "2D",
                    "fit_strategy": "consensus",
                    "priority": "baseline",
                    "reason": "recommended baseline",
                }
            ],
        }

        report = compare_wavelength_candidate_models(
            fake_lc,
            diagnostic_report=diagnostic_report,
            base_fit_kwargs={"training_iter": 0},
            copy_lightcurve=False,
        )

        self.assertEqual(report["summary"]["n_successful"], 1)
        self.assertEqual(fake_lc.calls[0]["model"], "2D")
        self.assertEqual(
            report["diagnostic_classification"],
            {"primary_class": "prefit_table_only"},
        )

    def test_lightcurve_compare_wavelength_models_delegates_to_module_function(self):
        lc = _make_multiband_lightcurve()
        expected = {"kind": "wavelength_model_comparison"}

        with mock.patch(
            "pgmuvi.wavelength_diagnostics.compare_wavelength_candidate_models",
            return_value=expected,
        ) as mocked:
            report = lc.compare_wavelength_models(
                candidates=["2D"],
                base_fit_kwargs={"training_iter": 0},
                copy_lightcurve=False,
            )

        self.assertIs(report, expected)
        mocked.assert_called_once()
        self.assertIs(mocked.call_args.args[0], lc)
        self.assertEqual(mocked.call_args.kwargs["candidates"], ["2D"])
        self.assertEqual(
            mocked.call_args.kwargs["base_fit_kwargs"],
            {"training_iter": 0},
        )


    def test_residual_diagnostics_score_training_predictions_by_band(self):
        fake_lc = _FakeLightcurveForComparison()
        fake_lc.fit(model="2D")

        report = compute_wavelength_residual_diagnostics(
            fake_lc,
            period=10.0,
            min_points_per_band=3,
        )

        self.assertTrue(report["available"])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["predictive_score"]["available"], True)
        self.assertEqual(len(report["by_band"]), 2)
        self.assertLess(report["overall"]["residual_rms"], 1e-12)
        for row in report["by_band"]:
            self.assertIn("fixed_frequency_residual", row)
            self.assertLess(row["fixed_frequency_residual"]["amplitude"], 1e-12)

    def test_compare_wavelength_models_scores_successful_candidates(self):
        fake_lc = _FakeLightcurveForComparison(
            prediction_offsets={"2D": 0.0, "2DAchromatic": 0.5}
        )

        report = compare_wavelength_candidate_models(
            fake_lc,
            candidates=["2D", "2DAchromatic"],
            base_fit_kwargs={"training_iter": 0},
            residual_diagnostic_kwargs={"period": 10.0},
            copy_lightcurve=False,
        )

        self.assertEqual(report["summary"]["n_successful"], 2)
        self.assertEqual(report["summary"]["n_scored_successful"], 2)
        self.assertEqual(report["summary"]["selection_status"], "scored_predictive")
        self.assertEqual(report["summary"]["best_candidate"]["model"], "2D")
        for result in report["results"]:
            self.assertIn("residual_diagnostics", result)
            self.assertIn("predictive_score", result)
            self.assertTrue(result["predictive_score"]["available"])

    def test_compare_wavelength_models_can_disable_success_scoring(self):
        fake_lc = _FakeLightcurveForComparison()

        report = compare_wavelength_candidate_models(
            fake_lc,
            candidates=["2D"],
            base_fit_kwargs={"training_iter": 0},
            score_successful_fits=False,
            copy_lightcurve=False,
        )

        self.assertEqual(report["summary"]["n_successful"], 1)
        self.assertEqual(report["summary"]["n_scored_successful"], 0)
        self.assertEqual(report["summary"]["selection_status"], "not_scored")
        self.assertNotIn("residual_diagnostics", report["results"][0])

    def test_lightcurve_compare_wavelength_models_passes_scoring_kwargs(self):
        lc = _make_multiband_lightcurve()
        expected = {"kind": "wavelength_model_comparison"}

        with mock.patch(
            "pgmuvi.wavelength_diagnostics.compare_wavelength_candidate_models",
            return_value=expected,
        ) as mocked:
            report = lc.compare_wavelength_models(
                candidates=["2D"],
                residual_diagnostic_kwargs={"period": 30.0},
                score_successful_fits=False,
                copy_lightcurve=False,
            )

        self.assertIs(report, expected)
        mocked.assert_called_once()
        self.assertEqual(
            mocked.call_args.kwargs["residual_diagnostic_kwargs"],
            {"period": 30.0},
        )
        self.assertFalse(mocked.call_args.kwargs["score_successful_fits"])


    def test_interpret_wavelength_model_comparison_flags_tied_scores(self):
        fake_lc = _FakeLightcurveForComparison(
            prediction_offsets={"2D": 0.0, "2DAchromatic": 0.01}
        )

        report = compare_wavelength_candidate_models(
            fake_lc,
            candidates=["2D", "2DAchromatic"],
            base_fit_kwargs={"training_iter": 0},
            residual_diagnostic_kwargs={"period": 10.0},
            interpretation_kwargs={"score_tie_tolerance": 1.0},
            copy_lightcurve=False,
        )

        interpretation = report["interpretation"]
        self.assertTrue(interpretation["available"])
        self.assertEqual(interpretation["decision"], "scores_indistinguishable")
        self.assertEqual(len(interpretation["candidate_rankings"]), 2)
        self.assertEqual(interpretation["candidate_rankings"][0]["model"], "2D")

    def test_interpret_wavelength_model_comparison_flags_poor_residual_quality(self):
        fake_lc = _FakeLightcurveForComparison(prediction_offsets={"2D": 0.5})

        report = compare_wavelength_candidate_models(
            fake_lc,
            candidates=["2D"],
            base_fit_kwargs={"training_iter": 0},
            residual_diagnostic_kwargs={"period": 10.0},
            interpretation_kwargs={
                "standardized_residual_rms_warning": 1.0,
                "poor_coverage_2sigma_min": 0.99,
                "band_standardized_residual_rms_warning": 1.0,
            },
            copy_lightcurve=False,
        )

        flags = report["interpretation"]["quality_flags"]
        flag_names = {flag["flag"] for flag in flags}
        self.assertIn("large_standardized_residual_rms", flag_names)
        self.assertIn("poor_two_sigma_coverage", flag_names)
        self.assertIn("band_specific_residual_mismatch", flag_names)

    def test_interpret_wavelength_model_comparison_handles_unscored_success(self):
        fake_lc = _FakeLightcurveForComparison()
        report = compare_wavelength_candidate_models(
            fake_lc,
            candidates=["2D"],
            base_fit_kwargs={"training_iter": 0},
            score_successful_fits=False,
            copy_lightcurve=False,
        )

        interpretation = interpret_wavelength_model_comparison(report)
        self.assertTrue(interpretation["available"])
        self.assertEqual(interpretation["decision"], "scores_unavailable")

    def test_lightcurve_compare_wavelength_models_passes_interpretation_kwargs(self):
        lc = _make_multiband_lightcurve()
        expected = {"kind": "wavelength_model_comparison"}

        with mock.patch(
            "pgmuvi.wavelength_diagnostics.compare_wavelength_candidate_models",
            return_value=expected,
        ) as mocked:
            report = lc.compare_wavelength_models(
                candidates=["2D"],
                interpretation_kwargs={"score_tie_tolerance": 0.2},
                interpret_results=False,
            )

        self.assertIs(report, expected)
        mocked.assert_called_once()
        self.assertEqual(
            mocked.call_args.kwargs["interpretation_kwargs"],
            {"score_tie_tolerance": 0.2},
        )
        self.assertFalse(mocked.call_args.kwargs["interpret_results"])



if __name__ == "__main__":
    unittest.main()
