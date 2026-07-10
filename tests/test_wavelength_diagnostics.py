"""Tests for pre-fit wavelength-dependence diagnostics."""

import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import diagnose_wavelength_dependence_prefit


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

        lags = [row["fixed_frequency_diagnostics"]["lag"] for row in report["band_table"]]
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

        self.assertAlmostEqual(via_period["fixed_frequency"], via_frequency["fixed_frequency"])
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


if __name__ == "__main__":
    unittest.main()
