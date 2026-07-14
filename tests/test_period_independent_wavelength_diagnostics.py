"""Tests for period-independent wavelength diagnostics."""

import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    diagnose_period_independent_wavelength_structure,
)


def _make_multiband_lightcurve(
    *,
    n_per_band=80,
    wavelengths=(1.0, 2.0, 4.0),
    medians=(10.0, 20.0, 40.0),
    amplitudes=(0.2, 0.4, 0.8),
    yerr_value=0.05,
    include_yerr=True,
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
        yerrs.append(np.full_like(t, yerr_value))

    x = np.column_stack([np.concatenate(times), np.concatenate(wls)])
    y = np.concatenate(ys)
    yerr = np.concatenate(yerrs) if include_yerr else None

    return Lightcurve(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        yerr=torch.as_tensor(yerr, dtype=torch.float64) if yerr is not None else None,
        max_samples=None,
    )


class TestPeriodIndependentWavelengthDiagnostics(unittest.TestCase):
    def test_report_is_explicitly_period_independent(self):
        lc = _make_multiband_lightcurve()

        report = diagnose_period_independent_wavelength_structure(lc)

        self.assertEqual(report["kind"], "period_independent_wavelength_diagnostics")
        self.assertEqual(report["stage"], "prefit_period_independent")
        self.assertTrue(report["is_period_independent"])
        self.assertFalse(report["summary"]["uses_temporal_consensus"])
        self.assertFalse(report["summary"]["uses_period_or_frequency"])
        self.assertNotIn("period", report)
        self.assertNotIn("frequency", report)

    def test_method_matches_module_function(self):
        lc = _make_multiband_lightcurve()

        via_method = lc.diagnose_period_independent_wavelength_structure()
        via_function = diagnose_period_independent_wavelength_structure(lc)

        self.assertEqual(via_method, via_function)

    def test_per_band_quantile_amplitude_tracks_wavelength_trend(self):
        lc = _make_multiband_lightcurve(amplitudes=(0.2, 0.4, 0.8))

        report = lc.diagnose_period_independent_wavelength_structure()
        rows = report["band_table"]
        amps = [
            row["period_independent_flux_summary"]["raw_half_amplitude_q05_q95"]
            for row in rows
        ]
        wide_amps = [
            row["period_independent_flux_summary"]["raw_half_amplitude_q02_5_q97_5"]
            for row in rows
        ]

        self.assertEqual(report["summary"]["n_bands"], 3)
        self.assertEqual(report["summary"]["n_usable_bands"], 3)
        self.assertGreater(amps[1], amps[0])
        self.assertGreater(amps[2], amps[1])
        self.assertGreater(wide_amps[0], amps[0])
        self.assertGreater(wide_amps[1], amps[1])
        self.assertGreater(wide_amps[2], amps[2])
        self.assertGreater(
            report["summary"]["raw_half_amplitude_q02_5_q97_5_ratio_max_to_min"],
            3.0,
        )
        self.assertGreater(
            report["summary"]["raw_half_amplitude_q05_q95_ratio_max_to_min"],
            3.0,
        )
        self.assertEqual(
            report["summary"]["raw_half_amplitude_q02_5_q97_5_monotonicity_class"],
            "increasing",
        )
        self.assertEqual(
            report["summary"]["raw_half_amplitude_q05_q95_monotonicity_class"],
            "increasing",
        )

    def test_noise_corrected_summaries_do_not_exceed_raw_scale(self):
        lc = _make_multiband_lightcurve(yerr_value=0.05)

        report = lc.diagnose_period_independent_wavelength_structure()
        for row in report["band_table"]:
            flux = row["period_independent_flux_summary"]
            self.assertIsNotNone(flux["noise_corrected_robust_scatter"])
            self.assertLessEqual(
                flux["noise_corrected_robust_scatter"],
                flux["robust_scatter"],
            )
            self.assertLessEqual(
                flux["noise_corrected_half_amplitude_q02_5_q97_5"],
                flux["raw_half_amplitude_q02_5_q97_5"],
            )
            self.assertLessEqual(
                flux["noise_corrected_half_amplitude_q05_q95"],
                flux["raw_half_amplitude_q05_q95"],
            )

    def test_no_yerr_leaves_noise_corrected_fields_unavailable(self):
        lc = _make_multiband_lightcurve(include_yerr=False)

        report = lc.diagnose_period_independent_wavelength_structure()

        self.assertFalse(report["summary"]["has_yerr"])
        self.assertTrue(any("No yerr" in warning for warning in report["warnings"]))
        for row in report["band_table"]:
            flux = row["period_independent_flux_summary"]
            self.assertIsNone(flux["median_yerr"])
            self.assertIsNone(flux["noise_corrected_robust_scatter"])

    def test_does_not_mutate_consensus_diagnostics(self):
        lc = _make_multiband_lightcurve()
        lc.consensus_diagnostics = {"sentinel": True}

        lc.diagnose_period_independent_wavelength_structure()

        self.assertEqual(lc.consensus_diagnostics, {"sentinel": True})

    def test_single_band_warns_but_returns_report(self):
        lc = _make_multiband_lightcurve(
            wavelengths=(2.0,),
            medians=(20.0,),
            amplitudes=(0.5,),
        )

        report = lc.diagnose_period_independent_wavelength_structure()

        self.assertEqual(report["summary"]["n_bands"], 1)
        self.assertEqual(report["summary"]["n_usable_bands"], 1)
        self.assertTrue(any("Only one wavelength" in w for w in report["warnings"]))

    def test_raises_for_1d_lightcurve(self):
        t = torch.linspace(0.0, 10.0, 20, dtype=torch.float64)
        y = torch.sin(t)
        lc = Lightcurve(t, y, yerr=torch.full_like(y, 0.1), max_samples=None)

        with self.assertRaisesRegex(ValueError, "requires 2-D"):
            lc.diagnose_period_independent_wavelength_structure()


if __name__ == "__main__":
    unittest.main()
