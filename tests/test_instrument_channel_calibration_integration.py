"""Integration contracts for shared-wavelength calibration metadata."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

import numpy as np

from pgmuvi.wavelength_estimation import (
    build_wavelength_estimation_context,
    build_wavelength_mean_estimation_context,
)
from pgmuvi.wavelength_validation_real_lpv import (
    build_representative_lpv_source_summary,
)


class TestCalibrationAssessmentIntegration(unittest.TestCase):
    @staticmethod
    def _shared_rows():
        wavelengths = np.asarray(
            [0.5] * 6 + [1.0] * 3 + [2.0] * 3 + [4.0] * 3,
            dtype=float,
        )
        channels = np.asarray(
            ["instrument-r-0"] * 3
            + ["instrument-r-1"] * 3
            + ["channel-j"] * 3
            + ["channel-h"] * 3
            + ["channel-k"] * 3,
            dtype=str,
        )
        fluxes = np.asarray(
            [
                1.0,
                1.1,
                0.9,
                1.4,
                1.5,
                1.3,
                2.0,
                2.1,
                1.9,
                3.0,
                3.1,
                2.9,
                5.0,
                5.1,
                4.9,
            ],
            dtype=float,
        )
        return wavelengths, channels, fluxes

    def test_wavelength_context_exposes_canonical_assessment(self):
        wavelengths, channels, fluxes = self._shared_rows()

        diagnostics, _ = build_wavelength_estimation_context(
            wavelengths,
            fluxes,
            observational_channel_labels=channels,
            min_points_per_observational_channel=3,
        )

        assessment = diagnostics.metadata[
            "instrument_channel_calibration"
        ]
        self.assertEqual(
            assessment["status"],
            "required_not_implemented",
        )
        self.assertTrue(assessment["required"])
        self.assertFalse(assessment["calibration_applied"])
        self.assertEqual(
            assessment["shared_wavelength_groups"][0][
                "observational_channels"
            ],
            ["instrument-r-0", "instrument-r-1"],
        )
        json.dumps(assessment, allow_nan=False)

    def test_mean_context_retains_current_exclusion_policy(self):
        wavelengths, channels, fluxes = self._shared_rows()
        model_wavelengths = wavelengths / 5.0

        diagnostics = build_wavelength_mean_estimation_context(
            wavelengths,
            model_wavelengths,
            fluxes,
            observational_channel_labels=channels,
            min_points_per_observational_channel=3,
        )

        self.assertEqual(diagnostics.raw_wavelengths, (1.0, 2.0, 4.0))
        self.assertNotIn(0.5, diagnostics.raw_wavelengths)

        assessment = diagnostics.metadata[
            "instrument_channel_calibration"
        ]
        self.assertEqual(
            assessment["policy"],
            "preserve_channels_without_calibration",
        )
        self.assertFalse(assessment["implemented"])

    def test_real_lpv_summary_uses_the_same_contract(self):
        wavelengths, channels, fluxes = self._shared_rows()
        times = np.arange(wavelengths.size, dtype=float)
        lightcurve = SimpleNamespace(
            _xdata_raw=np.column_stack([times, wavelengths]),
            _ydata_raw=fluxes,
            band=channels,
            observational_channel_labels=channels,
        )

        summary = build_representative_lpv_source_summary(lightcurve)

        self.assertEqual(
            summary["instrument_calibration_status"],
            "not_implemented",
        )
        self.assertTrue(summary["instrument_calibration_tbd"])
        self.assertEqual(
            summary["shared_wavelength_policy"],
            "preserve_channels_without_calibration",
        )
        self.assertEqual(
            summary["instrument_channel_calibration"]["status"],
            "required_not_implemented",
        )
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
