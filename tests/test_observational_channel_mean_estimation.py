"""Observational-channel contracts for wavelength-mean estimation."""

from __future__ import annotations

import unittest

import numpy as np

from pgmuvi.wavelength_estimation import (
    build_wavelength_mean_estimation_context,
)


class TestObservationalChannelMeanEstimation(unittest.TestCase):
    @staticmethod
    def _rows():
        raw_wavelengths = []
        model_wavelengths = []
        model_fluxes = []
        observational_channels = []

        channel_rows = [
            # Two independent observational channels at one wavelength.
            ("instrument-r-0", 0.5, 0.0, 1.0),
            ("instrument-r-1", 0.5, 0.0, 1.4),
            # Three unambiguous physical wavelengths.
            ("channel-j", 1.0, 0.2, 2.0),
            ("channel-h", 2.0, 0.4, 3.0),
            ("channel-k", 4.0, 0.8, 5.0),
        ]

        for channel, raw, model, flux in channel_rows:
            for offset in (-0.01, 0.0, 0.01):
                raw_wavelengths.append(raw)
                model_wavelengths.append(model)
                model_fluxes.append(flux + offset)
                observational_channels.append(channel)

        return (
            np.asarray(raw_wavelengths, dtype=float),
            np.asarray(model_wavelengths, dtype=float),
            np.asarray(model_fluxes, dtype=float),
            np.asarray(observational_channels, dtype=str),
        )

    def test_preferred_observational_channel_arguments_are_supported(self):
        raw, model, flux, channels = self._rows()

        diagnostics = build_wavelength_mean_estimation_context(
            raw,
            model,
            flux,
            observational_channel_labels=channels,
            min_points_per_observational_channel=3,
        )

        self.assertTrue(diagnostics.available)

    def test_legacy_band_arguments_remain_supported(self):
        raw, model, flux, channels = self._rows()

        diagnostics = build_wavelength_mean_estimation_context(
            raw,
            model,
            flux,
            band_labels=channels,
            min_points_per_band=3,
        )

        self.assertTrue(diagnostics.available)

    def test_preferred_and_legacy_labels_cannot_both_be_supplied(self):
        raw, model, flux, channels = self._rows()

        with self.assertRaisesRegex(
            ValueError,
            "observational_channel_labels.*band_labels",
        ):
            build_wavelength_mean_estimation_context(
                raw,
                model,
                flux,
                band_labels=channels,
                observational_channel_labels=channels,
            )

    def test_preferred_and_legacy_minimums_cannot_both_be_supplied(self):
        raw, model, flux, channels = self._rows()

        with self.assertRaisesRegex(
            ValueError,
            "min_points_per_observational_channel.*min_points_per_band",
        ):
            build_wavelength_mean_estimation_context(
                raw,
                model,
                flux,
                observational_channel_labels=channels,
                min_points_per_band=3,
                min_points_per_observational_channel=3,
            )

    def test_uncalibrated_shared_wavelength_is_excluded_from_mean_fit(self):
        raw, model, flux, channels = self._rows()

        diagnostics = build_wavelength_mean_estimation_context(
            raw,
            model,
            flux,
            observational_channel_labels=channels,
            min_points_per_observational_channel=3,
        )

        self.assertEqual(
            diagnostics.raw_wavelengths,
            (1.0, 2.0, 4.0),
        )
        self.assertEqual(
            diagnostics.model_wavelengths,
            (0.2, 0.4, 0.8),
        )
        self.assertEqual(
            diagnostics.model_median_fluxes,
            (2.0, 3.0, 5.0),
        )

        metadata = diagnostics.metadata
        self.assertEqual(metadata["n_usable_observational_channels"], 5)
        self.assertEqual(metadata["n_distinct_physical_wavelengths"], 4)
        self.assertEqual(
            metadata["n_physical_wavelengths_used_for_mean_estimation"],
            3,
        )
        self.assertTrue(
            metadata["multiple_observational_channels_per_wavelength"]
        )
        self.assertEqual(
            metadata["instrument_calibration_status"],
            "not_implemented",
        )
        self.assertTrue(metadata["instrument_calibration_tbd"])
        self.assertEqual(
            metadata["shared_wavelength_mean_policy"],
            "exclude_uncalibrated_shared_wavelengths",
        )
        self.assertEqual(
            metadata["observational_channels_by_shared_wavelength"],
            {
                "0.5": [
                    "instrument-r-0",
                    "instrument-r-1",
                ]
            },
        )

        for recommendation in diagnostics.recommendations.values():
            self.assertNotIn(0.5, diagnostics.raw_wavelengths)
            self.assertNotEqual(
                recommendation.get("reason"),
                "instrument_channels_silently_combined",
            )


if __name__ == "__main__":
    unittest.main()
