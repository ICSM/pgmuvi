"""Contracts for observational-channel and physical-wavelength terminology."""

from __future__ import annotations

import unittest

import numpy as np

from pgmuvi.parameter_context import (
    BandDiagnostics,
    ParameterEstimationContext,
    WavelengthEstimationDiagnostics,
)
from pgmuvi.wavelength_estimation import (
    build_wavelength_estimation_context,
)


class TestObservationalChannelCompatibility(unittest.TestCase):
    """Protect new terminology without breaking the legacy public API."""

    def test_band_diagnostics_exposes_observational_channel_alias(self):
        diagnostics = BandDiagnostics(
            band="KELT/OSN_Johnson.Cousins_R3_0",
            wavelength=0.6561154962791801,
            n_points=10,
        )

        self.assertEqual(
            diagnostics.observational_channel,
            diagnostics.band,
        )

    def test_parameter_context_exposes_observational_channel_accessors(self):
        diagnostics = BandDiagnostics(
            band="channel-a",
            wavelength=1.0,
            n_points=5,
        )
        context = ParameterEstimationContext(
            is_multiband=True,
            band_diagnostics={"channel-a": diagnostics},
        )

        self.assertEqual(
            context.observational_channels(),
            ["channel-a"],
        )
        self.assertIs(
            context.get_observational_channel("channel-a"),
            diagnostics,
        )
        self.assertIs(
            context.observational_channel_diagnostics,
            context.band_diagnostics,
        )

    def test_wavelength_summary_exposes_channel_count_aliases(self):
        diagnostics = WavelengthEstimationDiagnostics(
            n_distinct_wavelengths=1,
            n_usable_bands=2,
            excluded_bands=("channel-c",),
        )

        self.assertEqual(
            diagnostics.n_usable_observational_channels,
            2,
        )
        self.assertEqual(
            diagnostics.excluded_observational_channels,
            ("channel-c",),
        )

    def test_builder_accepts_observational_channel_labels(self):
        wavelengths = np.asarray([1.0, 1.0, 2.0, 2.0])
        fluxes = np.asarray([1.0, 1.1, 2.0, 2.1])
        labels = np.asarray(
            ["channel-a", "channel-a", "channel-b", "channel-b"]
        )

        summary, channel_diagnostics = build_wavelength_estimation_context(
            wavelengths=wavelengths,
            fluxes=fluxes,
            observational_channel_labels=labels,
            min_points_per_observational_channel=2,
        )

        self.assertEqual(summary.n_usable_observational_channels, 2)
        self.assertEqual(summary.n_distinct_wavelengths, 2)
        self.assertEqual(
            set(channel_diagnostics),
            {"channel-a", "channel-b"},
        )

    def test_legacy_band_arguments_remain_supported(self):
        wavelengths = np.asarray([1.0, 1.0, 2.0, 2.0])
        fluxes = np.asarray([1.0, 1.1, 2.0, 2.1])
        labels = np.asarray(["a", "a", "b", "b"])

        summary, diagnostics = build_wavelength_estimation_context(
            wavelengths=wavelengths,
            fluxes=fluxes,
            band_labels=labels,
            min_points_per_band=2,
        )

        self.assertEqual(summary.n_usable_observational_channels, 2)
        self.assertEqual(set(diagnostics), {"a", "b"})

    def test_new_and_legacy_label_arguments_cannot_both_be_given(self):
        values = np.asarray([1.0, 1.0])

        with self.assertRaisesRegex(
            ValueError,
            "observational_channel_labels.*band_labels",
        ):
            build_wavelength_estimation_context(
                wavelengths=values,
                fluxes=values,
                observational_channel_labels=np.asarray(["a", "a"]),
                band_labels=np.asarray(["a", "a"]),
            )

    def test_shared_wavelength_channels_are_reported_separately(self):
        wavelengths = np.asarray([1.0, 1.0, 1.0, 1.0])
        fluxes = np.asarray([1.0, 1.1, 1.2, 1.3])
        labels = np.asarray(
            ["telescope-a", "telescope-a", "telescope-b", "telescope-b"]
        )

        summary, diagnostics = build_wavelength_estimation_context(
            wavelengths=wavelengths,
            fluxes=fluxes,
            observational_channel_labels=labels,
            min_points_per_observational_channel=2,
        )

        self.assertEqual(summary.n_usable_observational_channels, 2)
        self.assertEqual(summary.n_distinct_wavelengths, 1)
        self.assertEqual(
            set(diagnostics),
            {"telescope-a", "telescope-b"},
        )
        self.assertTrue(
            summary.metadata[
                "multiple_observational_channels_per_wavelength"
            ]
        )
        self.assertEqual(
            summary.metadata["instrument_calibration_status"],
            "not_implemented",
        )
        self.assertTrue(
            summary.metadata["instrument_calibration_tbd"]
        )


if __name__ == "__main__":
    unittest.main()
