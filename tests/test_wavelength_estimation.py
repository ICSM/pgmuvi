import math
import unittest

import numpy as np

from pgmuvi.wavelength_estimation import (
    WAVELENGTH_ESTIMATION_SCHEMA_VERSION,
    build_wavelength_estimation_context,
)


class TestWavelengthEstimationContext(unittest.TestCase):
    def test_builds_sampling_and_robust_band_summaries(self):
        wavelengths = np.repeat([0.5, 1.0, 2.0, 5.0], 5)
        bands = np.repeat(["g", "r", "j", "m"], 5)
        fluxes = np.concatenate(
            [
                [8.0, 9.0, 10.0, 11.0, 12.0],
                [18.0, 19.0, 20.0, 21.0, 22.0],
                [28.0, 29.0, 30.0, 31.0, 32.0],
                [38.0, 39.0, 40.0, 41.0, 42.0],
            ]
        )
        uncertainties = np.full(fluxes.shape, 0.5)

        diagnostics, band_diagnostics = build_wavelength_estimation_context(
            wavelengths,
            fluxes,
            uncertainties,
            bands,
        )

        self.assertEqual(
            diagnostics.schema_version,
            WAVELENGTH_ESTIMATION_SCHEMA_VERSION,
        )
        self.assertTrue(diagnostics.available)
        self.assertEqual(diagnostics.coordinate_space, "raw_input")
        self.assertEqual(diagnostics.n_observations, 20)
        self.assertEqual(diagnostics.n_distinct_wavelengths, 4)
        self.assertEqual(diagnostics.n_usable_bands, 4)
        self.assertEqual(diagnostics.wavelengths, (0.5, 1.0, 2.0, 5.0))
        self.assertAlmostEqual(diagnostics.wavelength_span, 4.5)
        self.assertEqual(diagnostics.adjacent_spacings, (0.5, 1.0, 3.0))
        self.assertAlmostEqual(diagnostics.minimum_adjacent_spacing, 0.5)
        self.assertAlmostEqual(diagnostics.median_adjacent_spacing, 1.0)
        self.assertAlmostEqual(diagnostics.largest_gap, 3.0)
        self.assertAlmostEqual(
            diagnostics.largest_gap_ratio_to_median_spacing,
            3.0,
        )
        self.assertAlmostEqual(diagnostics.spacing_ratio_max_to_min, 6.0)
        self.assertEqual(diagnostics.coverage_class, "sparse")
        self.assertEqual(
            diagnostics.median_flux_monotonicity_class,
            "non_decreasing",
        )
        self.assertEqual(
            diagnostics.amplitude_monotonicity_class,
            "approximately_constant",
        )
        self.assertAlmostEqual(
            diagnostics.recommended_lengthscale_initial,
            math.sqrt(4.5),
        )
        self.assertEqual(
            diagnostics.recommended_lengthscale_bounds,
            (0.125, 22.5),
        )
        self.assertFalse(diagnostics.metadata["uses_log_flux"])
        self.assertFalse(
            diagnostics.metadata["lengthscale_applied_to_models"]
        )

        self.assertEqual(set(band_diagnostics), {"g", "r", "j", "m"})
        self.assertAlmostEqual(band_diagnostics["g"].median_flux, 10.0)
        self.assertAlmostEqual(band_diagnostics["g"].median_uncertainty, 0.5)
        self.assertAlmostEqual(
            band_diagnostics["g"].raw_half_amplitude_q05_q95,
            1.8,
        )
        self.assertAlmostEqual(
            band_diagnostics["g"].fractional_half_amplitude_q05_q95,
            0.18,
        )
        self.assertTrue(
            band_diagnostics["g"].metadata[
                "usable_for_wavelength_estimation"
            ]
        )

    def test_excludes_underpopulated_band_from_cross_wavelength_context(self):
        wavelengths = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0])
        fluxes = np.asarray([10.0, 11.0, 12.0, 20.0, 21.0])
        bands = np.asarray(["a", "a", "a", "b", "b"])

        diagnostics, band_diagnostics = build_wavelength_estimation_context(
            wavelengths,
            fluxes,
            band_labels=bands,
            min_points_per_band=3,
        )

        self.assertFalse(diagnostics.available)
        self.assertEqual(diagnostics.n_usable_bands, 1)
        self.assertEqual(diagnostics.n_distinct_wavelengths, 1)
        self.assertEqual(diagnostics.coverage_class, "unresolved")
        self.assertEqual(diagnostics.excluded_bands, ("b",))
        self.assertIsNone(diagnostics.recommended_lengthscale_initial)
        self.assertEqual(
            band_diagnostics["b"].metadata["exclusion_reason"],
            "insufficient_finite_flux_points",
        )


    def test_excludes_band_label_with_inconsistent_wavelengths(self):
        diagnostics, band_diagnostics = build_wavelength_estimation_context(
            wavelengths=[1.0, 1.1, 1.0, 2.0, 2.0, 2.0],
            fluxes=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            band_labels=["a", "a", "a", "b", "b", "b"],
        )

        self.assertFalse(diagnostics.available)
        self.assertEqual(diagnostics.excluded_bands, ("a",))
        self.assertEqual(
            band_diagnostics["a"].metadata["exclusion_reason"],
            "inconsistent_wavelength_within_band",
        )
        self.assertFalse(
            band_diagnostics["a"].metadata[
                "wavelength_consistent_within_band"
            ]
        )

    def test_noise_correction_uses_only_positive_finite_uncertainties(self):
        diagnostics, band_diagnostics = build_wavelength_estimation_context(
            wavelengths=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            fluxes=[9.0, 10.0, 11.0, 18.0, 20.0, 22.0],
            uncertainties=[0.5, -1.0, np.nan, 1.0, 1.0, 1.0],
            band_labels=["a", "a", "a", "b", "b", "b"],
        )

        self.assertTrue(diagnostics.available)
        self.assertAlmostEqual(band_diagnostics["a"].median_uncertainty, 0.5)
        self.assertEqual(
            band_diagnostics["a"].metadata[
                "n_positive_finite_uncertainties"
            ],
            1,
        )
        self.assertLessEqual(
            band_diagnostics["a"].noise_corrected_robust_scatter,
            band_diagnostics["a"].robust_scatter,
        )

    def test_fallback_labels_group_exact_numeric_wavelengths(self):
        diagnostics, band_diagnostics = build_wavelength_estimation_context(
            wavelengths=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            fluxes=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )

        self.assertTrue(diagnostics.available)
        self.assertEqual(len(band_diagnostics), 2)
        self.assertIn("wavelength=1", band_diagnostics)
        self.assertIn("wavelength=2", band_diagnostics)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            build_wavelength_estimation_context(
                wavelengths=[1.0, 2.0],
                fluxes=[1.0],
            )

        with self.assertRaisesRegex(ValueError, "same length"):
            build_wavelength_estimation_context(
                wavelengths=[1.0, 2.0],
                fluxes=[1.0, 2.0],
                uncertainties=[0.1],
            )

    def test_rejects_nonpositive_minimum_band_count(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            build_wavelength_estimation_context(
                wavelengths=[1.0],
                fluxes=[1.0],
                min_points_per_band=0,
            )


if __name__ == "__main__":
    unittest.main()
