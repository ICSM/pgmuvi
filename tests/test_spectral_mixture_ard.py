import math
import unittest

import numpy as np

from pgmuvi.parameter_context import WavelengthEstimationDiagnostics
from pgmuvi.spectral_mixture_ard import (
    ARD_COORDINATE_ORDER,
    SPECTRAL_MIXTURE_ARD_SCHEMA_VERSION,
    build_dimension_aware_sm_ard_estimates,
)


class TestDimensionAwareSpectralMixtureArd(unittest.TestCase):
    @staticmethod
    def _inputs():
        raw = np.asarray(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [8.0, 1.0],
                [0.0, 2.0],
                [1.0, 2.0],
                [2.0, 2.0],
                [8.0, 2.0],
                [0.0, 5.0],
                [1.0, 5.0],
                [2.0, 5.0],
                [8.0, 5.0],
            ]
        )
        model = raw.copy()
        model[:, 0] /= 8.0
        model[:, 1] = (model[:, 1] - 1.0) / 4.0
        return raw, model

    def test_schema_and_shapes_are_explicit(self):
        raw, model = self._inputs()
        result = build_dimension_aware_sm_ard_estimates(
            raw_inputs=raw,
            model_inputs=model,
            num_mixtures=3,
        )

        self.assertEqual(
            result["schema_version"],
            SPECTRAL_MIXTURE_ARD_SCHEMA_VERSION,
        )
        self.assertEqual(result["coordinate_order"], list(ARD_COORDINATE_ORDER))
        self.assertEqual(result["ard_index"]["temporal_frequency"], 0)
        self.assertEqual(result["ard_index"]["wavelength_frequency"], 1)
        self.assertEqual(result["constraint_shape"], [1, 1, 2])
        self.assertEqual(result["value_shape"], [3, 1, 2])
        self.assertEqual(
            np.asarray(
                result["model_coordinate"]["mixture_means"]["initial_value"]
            ).shape,
            (3, 1, 2),
        )

    def test_time_and_wavelength_bounds_are_independent(self):
        raw, model = self._inputs()
        result = build_dimension_aware_sm_ard_estimates(
            raw_inputs=raw,
            model_inputs=model,
            num_mixtures=2,
        )
        means = result["model_coordinate"]["mixture_means"]
        scales = result["model_coordinate"]["mixture_scales"]

        self.assertNotEqual(
            means["constraint_lower"][0][0][0],
            means["constraint_lower"][0][0][1],
        )
        self.assertNotEqual(
            scales["constraint_upper"][0][0][0],
            scales["constraint_upper"][0][0][1],
        )
        self.assertEqual(means["temporal_initialization"], "baseline_frequency_sequence")

    def test_raw_and_model_coordinates_are_recorded_separately(self):
        raw, model = self._inputs()
        result = build_dimension_aware_sm_ard_estimates(
            raw_inputs=raw,
            model_inputs=model,
            num_mixtures=1,
        )
        raw_lower = result["raw_coordinate"]["mixture_means"][
            "constraint_lower"
        ][0][0]
        model_lower = result["model_coordinate"]["mixture_means"][
            "constraint_lower"
        ][0][0]

        self.assertAlmostEqual(model_lower[0], 8.0 * raw_lower[0])
        self.assertAlmostEqual(model_lower[1], 4.0 * raw_lower[1])

    def test_wavelength_lengthscale_drives_wavelength_scale_only(self):
        raw, model = self._inputs()
        diagnostics = WavelengthEstimationDiagnostics(
            available=True,
            recommended_lengthscale_initial=2.0,
            recommended_lengthscale_bounds=(0.5, 8.0),
            model_recommended_lengthscale_initial=0.5,
            model_recommended_lengthscale_bounds=(0.125, 2.0),
        )
        result = build_dimension_aware_sm_ard_estimates(
            raw_inputs=raw,
            model_inputs=model,
            num_mixtures=1,
            wavelength_diagnostics=diagnostics,
        )
        raw_scale = result["raw_coordinate"]["mixture_scales"]
        model_scale = result["model_coordinate"]["mixture_scales"]

        self.assertAlmostEqual(
            raw_scale["initial_value"][0][0][1],
            1.0 / (2.0 * math.pi * 2.0),
        )
        self.assertAlmostEqual(
            model_scale["initial_value"][0][0][1],
            1.0 / (2.0 * math.pi * 0.5),
        )
        self.assertEqual(
            model_scale["wavelength_lengthscale_source"],
            "wavelength_estimation_context",
        )

    def test_single_wavelength_is_explicit_degenerate_fallback(self):
        raw, model = self._inputs()
        raw[:, 1] = 2.0
        model[:, 1] = 0.0
        result = build_dimension_aware_sm_ard_estimates(
            raw_inputs=raw,
            model_inputs=model,
            num_mixtures=1,
        )

        scales = result["model_coordinate"]["mixture_scales"]
        self.assertEqual(
            scales["wavelength_lengthscale_source"],
            "single_wavelength_degenerate_fallback",
        )
        self.assertGreater(scales["constraint_upper"][0][0][1], 0.0)

    def test_rejects_invalid_component_count(self):
        raw, model = self._inputs()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_dimension_aware_sm_ard_estimates(
                raw_inputs=raw,
                model_inputs=model,
                num_mixtures=0,
            )


if __name__ == "__main__":
    unittest.main()
