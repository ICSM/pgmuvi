import unittest

import numpy as np

from pgmuvi.wavelength_estimation import (
    build_wavelength_mean_estimation_context,
)


class TestWavelengthMeanEstimation(unittest.TestCase):
    def _rows(self, wavelengths, model_wavelengths, fluxes, n=5):
        raw = []
        model = []
        values = []
        bands = []
        for index, (raw_wl, model_wl, flux) in enumerate(
            zip(wavelengths, model_wavelengths, fluxes, strict=True)
        ):
            for offset in np.linspace(-0.02, 0.02, n):
                raw.append(raw_wl)
                model.append(model_wl)
                values.append(flux + offset)
                bands.append(f"b{index}")
        return raw, model, values, bands

    def test_quadratic_recommendation_uses_model_coordinate(self):
        model_wavelengths = np.asarray([-1.0, -0.2, 0.6, 1.4])
        fluxes = 3.0 + 2.0 * model_wavelengths - 0.5 * model_wavelengths**2
        inputs = self._rows(
            [0.5, 1.0, 2.0, 4.0],
            model_wavelengths,
            fluxes,
        )
        diagnostics = build_wavelength_mean_estimation_context(*inputs)
        record = diagnostics.recommendations["2DWavelengthDependent"]

        self.assertTrue(record["available"])
        self.assertEqual(
            record["coordinate_basis"],
            "model_wavelength_and_model_flux",
        )
        self.assertAlmostEqual(
            record["initial_values"]["mean_module.bias"],
            3.0,
            places=6,
        )
        weights = record["initial_values"]["mean_module.weights"]
        self.assertAlmostEqual(weights[0], 2.0, places=6)
        self.assertAlmostEqual(weights[1], -0.5, places=6)

    def test_power_law_recommendation_uses_physical_wavelength(self):
        raw_wavelengths = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0])
        model_wavelengths = (raw_wavelengths - raw_wavelengths.min()) / np.ptp(
            raw_wavelengths
        )
        fluxes = 1.5 + 2.25 * raw_wavelengths**1.7
        inputs = self._rows(
            raw_wavelengths,
            model_wavelengths,
            fluxes,
        )
        diagnostics = build_wavelength_mean_estimation_context(*inputs)
        record = diagnostics.recommendations["2DPowerLawMean"]

        self.assertTrue(record["available"])
        self.assertEqual(
            record["coordinate_basis"],
            "physical_wavelength_and_model_flux",
        )
        exponent = record["initial_values"]["mean_module.exponent"]
        self.assertAlmostEqual(exponent, 1.7, delta=0.08)
        self.assertLess(record["fit_rmse"], 0.1)
        profile = record["profile_support"]
        self.assertEqual(profile["criterion"], "same_sign_rmse_profile")
        self.assertGreater(profile["n_supported_candidates"], 0)
        for parameter, value in record["initial_values"].items():
            lower, upper = record["constraints"][parameter]
            self.assertLess(lower, value)
            self.assertLess(value, upper)
        exponent_bounds = record["constraints"]["mean_module.exponent"]
        weight_bounds = record["constraints"]["mean_module.weight"]
        self.assertGreater(exponent_bounds[0], 0.0)
        self.assertGreater(weight_bounds[0], 0.0)
        self.assertLess(exponent_bounds[1] - exponent_bounds[0], 5.0)


    def test_power_law_profile_preserves_negative_exponent_branch(self):
        raw_wavelengths = np.asarray([0.5, 0.8, 1.2, 2.0, 4.0, 6.0])
        model_wavelengths = (raw_wavelengths - raw_wavelengths.min()) / np.ptp(
            raw_wavelengths
        )
        fluxes = 2.0 + 5.0 * raw_wavelengths**-1.4
        inputs = self._rows(
            raw_wavelengths,
            model_wavelengths,
            fluxes,
        )
        diagnostics = build_wavelength_mean_estimation_context(*inputs)
        record = diagnostics.recommendations["2DPowerLawMean"]

        exponent = record["initial_values"]["mean_module.exponent"]
        lower, upper = record["constraints"]["mean_module.exponent"]
        weight_lower, weight_upper = record["constraints"][
            "mean_module.weight"
        ]
        self.assertLess(exponent, 0.0)
        self.assertLess(upper, 0.0)
        self.assertLess(lower, exponent)
        self.assertLess(exponent, upper)
        self.assertGreater(weight_lower, 0.0)
        self.assertGreater(weight_upper, weight_lower)

    def test_dust_recommendation_is_positive_in_physical_parameters(self):
        raw_wavelengths = np.asarray([0.45, 0.65, 1.2, 2.2, 4.0])
        model_wavelengths = (raw_wavelengths - 2.0) / 1.5
        fluxes = 0.2 + 3.0 * np.exp(-1.4 * raw_wavelengths ** -1.8)
        inputs = self._rows(
            raw_wavelengths,
            model_wavelengths,
            fluxes,
        )
        diagnostics = build_wavelength_mean_estimation_context(*inputs)
        record = diagnostics.recommendations["2DDustMean"]

        self.assertTrue(record["available"])
        values = record["initial_values"]
        self.assertGreater(values["mean_module.log_amplitude"], 0.0)
        self.assertGreater(values["mean_module.log_tau"], 0.0)
        self.assertGreater(values["mean_module.log_alpha"], 0.0)
        self.assertEqual(
            record["coordinate_basis"],
            "physical_wavelength_and_model_flux",
        )

    def test_two_band_power_law_uses_fixed_default_exponent(self):
        inputs = self._rows([0.5, 2.0], [0.0, 1.0], [4.0, 1.0])
        diagnostics = build_wavelength_mean_estimation_context(*inputs)
        record = diagnostics.recommendations["2DPowerLawMean"]

        self.assertTrue(record["available"])
        self.assertEqual(record["exponent_estimation"], "fixed_default_two_band")
        self.assertEqual(
            record["initial_values"]["mean_module.exponent"],
            -2.0,
        )
        self.assertIsNone(record["profile_support"])
        self.assertEqual(
            record["constraints"]["mean_module.exponent"],
            [-10.0, 10.0],
        )

    def test_flat_trend_does_not_invent_dust_shape(self):
        inputs = self._rows(
            [0.5, 1.0, 2.0, 4.0],
            [0.0, 0.3, 0.7, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            n=3,
        )
        # Remove the small within-band offsets added by _rows so the band
        # medians are exactly constant.
        inputs = (inputs[0], inputs[1], [2.0] * len(inputs[2]), inputs[3])
        diagnostics = build_wavelength_mean_estimation_context(*inputs)
        dust = diagnostics.recommendations["2DDustMean"]
        power = diagnostics.recommendations["2DPowerLawMean"]

        self.assertFalse(dust["available"])
        self.assertEqual(dust["reason"], "wavelength_mean_not_resolved")
        self.assertTrue(power["available"])
        self.assertEqual(power["exponent_estimation"], "fixed_default_flat_trend")

    def test_insufficient_bands_are_reported_per_model(self):
        inputs = self._rows([0.5, 1.0], [0.0, 1.0], [1.0, 2.0])
        diagnostics = build_wavelength_mean_estimation_context(*inputs)

        self.assertTrue(
            diagnostics.recommendations["2DWavelengthDependent"]["available"]
        )
        self.assertFalse(diagnostics.recommendations["2DDustMean"]["available"])
        self.assertIn("2DDustMean", " ".join(diagnostics.warnings))


if __name__ == "__main__":
    unittest.main()
