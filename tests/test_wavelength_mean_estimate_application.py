import unittest

import gpytorch
import numpy as np
import torch

from pgmuvi.constraint_utils import register_constraint_preserving_value
from pgmuvi.lightcurve import Lightcurve


class TestWavelengthMeanEstimateApplication(unittest.TestCase):
    def _lightcurve(self, flux_function, *, xtransform="minmax"):
        wavelengths = np.asarray([0.45, 0.65, 1.2, 2.2, 4.0])
        rows = []
        fluxes = []
        bands = []
        for band_index, wavelength in enumerate(wavelengths):
            for sample in range(8):
                rows.append([float(sample), wavelength])
                fluxes.append(
                    float(flux_function(wavelength))
                    + 0.01 * np.sin(float(sample))
                )
                bands.append(f"b{band_index}")
        return Lightcurve(
            np.asarray(rows),
            np.asarray(fluxes),
            band=np.asarray(bands),
            xtransform=xtransform,
            ytransform="zscore",
        )

    def test_power_law_mean_uses_reconstructed_physical_wavelength(self):
        for transform in ("minmax", "zscore", "robust_zscore", "time_center"):
            with self.subTest(transform=transform):
                lc = self._lightcurve(
                    lambda wavelength: 1.0 + 2.0 * wavelength**1.5,
                    xtransform=transform,
                )
                lc.set_model("2DPowerLawMean")
                result = lc._apply_parameter_workflow_estimates()

                mean = lc.model.mean_module
                reconstructed = mean.physical_wavelength(lc._xdata_transformed)
                self.assertTrue(
                    torch.allclose(
                        reconstructed,
                        lc._xdata_raw[:, 1],
                        atol=1.0e-6,
                        rtol=1.0e-6,
                    )
                )
                for name in ("offset", "weight", "exponent"):
                    record = result[f"mean_module.{name}"]
                    self.assertTrue(record["value"])
                    self.assertTrue(record["constraint"])
                    self.assertIn("wavelength_mean_estimate_provenance", record)
                    self.assertIsInstance(
                        getattr(mean, f"raw_{name}_constraint"),
                        gpytorch.constraints.Interval,
                    )
                self.assertTrue(
                    torch.all(torch.isfinite(mean(lc._xdata_transformed)))
                )
                parameters = lc.get_parameters(transform=False)
                self.assertIn("mean_module.weight", parameters)
                self.assertNotIn("mean_module.eight", parameters)
                self.assertIsNotNone(
                    result["mean_module.weight"][
                        "wavelength_mean_estimate_provenance"
                    ]["effective_constraint"]
                )

    def test_dust_mean_applies_positive_physical_shape_estimates(self):
        lc = self._lightcurve(
            lambda wavelength: 0.2
            + 3.0 * np.exp(-1.4 * wavelength ** -1.8)
        )
        lc.set_model("2DDustMean")
        result = lc._apply_parameter_workflow_estimates()

        mean = lc.model.mean_module
        for name in ("offset", "log_amplitude", "log_tau", "log_alpha"):
            record = result[f"mean_module.{name}"]
            self.assertTrue(record["value"])
            self.assertTrue(record["constraint"])
            self.assertIn("wavelength_mean_estimate_provenance", record)
        self.assertGreater(float(mean.log_amplitude.detach().exp()), 0.0)
        self.assertGreater(float(mean.log_tau.detach().exp()), 0.0)
        self.assertGreater(float(mean.log_alpha.detach().exp()), 0.0)
        self.assertTrue(torch.all(torch.isfinite(mean(lc._xdata_transformed))))

    def test_quadratic_mean_applies_vector_constraints(self):
        lc = self._lightcurve(
            lambda wavelength: 1.0 + 0.5 * wavelength - 0.1 * wavelength**2
        )
        lc.set_model("2DWavelengthDependent")
        result = lc._apply_parameter_workflow_estimates()

        mean = lc.model.mean_module
        weights = result["mean_module.weights"]
        bias = result["mean_module.bias"]
        self.assertTrue(weights["value"])
        self.assertTrue(weights["constraint"])
        self.assertTrue(bias["value"])
        self.assertTrue(bias["constraint"])
        lower = mean.raw_weights_constraint.lower_bound
        upper = mean.raw_weights_constraint.upper_bound
        self.assertEqual(tuple(lower.shape), (2,))
        self.assertTrue(torch.all(lower < mean.weights))
        self.assertTrue(torch.all(mean.weights < upper))

    def test_existing_tighter_exponent_constraint_is_preserved(self):
        lc = self._lightcurve(lambda wavelength: 1.0 + 2.0 * wavelength**0.5)
        lc.set_model("2DPowerLawMean")
        mean = lc.model.mean_module
        register_constraint_preserving_value(
            mean,
            "raw_exponent",
            gpytorch.constraints.Interval(0.25, 0.75),
        )

        result = lc._apply_parameter_workflow_estimates()
        constraint = mean.raw_exponent_constraint
        self.assertAlmostEqual(float(constraint.lower_bound), 0.25)
        self.assertAlmostEqual(float(constraint.upper_bound), 0.75)
        self.assertGreaterEqual(float(mean.exponent.detach()), 0.25)
        self.assertLessEqual(float(mean.exponent.detach()), 0.75)
        provenance = result["mean_module.exponent"][
            "wavelength_mean_estimate_provenance"
        ]
        self.assertEqual(provenance["effective_constraint"], [0.25, 0.75])

    def test_quasi_periodic_time_handoff_is_unchanged(self):
        lc = self._lightcurve(lambda wavelength: 1.0 + wavelength**1.2)
        lc.set_model(
            "2DPowerLawMean",
            time_kernel_type="quasi_periodic",
            period=3.0,
        )
        result = lc._apply_parameter_workflow_estimates()

        self.assertTrue(result["mean_module.exponent"]["value"])
        period_length = (
            lc.model.covar_module.kernels[0]
            .base_kernel.kernels[0]
            .period_length.detach()
            .reshape(-1)[0]
        )
        self.assertAlmostEqual(float(period_length), 3.0, places=6)

    def test_non_affine_wavelength_transform_is_rejected(self):
        class SquareWavelengthTransform(torch.nn.Module):
            pass

        lc = self._lightcurve(lambda wavelength: wavelength, xtransform=None)
        lc._xdata_transformed = lc._xdata_raw.clone()
        lc._xdata_transformed[:, 1] = lc._xdata_raw[:, 1] ** 2
        lc.xtransform = SquareWavelengthTransform()
        with self.assertRaisesRegex(ValueError, "affine wavelength input transform"):
            lc.set_model("2DPowerLawMean")


if __name__ == "__main__":
    unittest.main()
