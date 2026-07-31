from pathlib import Path
import unittest
import warnings

import numpy as np

from pgmuvi.wavelength_constraint_tutorial import (
    load_wavelength_constraint_tutorial_lightcurve,
)
from pgmuvi.wavelength_estimation import (
    build_wavelength_mean_estimation_context,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_CSV = (
    REPOSITORY_ROOT / "examples/data/10131+3049.csv"
)


class TestWavelengthMeanBiasConstraint(unittest.TestCase):
    @staticmethod
    def _rows(
        model_wavelengths,
        fluxes,
        *,
        n_points=5,
    ):
        raw = []
        model = []
        values = []
        labels = []
        for index, (wavelength, flux) in enumerate(
            zip(model_wavelengths, fluxes, strict=True)
        ):
            for offset in np.linspace(-0.01, 0.01, n_points):
                raw.append(float(wavelength))
                model.append(float(wavelength))
                values.append(float(flux + offset))
                labels.append(f"band_{index}")
        return raw, model, values, labels

    def test_quadratic_bias_constraint_contains_extrapolated_intercept(self):
        model_wavelengths = np.asarray([3.5, 4.0, 4.5, 5.0])
        fluxes = 2.0 + 5.0 * model_wavelengths
        diagnostics = build_wavelength_mean_estimation_context(
            *self._rows(model_wavelengths, fluxes)
        )
        record = diagnostics.recommendations[
            "2DWavelengthDependent"
        ]

        self.assertTrue(record["available"])
        bias = float(
            record["initial_values"]["mean_module.bias"]
        )
        lower, upper = record["constraints"]["mean_module.bias"]

        self.assertAlmostEqual(bias, 2.0, places=6)
        self.assertLess(lower, bias)
        self.assertLess(bias, upper)

        for parameter, value in record["initial_values"].items():
            constraint_lower, constraint_upper = record[
                "constraints"
            ][parameter]
            value_array = np.asarray(value, dtype=float)
            lower_array = np.asarray(
                constraint_lower,
                dtype=float,
            )
            upper_array = np.asarray(
                constraint_upper,
                dtype=float,
            )
            self.assertTrue(np.all(lower_array < value_array))
            self.assertTrue(np.all(value_array < upper_array))

    def test_representative_source_parameter_workflow_applies(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lightcurve, summary = (
                load_wavelength_constraint_tutorial_lightcurve(
                    REPRESENTATIVE_CSV,
                    max_samples_per_observational_channel=50,
                    check_sampling=True,
                    sampling_kwargs=None,
                    name="representative bias-constraint regression",
                )
            )
            lightcurve._apply_duplicate_wavelength_channel_policy(
                policy="first",
                selection=None,
            )
            lightcurve.set_model(
                "2DWavelengthDependent",
                time_kernel_type="quasi_periodic",
                wavelength_kernel_type="rbf",
            )
            lightcurve.set_default_constraints(
                constraint_set="LPV",
            )
            result = (
                lightcurve._apply_parameter_workflow_estimates()
            )

        self.assertGreater(summary["n_rows_retained"], 0)
        bias_result = result["mean_module.bias"]
        self.assertTrue(bias_result["value"])
        self.assertTrue(bias_result["constraint"])

        recommendation = (
            lightcurve._parameter_estimation_context
            .wavelength_mean_diagnostics
            .recommendations["2DWavelengthDependent"]
        )
        bias = float(
            recommendation["initial_values"]["mean_module.bias"]
        )
        lower, upper = recommendation["constraints"][
            "mean_module.bias"
        ]
        self.assertLess(lower, bias)
        self.assertLess(bias, upper)

        effective_constraint = (
            lightcurve.model.mean_module.raw_bias_constraint
        )
        fitted_bias = float(
            lightcurve.model.mean_module.bias.detach()
        )
        self.assertLess(
            float(effective_constraint.lower_bound),
            fitted_bias,
        )
        self.assertLess(
            fitted_bias,
            float(effective_constraint.upper_bound),
        )


if __name__ == "__main__":
    unittest.main()
