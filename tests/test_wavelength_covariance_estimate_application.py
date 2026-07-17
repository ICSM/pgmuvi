import math
import unittest

import gpytorch
import torch

from pgmuvi.lightcurve import Lightcurve, Transformer
from pgmuvi.parameter_application import ParameterEstimateApplicator
from pgmuvi.parameter_estimates import ParameterEstimate, ParameterEstimateCollection
from pgmuvi.parameter_specs import (
    ConstraintStrategy,
    GuessStrategy,
    ParameterDomain,
    ParameterRole,
    ParameterScale,
    ParameterSpec,
)


class BrokenScaleTransform(Transformer):
    def transform(self, data, **kwargs):
        del kwargs
        return data

    def inverse(self, data, **kwargs):
        del kwargs
        return data

    def transform_uncertainty(self, data, **kwargs):
        del data, kwargs
        raise RuntimeError("scale transform unavailable")


class TestWavelengthCoordinateTransformation(unittest.TestCase):
    @staticmethod
    def _data():
        xdata = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [0.0, 2.0],
                [1.0, 2.0],
                [2.0, 2.0],
                [0.0, 5.0],
                [1.0, 5.0],
                [2.0, 5.0],
            ]
        )
        ydata = torch.tensor(
            [1.0, 1.2, 0.8, 2.0, 2.2, 1.8, 4.0, 4.2, 3.8]
        )
        return xdata, ydata

    def test_minmax_transforms_raw_wavelength_scale_into_model_space(self):
        xdata, ydata = self._data()
        lc = Lightcurve(xdata, ydata, xtransform="minmax")

        diagnostics = lc._build_parameter_estimation_context().wavelength_diagnostics

        self.assertEqual(
            diagnostics.schema_version,
            "pgmuvi-wavelength-estimation-v2",
        )
        self.assertAlmostEqual(
            diagnostics.recommended_lengthscale_initial,
            math.sqrt(8.0),
        )
        self.assertEqual(
            diagnostics.recommended_lengthscale_bounds,
            (0.25, 20.0),
        )
        self.assertEqual(diagnostics.model_coordinate_space, "transformed_input")
        self.assertAlmostEqual(
            diagnostics.model_recommended_lengthscale_initial,
            math.sqrt(8.0) / 4.0,
        )
        lower, upper = diagnostics.model_recommended_lengthscale_bounds
        self.assertAlmostEqual(lower, 0.25 / 4.0)
        self.assertAlmostEqual(upper, 20.0 / 4.0)
        self.assertEqual(diagnostics.lengthscale_transform_status, "applied")
        self.assertEqual(diagnostics.lengthscale_transform_name, "MinMax")

    def test_failed_scale_transform_does_not_fall_back_to_raw_units(self):
        xdata, ydata = self._data()
        lc = Lightcurve(xdata, ydata, xtransform=BrokenScaleTransform())

        diagnostics = lc._build_parameter_estimation_context().wavelength_diagnostics

        self.assertIsNotNone(diagnostics.recommended_lengthscale_initial)
        self.assertIsNone(diagnostics.model_recommended_lengthscale_initial)
        self.assertIsNone(diagnostics.model_recommended_lengthscale_bounds)
        self.assertEqual(diagnostics.lengthscale_transform_status, "failed")
        self.assertEqual(
            diagnostics.lengthscale_transform_name,
            "BrokenScaleTransform",
        )
        self.assertIn(
            "scale transform unavailable",
            diagnostics.metadata["lengthscale_transform_error"],
        )

    def test_time_center_preserves_wavelength_scale(self):
        xdata, ydata = self._data()
        lc = Lightcurve(xdata, ydata, xtransform="time_center")

        diagnostics = lc._build_parameter_estimation_context().wavelength_diagnostics

        self.assertAlmostEqual(
            diagnostics.model_recommended_lengthscale_initial,
            diagnostics.recommended_lengthscale_initial,
        )
        self.assertEqual(
            diagnostics.model_recommended_lengthscale_bounds,
            diagnostics.recommended_lengthscale_bounds,
        )
        self.assertEqual(diagnostics.lengthscale_transform_status, "applied")
        self.assertEqual(diagnostics.lengthscale_transform_name, "TimeCenter")


class TestSeparableWavelengthCovarianceApplication(unittest.TestCase):
    @staticmethod
    def _lightcurve(*, xtransform=None):
        xdata, ydata = TestWavelengthCoordinateTransformation._data()
        kwargs = {"center_time": False} if xtransform is None else {}
        return Lightcurve(xdata, ydata, xtransform=xtransform, **kwargs)

    def test_applies_wavelength_estimate_to_priority_separable_models(self):
        for model_name in (
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            "2DSeparable",
        ):
            with self.subTest(model=model_name):
                lc = self._lightcurve()
                lc.set_model(model_name)

                result = lc._apply_parameter_workflow_estimates()
                key = "covar_module.kernels.1.base_kernel.lengthscale"

                self.assertIn(key, result)
                self.assertTrue(result[key]["value"])
                self.assertTrue(result[key]["constraint"])
                self.assertIn(
                    result[key]["constraint_action"],
                    {"applied", "tightened", "kept_existing"},
                )
                provenance = result[key]["wavelength_estimate_provenance"]
                self.assertEqual(provenance["value_source"], "wavelength_range")
                self.assertEqual(
                    provenance["constraint_source"],
                    "wavelength_range",
                )
                self.assertAlmostEqual(
                    provenance["estimated_value"],
                    math.sqrt(8.0),
                )
                self.assertEqual(
                    provenance["estimated_constraint"],
                    [0.25, 20.0],
                )
                self.assertEqual(
                    provenance["diagnostics"]["model_coordinate_space"],
                    "raw_input",
                )
                fitted_value = float(
                    lc.model.covar_module.kernels[1]
                    .base_kernel.lengthscale.detach()
                    .cpu()
                    .reshape(-1)[0]
                )
                self.assertAlmostEqual(fitted_value, math.sqrt(8.0), places=5)

                report = lc.get_parameter_workflow_report()
                report_item = next(
                    item
                    for item in report["applied"]
                    if item["parameter"] == key
                )
                self.assertEqual(
                    report_item["constraint_action"],
                    result[key]["constraint_action"],
                )
                self.assertEqual(
                    report_item["wavelength_estimate_provenance"],
                    provenance,
                )

    def test_failed_model_scale_transform_skips_raw_wavelength_estimate(self):
        lc = self._lightcurve(xtransform=BrokenScaleTransform())
        lc.set_model("2DWavelengthDependent")

        result = lc._apply_parameter_workflow_estimates()
        key = "covar_module.kernels.1.base_kernel.lengthscale"
        item = result[key]

        self.assertFalse(item["value"])
        self.assertFalse(item["constraint"])
        self.assertEqual(
            item["value_reason"],
            "wavelength_lengthscale_model_transform_unavailable",
        )
        provenance = item["wavelength_estimate_provenance"]
        self.assertIsNone(provenance["estimated_value"])
        self.assertIsNone(provenance["estimated_constraint"])
        self.assertEqual(
            provenance["diagnostics"]["lengthscale_transform_status"],
            "failed",
        )
        self.assertIn(
            "scale transform unavailable",
            provenance["diagnostics"]["lengthscale_transform_error"],
        )

    def test_quasi_periodic_time_kernel_keeps_independent_wavelength_estimate(self):
        lc = self._lightcurve()
        lc.set_model(
            "2DDustMean",
            time_kernel_type="quasi_periodic",
            period=2.0,
        )

        result = lc._apply_parameter_workflow_estimates()
        key = "covar_module.kernels.1.base_kernel.lengthscale"

        self.assertTrue(result[key]["value"])
        self.assertTrue(result[key]["constraint"])
        self.assertAlmostEqual(
            result[key]["wavelength_estimate_provenance"]["estimated_value"],
            math.sqrt(8.0),
        )
        time_decay_key = (
            "covar_module.kernels.0.base_kernel.kernels.1.lengthscale"
        )
        self.assertIn(time_decay_key, result)
        self.assertNotIn(
            "wavelength_estimate_provenance",
            result[time_decay_key],
        )
        period_length = float(
            lc.model.covar_module.kernels[0]
            .base_kernel.kernels[0].period_length.detach()
            .cpu()
            .reshape(-1)[0]
        )
        self.assertAlmostEqual(period_length, 2.0)

    def test_applies_transformed_estimate_not_raw_value(self):
        lc = self._lightcurve(xtransform="minmax")
        lc.set_model("2DWavelengthDependent")

        result = lc._apply_parameter_workflow_estimates()
        key = "covar_module.kernels.1.base_kernel.lengthscale"
        provenance = result[key]["wavelength_estimate_provenance"]

        self.assertAlmostEqual(
            provenance["estimated_value"],
            math.sqrt(8.0) / 4.0,
        )
        self.assertEqual(
            provenance["diagnostics"]["raw_recommended_lengthscale_bounds"],
            [0.25, 20.0],
        )
        self.assertEqual(
            provenance["diagnostics"]["model_recommended_lengthscale_bounds"],
            [0.0625, 5.0],
        )
        fitted_value = float(
            lc.model.covar_module.kernels[1]
            .base_kernel.lengthscale.detach()
            .cpu()
            .reshape(-1)[0]
        )
        self.assertAlmostEqual(fitted_value, math.sqrt(8.0) / 4.0, places=5)

    def test_full_2d_uses_dimension_aware_spectral_mixture_strategy(self):
        lc = self._lightcurve()
        lc.set_model("2D", num_mixtures=1)

        schema = lc.model.parameter_schema()
        means = schema["covar_module.mixture_means"]
        scales = schema["covar_module.mixture_scales"]

        for spec in (means, scales):
            self.assertIs(
                spec.guess_strategy,
                GuessStrategy.DIMENSION_AWARE_SM_ARD,
            )
            self.assertIs(
                spec.constraint_strategy,
                ConstraintStrategy.DIMENSION_AWARE_SM_ARD,
            )
        self.assertFalse(
            any(
                spec.guess_strategy is GuessStrategy.WAVELENGTH_RANGE
                or spec.constraint_strategy is ConstraintStrategy.WAVELENGTH_RANGE
                for spec in schema
            )
        )


class TestWavelengthConstraintIntersection(unittest.TestCase):
    def test_wavelength_constraint_intersects_existing_interval(self):
        class Model:
            def __init__(self):
                self.covar_module = gpytorch.kernels.RBFKernel()

        model = Model()
        model.covar_module.register_constraint(
            "raw_lengthscale",
            gpytorch.constraints.Interval(0.5, 2.0),
        )
        estimate = ParameterEstimate(
            spec=ParameterSpec(
                name="covar_module.lengthscale",
                role=ParameterRole.WAVELENGTH_SCALE,
                domain=ParameterDomain.WAVELENGTH,
                scale=ParameterScale.LOG,
                guess_strategy=GuessStrategy.WAVELENGTH_RANGE,
                constraint_strategy=ConstraintStrategy.WAVELENGTH_RANGE,
            ),
            value=1.5,
            constraint=(0.1, 10.0),
            value_source="wavelength_range",
            constraint_source="wavelength_range",
            diagnostics={"model_coordinate_space": "raw_input"},
        )

        result = ParameterEstimateApplicator().apply(
            model=model,
            estimates=ParameterEstimateCollection([estimate]),
        )

        item = result["covar_module.lengthscale"]
        self.assertEqual(item["constraint_action"], "kept_existing")
        self.assertEqual(
            item["wavelength_estimate_provenance"]["effective_constraint"],
            [0.5, 2.0],
        )
        self.assertAlmostEqual(
            float(model.covar_module.lengthscale.detach().reshape(-1)[0]),
            1.5,
        )


if __name__ == "__main__":
    unittest.main()
