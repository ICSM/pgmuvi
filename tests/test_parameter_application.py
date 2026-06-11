import unittest

from pgmuvi.parameter_application import ParameterEstimateApplicator
from pgmuvi.parameter_estimates import ParameterEstimateCollection
from pgmuvi.parameter_estimates import ParameterEstimate
from pgmuvi.parameter_specs import (
    ParameterDomain,
    ParameterRole,
    ParameterScale,
    ParameterSpec,
)
import math
import torch
import gpytorch


class DummyRawConstraintModule:
    def __init__(self):
        self.raw_lengthscale = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def register_constraint(self, parameter_name, constraint):
        self.calls.append((parameter_name, constraint))


class DummyRawConstraintModel:
    def __init__(self):
        self.covar_module = DummyRawConstraintModule()


class DummyTensorMeanModule:
    def __init__(self):
        self.offset = torch.nn.Parameter(torch.zeros(1))
        self.log_amplitude = torch.nn.Parameter(torch.zeros(1))


class DummyTensorModel:
    def __init__(self):
        self.mean_module = DummyTensorMeanModule()


class DummyConstraintModule:
    def __init__(self):
        self.calls = []

    def register_constraint(
        self,
        parameter_name,
        constraint,
    ):
        self.calls.append(
            (
                parameter_name,
                constraint,
            )
        )


class DummyConstraintModel:
    def __init__(self):
        self.mean_module = DummyConstraintModule()


class TestParameterEstimateApplicator(unittest.TestCase):

    def test_apply_empty_collection_returns_empty_results(self):
        applicator = ParameterEstimateApplicator()

        results = applicator.apply(
            model=object(),
            estimates=ParameterEstimateCollection(),
        )

        self.assertEqual(results, {})

    def test_resolve_parameter(self):
        applicator = ParameterEstimateApplicator()

        model = DummyModel()

        result = applicator._resolve_parameter(
            model,
            "mean_module.offset",
        )

        self.assertEqual(result, 123.0)

    def test_transform_linear_value_returns_value_unchanged(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
        )
        estimate = ParameterEstimate(
            spec=spec,
            value=123.0,
        )

        applicator = ParameterEstimateApplicator()

        self.assertEqual(
            applicator._transform_value(estimate),
            123.0,
        )

    def test_transform_value_returns_none_when_value_missing(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
        )
        estimate = ParameterEstimate(spec=spec)

        applicator = ParameterEstimateApplicator()

        self.assertIsNone(
            applicator._transform_value(estimate)
        )

    def test_transform_unsupported_value_scale_is_not_implemented_yet(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LOG10,
        )
        estimate = ParameterEstimate(
            spec=spec,
            value=100.0,
        )

        applicator = ParameterEstimateApplicator()

        with self.assertRaises(NotImplementedError):
            applicator._transform_value(estimate)

    def test_transform_log_value_applies_natural_log(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LOG,
        )
        estimate = ParameterEstimate(
            spec=spec,
            value=100.0,
        )

        applicator = ParameterEstimateApplicator()

        self.assertAlmostEqual(
            applicator._transform_value(estimate),
            4.605170185988092,
        )

    def test_transform_log_value_rejects_non_positive_values(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LOG,
        )
        estimate = ParameterEstimate(
            spec=spec,
            value=0.0,
        )

        applicator = ParameterEstimateApplicator()

        with self.assertRaisesRegex(ValueError, "non-positive"):
            applicator._transform_value(estimate)

    def test_apply_linear_value_to_tensor_parameter(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
        )
        estimate = ParameterEstimate(
            spec=spec,
            value=123.0,
        )

        model = DummyTensorModel()
        applicator = ParameterEstimateApplicator()

        applied = applicator._apply_value(
            model.mean_module,
            "offset",
            estimate,
        )

        self.assertTrue(applied)
        self.assertAlmostEqual(
            float(model.mean_module.offset.item()),
            123.0,
        )

    def test_apply_log_value_to_tensor_parameter(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LOG,
        )
        estimate = ParameterEstimate(
            spec=spec,
            value=100.0,
        )

        model = DummyTensorModel()
        applicator = ParameterEstimateApplicator()

        applied = applicator._apply_value(
            model.mean_module,
            "log_amplitude",
            estimate,
        )

        self.assertTrue(applied)
        self.assertAlmostEqual(
            float(model.mean_module.log_amplitude.item()),
            4.605170185988092,
            places=6,
        )

    def test_apply_value_returns_false_when_value_missing(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
        )
        estimate = ParameterEstimate(spec=spec)

        model = DummyTensorModel()
        applicator = ParameterEstimateApplicator()

        applied = applicator._apply_value(
            model.mean_module,
            "offset",
            estimate,
        )

        self.assertFalse(applied)
        self.assertAlmostEqual(
            float(model.mean_module.offset.item()),
            0.0,
        )

    def test_apply_collection_applies_available_values(self):
        offset_spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
        )
        amplitude_spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LOG,
        )

        estimates = ParameterEstimateCollection(
            [
                ParameterEstimate(
                    spec=offset_spec,
                    value=123.0,
                ),
                ParameterEstimate(
                    spec=amplitude_spec,
                    value=100.0,
                ),
            ]
        )

        model = DummyTensorModel()
        applicator = ParameterEstimateApplicator()

        results = applicator.apply(
            model=model,
            estimates=estimates,
        )

        self.assertEqual(
            results,
            {
                "mean_module.offset": {
                    "value": True,
                    "constraint": False,
                },
                "mean_module.log_amplitude": {
                    "value": True,
                    "constraint": False,
                },
            },
        )

        self.assertAlmostEqual(
            float(model.mean_module.offset.item()),
            123.0,
        )
        self.assertAlmostEqual(
            float(model.mean_module.log_amplitude.item()),
            4.605170185988092,
            places=6,
        )

    def test_transform_linear_constraint_returns_constraint_unchanged(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
        )

        estimate = ParameterEstimate(
            spec=spec,
            constraint=(10.0, 100.0),
        )

        applicator = ParameterEstimateApplicator()

        self.assertEqual(
            applicator._transform_constraint(estimate),
            (10.0, 100.0),
        )

    def test_transform_log_constraint_applies_natural_log(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LOG,
        )

        estimate = ParameterEstimate(
            spec=spec,
            constraint=(1.0e-4, 450.0),
        )

        applicator = ParameterEstimateApplicator()

        lower, upper = applicator._transform_constraint(
            estimate
        )

        self.assertAlmostEqual(
            lower,
            math.log(1.0e-4),
        )

        self.assertAlmostEqual(
            upper,
            math.log(450.0),
        )

    def test_transform_constraint_returns_none_when_missing(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
        )

        estimate = ParameterEstimate(
            spec=spec,
        )

        applicator = ParameterEstimateApplicator()

        self.assertIsNone(
            applicator._transform_constraint(estimate)
        )

    def test_transform_log_constraint_rejects_non_positive_bounds(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitude",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LOG,
        )

        estimate = ParameterEstimate(
            spec=spec,
            constraint=(0.0, 100.0),
        )

        applicator = ParameterEstimateApplicator()

        with self.assertRaisesRegex(
            ValueError,
            "non-positive",
        ):
            applicator._transform_constraint(estimate)

    def test_split_nested_parameter_path(self):
        module_path, local_name = ParameterEstimateApplicator._split_parameter_path(
            "mean_module.log_amplitude"
        )

        self.assertEqual(module_path, "mean_module")
        self.assertEqual(local_name, "log_amplitude")

    def test_split_bare_parameter_path(self):
        module_path, local_name = ParameterEstimateApplicator._split_parameter_path(
            "offset"
        )

        self.assertEqual(module_path, "")
        self.assertEqual(local_name, "offset")

    def test_apply_linear_constraint(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
        )

        estimate = ParameterEstimate(
            spec=spec,
            constraint=(10.0, 100.0),
        )

        model = DummyConstraintModel()

        applicator = ParameterEstimateApplicator()

        applied = applicator._apply_constraint(
            model,
            estimate,
        )

        self.assertTrue(applied)

        self.assertEqual(
            len(model.mean_module.calls),
            1,
        )

        parameter_name, constraint = (
            model.mean_module.calls[0]
        )

        self.assertEqual(
            parameter_name,
            "offset",
        )

        self.assertAlmostEqual(
            float(constraint.lower_bound),
            10.0,
        )

        self.assertAlmostEqual(
            float(constraint.upper_bound),
            100.0,
        )

    def test_apply_constraint_returns_false_when_missing(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
        )

        estimate = ParameterEstimate(
            spec=spec,
        )

        model = DummyConstraintModel()

        applicator = ParameterEstimateApplicator()

        self.assertFalse(
            applicator._apply_constraint(
                model,
                estimate,
            )
        )

        self.assertEqual(
            len(model.mean_module.calls),
            0,
        )

    def test_apply_constraint_prefers_raw_parameter_name_when_available(self):
        spec = ParameterSpec(
            name="covar_module.lengthscale",
            role=ParameterRole.LENGTHSCALE,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LOG,
        )

        estimate = ParameterEstimate(
            spec=spec,
            constraint=(1.0, 100.0),
        )

        model = DummyRawConstraintModel()
        applicator = ParameterEstimateApplicator()

        applied = applicator._apply_constraint(
            model,
            estimate,
        )

        self.assertTrue(applied)
        self.assertEqual(len(model.covar_module.calls), 1)

        parameter_name, constraint = model.covar_module.calls[0]

        self.assertEqual(parameter_name, "raw_lengthscale")
        self.assertAlmostEqual(float(constraint.lower_bound), 1.0)
        self.assertAlmostEqual(float(constraint.upper_bound), 100.0)

    def test_transform_log_vector_value_applies_elementwise_log(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitudes",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LOG,
            shape=(2,),
        )

        estimate = ParameterEstimate(
            spec=spec,
            value=[1.0, 100.0],
        )

        applicator = ParameterEstimateApplicator()
        transformed = applicator._transform_value(estimate)

        self.assertTrue(torch.allclose(
            transformed,
            torch.tensor([0.0, 4.605170185988092]),
        ))

    def test_transform_log_vector_constraint_applies_elementwise_log(self):
        spec = ParameterSpec(
            name="mean_module.log_amplitudes",
            role=ParameterRole.AMPLITUDE,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LOG,
            shape=(2,),
        )

        estimate = ParameterEstimate(
            spec=spec,
            constraint=(
                [1.0, 10.0],
                [100.0, 1000.0],
            ),
        )

        applicator = ParameterEstimateApplicator()
        lower, upper = applicator._transform_constraint(estimate)

        self.assertTrue(
            torch.allclose(
                lower,
                torch.tensor([0.0, 2.302585092994046]),
            )
        )
        self.assertTrue(
            torch.allclose(
                upper,
                torch.tensor([4.605170185988092, 6.907755278982137]),
            )
        )

    def test_apply_value_updates_gpytorch_property_parameter(self):
        spec = ParameterSpec(
            name="covar_module.lengthscale",
            role=ParameterRole.LENGTHSCALE,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LOG,
        )

        estimates = ParameterEstimateCollection(
            [
                ParameterEstimate(
                    spec=spec,
                    value=2.0,
                )
            ]
        )

        class Model:
            def __init__(self):
                self.covar_module = gpytorch.kernels.RBFKernel()

        model = Model()
        applicator = ParameterEstimateApplicator()

        results = applicator.apply(
            model=model,
            estimates=estimates,
        )

        self.assertEqual(
            results,
            {
                "covar_module.lengthscale": {
                    "value": True,
                    "constraint": False,
                },
            },
        )

        self.assertAlmostEqual(
            float(model.covar_module.lengthscale.detach().cpu().view(-1)[0]),
            2.0,
        )


class DummyMeanModule:
    def __init__(self):
        self.offset = 123.0


class DummyModel:
    def __init__(self):
        self.mean_module = DummyMeanModule()


if __name__ == "__main__":
    unittest.main()
