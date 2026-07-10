import unittest

from pgmuvi.constraint_utils import clamp_to_constraint_interior
from pgmuvi.constraint_utils import get_bounds
from pgmuvi.constraint_utils import register_constraint_preserving_value
from pgmuvi.parameter_application import ParameterEstimateApplicator
from pgmuvi.parameter_estimates import ParameterEstimateCollection
from pgmuvi.parameter_estimates import ParameterEstimate
from pgmuvi.parameter_specs import (
    ParameterDomain,
    ParameterRole,
    ParameterScale,
    ParameterSpec,
    ConstraintStrategy,
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


class TestConstraintUtils(unittest.TestCase):

    def test_get_bounds_for_interval(self):
        constraint = gpytorch.constraints.Interval(0.1, 10.0)

        lower, upper = get_bounds(constraint)

        self.assertAlmostEqual(float(lower), 0.1)
        self.assertAlmostEqual(float(upper), 10.0)

    def test_get_bounds_for_greater_than(self):
        constraint = gpytorch.constraints.GreaterThan(0.1)

        lower, upper = get_bounds(constraint)

        self.assertAlmostEqual(float(lower), 0.1)
        self.assertTrue(torch.isinf(torch.as_tensor(upper)))

    def test_get_bounds_for_positive(self):
        constraint = gpytorch.constraints.Positive()

        lower, upper = get_bounds(constraint)

        self.assertAlmostEqual(float(lower), 0.0)
        self.assertTrue(torch.isinf(torch.as_tensor(upper)))

    def test_clamp_to_constraint_interior_moves_endpoint_values(self):
        constraint = gpytorch.constraints.Interval(0.05, 10.0)
        value = torch.tensor([0.05, 1.0, 10.0])

        clipped = clamp_to_constraint_interior(value, constraint)

        self.assertGreater(float(clipped[0]), 0.05)
        self.assertAlmostEqual(float(clipped[1]), 1.0)
        self.assertLess(float(clipped[2]), 10.0)

    def test_register_constraint_preserving_value_keeps_gpytorch_property(self):
        kernel = gpytorch.kernels.RBFKernel()
        kernel.initialize(lengthscale=torch.tensor(2.0))

        register_constraint_preserving_value(
            kernel,
            "raw_lengthscale",
            gpytorch.constraints.Interval(0.1, 10.0),
        )

        self.assertAlmostEqual(
            float(kernel.lengthscale.detach().cpu().view(-1)[0]),
            2.0,
            places=5,
        )

    def test_register_constraint_preserving_value_clips_to_safe_interior(self):
        kernel = gpytorch.kernels.RBFKernel()
        kernel.initialize(lengthscale=torch.tensor(0.05))

        register_constraint_preserving_value(
            kernel,
            "raw_lengthscale",
            gpytorch.constraints.Interval(0.05, 10.0),
        )

        value = float(kernel.lengthscale.detach().cpu().view(-1)[0])
        self.assertGreater(value, 0.05)
        self.assertLess(value, 10.0)


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
                    "value_reason": None,
                    "constraint_reason": "constraint_unavailable",
                },
                "mean_module.log_amplitude": {
                    "value": True,
                    "constraint": False,
                    "value_reason": None,
                    "constraint_reason": "constraint_unavailable",
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
                    "value_reason": None,
                    "constraint_reason": "constraint_unavailable",
                },
            },
        )

        self.assertAlmostEqual(
            float(model.covar_module.lengthscale.detach().cpu().view(-1)[0]),
            2.0,
        )

    def test_apply_constraint_then_value_keeps_estimated_gpytorch_value(self):
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
                    constraint=(0.1, 10.0),
                )
            ]
        )

        class Model:
            def __init__(self):
                self.covar_module = gpytorch.kernels.RBFKernel()

        model = Model()
        result = ParameterEstimateApplicator().apply(
            model=model,
            estimates=estimates,
        )

        self.assertTrue(result["covar_module.lengthscale"]["constraint"])
        self.assertTrue(result["covar_module.lengthscale"]["value"])
        self.assertAlmostEqual(
            float(model.covar_module.lengthscale.detach().cpu().view(-1)[0]),
            2.0,
            places=5,
        )

    def test_apply_value_clips_gpytorch_endpoint_to_safe_interior(self):
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
                    value=0.1,
                    constraint=(0.1, 10.0),
                )
            ]
        )

        class Model:
            def __init__(self):
                self.covar_module = gpytorch.kernels.RBFKernel()

        model = Model()
        result = ParameterEstimateApplicator().apply(
            model=model,
            estimates=estimates,
        )

        value = float(model.covar_module.lengthscale.detach().cpu().view(-1)[0])
        self.assertTrue(result["covar_module.lengthscale"]["value"])
        self.assertGreater(value, 0.1)
        self.assertLess(value, 10.0)

    def test_apply_value_reports_shape_mismatch(self):
        class ModelWithVectorParameter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.target = torch.nn.Module()
                self.target.param = torch.nn.Parameter(
                    torch.zeros(3, 1, 2)
                )

        model = ModelWithVectorParameter()

        estimate = ParameterEstimate(
            spec=ParameterSpec(
                name="target.param",
                role=ParameterRole.FREQUENCY,
                domain=ParameterDomain.FREQUENCY,
                scale=ParameterScale.LINEAR,
            ),
            value=torch.tensor([1.0, 2.0, 3.0]),
            constraint=None,
            metadata={},
        )

        result = ParameterEstimateApplicator().apply(
            model=model,
            estimates=ParameterEstimateCollection([estimate]),
        )

        self.assertEqual(
            result,
            {
                "target.param": {
                    "value": False,
                    "constraint": False,
                    "value_reason": "shape_mismatch",
                    "constraint_reason": "constraint_unavailable",
                },
            },
        )

        self.assertEqual(
            tuple(estimate.metadata["expected_shape"]),
            (3, 1, 2),
        )
        self.assertEqual(
            tuple(estimate.metadata["actual_shape"]),
            (3,),
        )

    def test_apply_value_reshapes_when_element_count_matches(self):
        class ModelWithVectorParameter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.target = torch.nn.Module()
                self.target.param = torch.nn.Parameter(
                    torch.zeros(3, 1, 2)
                )

        model = ModelWithVectorParameter()

        estimate = ParameterEstimate(
            spec=ParameterSpec(
                name="target.param",
                role=ParameterRole.FREQUENCY,
                domain=ParameterDomain.FREQUENCY,
                scale=ParameterScale.LINEAR,
            ),
            value=torch.arange(6.0),
            constraint=None,
            metadata={},
        )

        result = ParameterEstimateApplicator().apply(
            model=model,
            estimates=ParameterEstimateCollection([estimate]),
        )

        self.assertEqual(
            result,
            {
                "target.param": {
                    "value": True,
                    "constraint": False,
                    "value_reason": None,
                    "constraint_reason": "constraint_unavailable",
                },
            },
        )

        self.assertTrue(
            torch.equal(
                model.target.param.detach(),
                torch.arange(6.0).reshape(3, 1, 2),
            )
        )


class DummyMeanModule:
    def __init__(self):
        self.offset = 123.0


class DummyModel:
    def __init__(self):
        self.mean_module = DummyMeanModule()

    def test_default_constraint_keeps_tighter_existing_interval(self):
        spec = ParameterSpec(
            name="covar_module.lengthscale",
            role=ParameterRole.LENGTHSCALE,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LOG,
            constraint=(1.0e-6, 1.0e6),
            constraint_strategy=ConstraintStrategy.DEFAULT,
        )
        estimate = ParameterEstimate(spec=spec, constraint=(1.0e-6, 1.0e6))

        class Model:
            def __init__(self):
                self.covar_module = gpytorch.kernels.RBFKernel()
                self.covar_module.register_constraint(
                    "raw_lengthscale",
                    gpytorch.constraints.Interval(0.1, 10.0),
                )

        model = Model()
        result = ParameterEstimateApplicator().apply(
            model=model,
            estimates=ParameterEstimateCollection([estimate]),
        )

        constraint = model.covar_module.raw_lengthscale_constraint
        self.assertTrue(result["covar_module.lengthscale"]["constraint"])
        self.assertEqual(
            result["covar_module.lengthscale"]["constraint_action"],
            "kept_existing",
        )
        self.assertAlmostEqual(float(constraint.lower_bound), 0.1)
        self.assertAlmostEqual(float(constraint.upper_bound), 10.0)

    def test_default_constraint_tightens_one_sided_existing_constraint(self):
        spec = ParameterSpec(
            name="covar_module.lengthscale",
            role=ParameterRole.LENGTHSCALE,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LOG,
            constraint=(1.0e-6, 10.0),
            constraint_strategy=ConstraintStrategy.DEFAULT,
        )
        estimate = ParameterEstimate(spec=spec, constraint=(1.0e-6, 10.0))

        class Model:
            def __init__(self):
                self.covar_module = gpytorch.kernels.RBFKernel()
                self.covar_module.register_constraint(
                    "raw_lengthscale",
                    gpytorch.constraints.GreaterThan(0.1),
                )

        model = Model()
        result = ParameterEstimateApplicator().apply(
            model=model,
            estimates=ParameterEstimateCollection([estimate]),
        )

        constraint = model.covar_module.raw_lengthscale_constraint
        self.assertTrue(result["covar_module.lengthscale"]["constraint"])
        self.assertEqual(
            result["covar_module.lengthscale"]["constraint_action"],
            "tightened",
        )
        self.assertAlmostEqual(float(constraint.lower_bound), 0.1, places=5)
        self.assertAlmostEqual(float(constraint.upper_bound), 10.0, places=5)

    def test_default_constraint_conflict_keeps_existing_constraint(self):
        spec = ParameterSpec(
            name="covar_module.lengthscale",
            role=ParameterRole.LENGTHSCALE,
            domain=ParameterDomain.TIME,
            scale=ParameterScale.LOG,
            constraint=(20.0, 30.0),
            constraint_strategy=ConstraintStrategy.DEFAULT,
        )
        estimate = ParameterEstimate(spec=spec, constraint=(20.0, 30.0))

        class Model:
            def __init__(self):
                self.covar_module = gpytorch.kernels.RBFKernel()
                self.covar_module.register_constraint(
                    "raw_lengthscale",
                    gpytorch.constraints.Interval(0.1, 10.0),
                )

        model = Model()
        result = ParameterEstimateApplicator().apply(
            model=model,
            estimates=ParameterEstimateCollection([estimate]),
        )

        constraint = model.covar_module.raw_lengthscale_constraint
        self.assertTrue(result["covar_module.lengthscale"]["constraint"])
        self.assertEqual(
            result["covar_module.lengthscale"]["constraint_action"],
            "conflict_kept_existing",
        )
        self.assertAlmostEqual(float(constraint.lower_bound), 0.1)
        self.assertAlmostEqual(float(constraint.upper_bound), 10.0)

    def test_plain_parameter_constraint_is_reported_unenforceable(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
            constraint_strategy=ConstraintStrategy.ROBUST_FLUX_RANGE,
        )
        estimate = ParameterEstimate(
            spec=spec,
            value=5.0,
            constraint=(0.0, 10.0),
        )

        model = DummyTensorModel()
        result = ParameterEstimateApplicator().apply(
            model=model,
            estimates=ParameterEstimateCollection([estimate]),
        )

        self.assertEqual(
            result["mean_module.offset"]["constraint_reason"],
            "constraint_not_enforceable_plain_parameter",
        )
        self.assertFalse(result["mean_module.offset"]["constraint"])
        self.assertTrue(result["mean_module.offset"]["value"])
        self.assertAlmostEqual(float(model.mean_module.offset.item()), 5.0)

    def test_plain_parameter_value_is_clamped_to_requested_constraint(self):
        spec = ParameterSpec(
            name="mean_module.offset",
            role=ParameterRole.OFFSET,
            domain=ParameterDomain.FLUX,
            scale=ParameterScale.LINEAR,
            constraint_strategy=ConstraintStrategy.ROBUST_FLUX_RANGE,
        )
        estimate = ParameterEstimate(
            spec=spec,
            value=-100.0,
            constraint=(0.0, 10.0),
            metadata={
                "constraint_enforceability": "plain_parameter",
            },
        )

        model = DummyTensorModel()
        result = ParameterEstimateApplicator().apply(
            model=model,
            estimates=ParameterEstimateCollection([estimate]),
        )

        value = float(model.mean_module.offset.item())
        self.assertFalse(result["mean_module.offset"]["constraint"])
        self.assertEqual(
            result["mean_module.offset"]["constraint_reason"],
            "constraint_not_enforceable_plain_parameter",
        )
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 10.0)



if __name__ == "__main__":
    unittest.main()
