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
import torch


class DummyTensorMeanModule:
    def __init__(self):
        self.offset = torch.nn.Parameter(torch.zeros(1))
        self.log_amplitude = torch.nn.Parameter(torch.zeros(1))


class DummyTensorModel:
    def __init__(self):
        self.mean_module = DummyTensorMeanModule()


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
            model.mean_module.offset,
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
            model.mean_module.log_amplitude,
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
            model.mean_module.offset,
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
                "mean_module.offset": True,
                "mean_module.log_amplitude": True,
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


class DummyMeanModule:
    def __init__(self):
        self.offset = 123.0


class DummyModel:
    def __init__(self):
        self.mean_module = DummyMeanModule()


if __name__ == "__main__":
    unittest.main()
