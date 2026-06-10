import unittest

from pgmuvi.parameter_application import ParameterEstimateApplicator
from pgmuvi.parameter_estimates import ParameterEstimateCollection


class TestParameterEstimateApplicator(unittest.TestCase):

    def test_apply_is_not_implemented_initially(self):
        applicator = ParameterEstimateApplicator()

        with self.assertRaises(NotImplementedError):
            applicator.apply(
                model=object(),
                estimates=ParameterEstimateCollection(),
            )

    def test_resolve_parameter(self):
        applicator = ParameterEstimateApplicator()

        model = DummyModel()

        result = applicator._resolve_parameter(
            model,
            "mean_module.offset",
        )

        self.assertEqual(result, 123.0)


class DummyMeanModule:
    def __init__(self):
        self.offset = 123.0


class DummyModel:
    def __init__(self):
        self.mean_module = DummyMeanModule()


if __name__ == "__main__":
    unittest.main()
