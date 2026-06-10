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


if __name__ == "__main__":
    unittest.main()