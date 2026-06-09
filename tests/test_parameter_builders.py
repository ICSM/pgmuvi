import unittest

from pgmuvi.parameter_builders import ParameterEstimateBuilder
from pgmuvi.parameter_context import ParameterEstimationContext
from pgmuvi.parameter_specs import (
    ParameterDomain,
    ParameterRole,
    ParameterSpec,
    ParameterSpecCollection,
)


class TestParameterEstimateBuilder(unittest.TestCase):

    def test_build_returns_estimate_collection(self):
        schema = ParameterSpecCollection(
            [
                ParameterSpec(
                    name="mean_module.offset",
                    role=ParameterRole.OFFSET,
                    domain=ParameterDomain.FLUX,
                ),
                ParameterSpec(
                    name="mean_module.log_amplitude",
                    role=ParameterRole.AMPLITUDE,
                    domain=ParameterDomain.FLUX,
                ),
            ]
        )

        context = ParameterEstimationContext(
            is_multiband=False,
        )

        builder = ParameterEstimateBuilder()

        estimates = builder.build(
            schema=schema,
            context=context,
        )

        self.assertEqual(
            estimates.names(),
            [
                "mean_module.offset",
                "mean_module.log_amplitude",
            ],
        )

        self.assertIsNone(
            estimates["mean_module.offset"].value
        )

        self.assertIsNone(
            estimates["mean_module.log_amplitude"].value
        )


if __name__ == "__main__":
    unittest.main()