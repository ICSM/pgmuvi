import unittest

from pgmuvi.parameter_workflow import get_parameter_schema


class ModelWithSchema:
    def parameter_schema(self):
        return "schema"


class ModelWithoutSchema:
    pass


class TestParameterWorkflow(unittest.TestCase):

    def test_returns_schema_when_available(self):
        self.assertEqual(
            get_parameter_schema(ModelWithSchema()),
            "schema",
        )

    def test_returns_none_when_unavailable(self):
        self.assertIsNone(
            get_parameter_schema(ModelWithoutSchema())
        )


if __name__ == "__main__":
    unittest.main()