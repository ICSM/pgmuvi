import unittest
from pathlib import Path


class TestWavelengthEstimationDocumentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.api_text = (root / "docs/source/api.rst").read_text(
            encoding="utf-8"
        )
        cls.module_text = (
            root / "docs/source/pgmuvi.wavelength_estimation.rst"
        ).read_text(encoding="utf-8")
        cls.future_work_text = (
            root / "docs/source/future_work.rst"
        ).read_text(encoding="utf-8")

    def test_api_reference_includes_wavelength_estimation_module(self):
        self.assertIn("pgmuvi.wavelength_estimation", self.api_text)

    def test_module_page_states_raw_coordinate_and_no_application(self):
        self.assertIn("raw wavelength", self.module_text)
        self.assertIn("not applied to any GP model", self.module_text)
        self.assertIn("No logarithmic flux transformation", self.module_text)

    def test_future_work_retains_model_application_and_validation(self):
        self.assertIn("actual model-input coordinate", self.future_work_text)
        self.assertIn("2DWavelengthDependent", self.future_work_text)
        self.assertIn("wavelength-constraint-validation", self.future_work_text)


if __name__ == "__main__":
    unittest.main()
