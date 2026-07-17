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
        cls.wavelength_models_text = (
            root / "docs/source/howto/wavelength_models.rst"
        ).read_text(encoding="utf-8")
        cls.consensus_text = (
            root / "docs/source/howto/consensus_fitting.rst"
        ).read_text(encoding="utf-8")

    def test_api_reference_includes_wavelength_estimation_module(self):
        self.assertIn("pgmuvi.wavelength_estimation", self.api_text)

    def test_module_page_states_raw_and_model_coordinate_application(self):
        self.assertIn("wavelength coordinate", self.module_text)
        self.assertIn("coordinate supplied to the GP", self.module_text)
        self.assertIn("Existing registered bounds are", self.module_text)
        self.assertIn("2DWavelengthDependent", self.module_text)
        self.assertIn("full non-separable ``2D``", self.module_text)
        self.assertIn("No logarithmic flux fitting", self.module_text)

    def test_model_guide_documents_application_provenance_and_scope(self):
        self.assertIn(
            "Data-derived wavelength covariance initialization",
            self.wavelength_models_text,
        )
        self.assertIn(
            "wavelength_estimate_provenance",
            self.wavelength_models_text,
        )
        self.assertIn("``2DSeparable``", self.wavelength_models_text)

    def test_consensus_guide_documents_independent_wavelength_handoff(self):
        self.assertIn(
            "consensus period handoff",
            self.consensus_text,
        )
        self.assertIn("quasi-periodic", self.consensus_text)

    def test_future_work_records_mean_completion_and_retains_ard_validation(self):
        self.assertIn(
            "Data-derived wavelength-mean initialization",
            self.future_work_text,
        )
        self.assertIn("temporal and wavelength ARD", self.future_work_text)
        self.assertIn("wavelength-constraint-validation", self.future_work_text)


if __name__ == "__main__":
    unittest.main()
