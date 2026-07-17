from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestWavelengthMeanEstimateDocumentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = (
            ROOT / "docs/source/pgmuvi.wavelength_estimation.rst"
        ).read_text(encoding="utf-8")
        cls.models = (
            ROOT / "docs/source/howto/wavelength_models.rst"
        ).read_text(encoding="utf-8")
        cls.constraints = (
            ROOT / "docs/source/howto/priors_constraints.rst"
        ).read_text(encoding="utf-8")
        cls.future = (ROOT / "docs/source/future_work.rst").read_text(
            encoding="utf-8"
        )
        cls.module_normalized = " ".join(cls.module.split())
        cls.constraints_normalized = " ".join(cls.constraints.split())
        cls.future_normalized = " ".join(cls.future.split())

    def test_module_documents_mean_strategies_and_coordinate_bases(self):
        for token in (
            "GuessStrategy.WAVELENGTH_MEAN",
            "ConstraintStrategy.WAVELENGTH_MEAN",
            "reconstructed physical",
            "training-target coordinate",
            "registered GPyTorch raw-parameter interval",
            "No logarithmic flux fitting",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.module_normalized)

    def test_model_guide_documents_all_target_mean_families(self):
        for token in (
            "Data-derived wavelength mean initialization",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            "wavelength_mean_estimate_provenance",
            "custom non-affine wavelength transform is rejected",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.models)

    def test_constraints_are_documented_as_enforced(self):
        self.assertIn(
            "registered GPyTorch raw-parameter constraints",
            self.constraints_normalized,
        )
        self.assertIn("enforced during", self.constraints_normalized)
        self.assertIn("optimization", self.constraints_normalized)

    def test_full_2d_ard_and_validation_remain_future_work(self):
        self.assertIn("Data-derived wavelength-mean initialization", self.future_normalized)
        self.assertIn("full ``2D``", self.future_normalized)
        self.assertIn("temporal and wavelength ARD dimensions", self.future_normalized)
        self.assertIn("TBD[wavelength-constraint-validation]", self.future_normalized)


if __name__ == "__main__":
    unittest.main()
