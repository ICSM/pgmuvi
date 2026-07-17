from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestSpectralMixtureArdDocumentation(unittest.TestCase):
    def test_api_reference_includes_module(self):
        api = (ROOT / "docs/source/api.rst").read_text()
        self.assertIn("pgmuvi.spectral_mixture_ard", api)
        self.assertTrue(
            (ROOT / "docs/source/pgmuvi.spectral_mixture_ard.rst").exists()
        )

    def test_module_page_documents_coordinate_order(self):
        text = (
            ROOT / "docs/source/pgmuvi.spectral_mixture_ard.rst"
        ).read_text()
        normalized = " ".join(text.split())
        self.assertIn("index 0: temporal frequency", normalized)
        self.assertIn("index 1: wavelength frequency", normalized)
        self.assertIn("(1, 1, 2)", normalized)

    def test_constraint_guide_documents_tensor_intervals(self):
        text = (
            ROOT / "docs/source/howto/priors_constraints.rst"
        ).read_text()
        normalized = " ".join(text.split())
        self.assertIn("broadcast across components", normalized)
        self.assertIn("Source-type period limits modify only the temporal entry", normalized)
        self.assertIn("Consensus fitting also respects this separation", normalized)
        self.assertIn("wavelength-frequency bounds at ARD index 1 remain unchanged", normalized)

    def test_consensus_guide_documents_temporal_only_interval(self):
        text = (ROOT / "docs/source/howto/consensus_fitting.rst").read_text()
        normalized = " ".join(text.split())
        self.assertIn("temporal-frequency** interval", normalized)
        self.assertIn("updates only ARD index 0", normalized)
        self.assertIn("wavelength-frequency bounds at ARD index 1 are preserved", normalized)

    def test_future_work_keeps_saturation_validation_scope(self):
        text = (ROOT / "docs/source/future_work.rst").read_text()
        normalized = " ".join(text.split())
        self.assertIn("component/dimension saturation", normalized)
        self.assertIn("scientifically long wavelength correlation scale", normalized)
        self.assertIn("TBD[wavelength-constraint-validation]", text)


if __name__ == "__main__":
    unittest.main()
