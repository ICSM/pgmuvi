"""Documentation contract tests for wavelength-model hypothesis taxonomy."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestWavelengthHypothesisDocumentation(unittest.TestCase):
    def test_api_reference_includes_hypothesis_module(self):
        api = (ROOT / "docs/source/api.rst").read_text()
        module_page = ROOT / "docs/source/pgmuvi.wavelength_hypotheses.rst"
        self.assertIn("pgmuvi.wavelength_hypotheses", api)
        self.assertTrue(module_page.exists())
        self.assertIn(
            "automodule:: pgmuvi.wavelength_hypotheses",
            module_page.read_text(),
        )

    def test_advisory_docs_distinguish_mean_and_covariance_roles(self):
        advisory = (
            ROOT / "docs/source/howto/wavelength_advisory.rst"
        ).read_text()
        interpretation = (
            ROOT / "docs/source/howto/interpreting_results.rst"
        ).read_text()
        for text in (advisory, interpretation):
            self.assertIn("model_hypothesis", text)
            self.assertIn("2DDustMean", text)
            self.assertIn("2DPowerLawMean", text)
            self.assertIn("mean-only", text)
        self.assertIn("2DAchromatic", advisory)
        self.assertIn("not part of that default LPV priority", advisory)


if __name__ == "__main__":
    unittest.main()
