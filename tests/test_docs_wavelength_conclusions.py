"""Documentation contracts for wavelength-advisory conclusions."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestWavelengthConclusionDocumentation(unittest.TestCase):
    def test_api_reference_includes_conclusion_module(self):
        api = (ROOT / "docs/source/api.rst").read_text()
        module_page = ROOT / "docs/source/pgmuvi.wavelength_conclusions.rst"
        self.assertIn("pgmuvi.wavelength_conclusions", api)
        self.assertTrue(module_page.exists())
        self.assertIn(
            "automodule:: pgmuvi.wavelength_conclusions",
            module_page.read_text(),
        )


    def test_future_work_tracks_constraint_and_validation_tranche(self):
        future_work = (ROOT / "docs/source/future_work.rst").read_text()
        advisory = (ROOT / "docs/source/howto/wavelength_advisory.rst").read_text()
        for marker in (
            "TBD[wavelength-derived-constraints]",
            "TBD[wavelength-constraint-validation]",
        ):
            self.assertIn(marker, future_work)
            self.assertIn(marker, advisory)
        self.assertIn("2DWavelengthDependent", future_work)
        self.assertIn("2DDustMean", future_work)
        self.assertIn("2DPowerLawMean", future_work)
        self.assertIn("2DSeparable", future_work)
        self.assertIn("temporal and wavelength ARD dimensions", future_work)

    def test_advisory_docs_define_conclusions_and_ambiguities(self):
        advisory = (ROOT / "docs/source/howto/wavelength_advisory.rst").read_text()
        interpretation = (
            ROOT / "docs/source/howto/interpreting_results.rst"
        ).read_text()
        for text in (advisory, interpretation):
            self.assertIn("advisory_conclusions", text)
            self.assertIn("unresolved_ambiguities", text)
            self.assertIn("remains_plausible", text)
            self.assertIn("technically_unevaluable", text)
            self.assertIn("scientifically_ambiguous", text)
            self.assertIn("incomparable", text)
            self.assertIn("training-residual", text)
            self.assertIn("not", text)
            self.assertIn("automatic model selection", text)


if __name__ == "__main__":
    unittest.main()
