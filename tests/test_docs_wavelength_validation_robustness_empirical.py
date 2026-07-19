"""Documentation contract for the PR133 empirical D2 calibration."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestWavelengthRobustnessEmpiricalDocumentation(unittest.TestCase):
    def test_empirical_calibration_is_recorded(self):
        text = (
            ROOT
            / "docs"
            / "source"
            / "pgmuvi.wavelength_validation_robustness_calibration.rst"
        ).read_text(encoding="utf-8")

        self.assertIn("Twenty-seed empirical calibration", text)
        self.assertIn("420 canonical runs", text)
        self.assertIn("393 completed fits", text)
        self.assertIn("d2-uneven-band-counts", text)
        self.assertIn("d2-longer-sparse-baseline", text)
        self.assertIn("d2-large-wavelength-gap", text)
        self.assertIn("advisory evidence only", text)
        self.assertIn("validation_outputs/", text)

    def test_notebook_todo_retains_calibration_context(self):
        notebook_status = (
            ROOT / "docs" / "source" / "notebook_status.rst"
        ).read_text(encoding="utf-8")
        future_work = (
            ROOT / "docs" / "source" / "future_work.rst"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "TBD[wavelength-constraint-notebook]",
            notebook_status,
        )
        self.assertIn("canonical 20-seed", notebook_status)
        self.assertIn(
            "TBD[wavelength-constraint-notebook]",
            future_work,
        )
        self.assertIn("canonical 20-seed", future_work)

    def test_generated_validation_outputs_are_ignored(self):
        patterns = (
            ROOT / ".gitignore"
        ).read_text(encoding="utf-8").splitlines()
        self.assertIn("validation_outputs/", patterns)


if __name__ == "__main__":
    unittest.main()
