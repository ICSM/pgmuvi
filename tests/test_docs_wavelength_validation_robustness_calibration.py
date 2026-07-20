import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestSyntheticWavelengthRobustnessCalibrationDocumentation(
    unittest.TestCase
):
    def test_calibration_module_is_in_public_api(self):
        api = (ROOT / "docs/source/api.rst").read_text(encoding="utf-8")
        page = ROOT / (
            "docs/source/"
            "pgmuvi.wavelength_validation_robustness_calibration.rst"
        )

        self.assertIn(
            "pgmuvi.wavelength_validation_robustness_calibration", api
        )
        self.assertTrue(page.exists())
        text = page.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        self.assertIn(
            "automodule:: "
            "pgmuvi.wavelength_validation_robustness_calibration",
            text,
        )
        self.assertIn("truth-matched reference", normalized)
        self.assertIn("same base seed", normalized)
        self.assertIn("resume from those records", normalized)
        self.assertIn("do not reuse the D1 aggregate gates", normalized)
        self.assertIn("wavelength-frequency pressure", normalized)
        self.assertIn("warnings, not automatic recovery failures", normalized)
        self.assertIn("do not compare candidate models", normalized)

    def test_package_exports_calibration_module_name(self):
        package_init = (ROOT / "pgmuvi/__init__.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            '"wavelength_validation_robustness_calibration"', package_init
        )

    def test_constraint_notebook_is_completed_after_empirical_calibration(self):
        future = (ROOT / "docs/source/future_work.rst").read_text(
            encoding="utf-8"
        )
        status = (ROOT / "docs/source/notebook_status.rst").read_text(
            encoding="utf-8"
        )

        self.assertIn("TBD[wavelength-constraint-validation]", future)
        self.assertNotIn("TBD[wavelength-constraint-notebook]", status)
        self.assertIn("tutorial_wavelength_constraints.ipynb", status)
        normalized = " ".join(future.split())
        self.assertIn("canonical 20-seed D2 robustness calibration", normalized)
        self.assertIn("maintained wavelength-constraint notebook are complete", normalized)
        self.assertIn("D3 representative observed-LPV validation infrastructure", normalized)
        self.assertNotIn("Execute and document the full D2 population", future)


if __name__ == "__main__":
    unittest.main()
