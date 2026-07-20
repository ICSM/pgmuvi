"""Regression checks for the light-curve input and validation guide."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "source" / "howto" / "loading_data.rst"
EXAMPLE = ROOT / "examples" / "validate_lightcurve_input.py"


class TestLightcurveValidationDocs(unittest.TestCase):
    def setUp(self):
        self.guide = GUIDE.read_text(encoding="utf-8")

    def test_guide_and_runnable_example_exist(self):
        self.assertTrue(GUIDE.is_file())
        self.assertTrue(EXAMPLE.is_file())
        self.assertIn("Runnable validation example", self.guide)
        self.assertIn("examples/validate_lightcurve_input.py", self.guide)

    def test_csv_contract_and_model_coordinates_are_explicit(self):
        required = [
            "CSV input contract",
            "Auto-detected CSV columns",
            "Numeric wavelength",
            "Observational-channel label",
            "shape ``(N, 2)``",
            "arbitrary numeric indices",
            "actual numeric wavelengths",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.guide)

    def test_finite_and_positive_value_policies_are_distinguished(self):
        required = [
            "Finite-value filtering",
            "Positive fluxes and uncertainties",
            "universal positive-flux rule",
            "Non-positive uncertainty values",
            "wavelength-advisory batch workflow applies stricter defaults",
            "--drop-nonpositive-errors",
            "--drop-nonpositive-flux",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.guide)

    def test_magnitude_cross_reference_target_and_conversion_are_present(self):
        required = [
            ".. _working-with-magnitudes:",
            "Working with magnitudes",
            "Native magnitude support is not currently implemented",
            "reference_magnitude = np.nanmedian(magnitude)",
            "flux_error = (np.log(10.0) / 2.5)",
            "smaller magnitudes must map to larger fluxes",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.guide)

    def test_sampling_subsampling_and_variability_behaviour_are_protected(self):
        required = [
            "Sampling checks and band removal",
            "Default subsampling behaviour",
            "max_samples=1000",
            "max_samples_per_band",
            "subsample_seed",
            "check_variability=True",
            "check_variability_per_band()",
            "filter_variable_bands()",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.guide)

    def test_common_warnings_and_notebook_tutorial_are_present(self):
        required = [
            "Common warnings and what they mean",
            "Fewer than 10 elements remain",
            "Skipping band",
            "median_cadence is zero",
            "Notebook tutorial",
            ":doc:`../notebooks/tutorial_preprocessing`",
            "non-finite-row handling",
            "reproducible subsampling",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.guide)
        self.assertNotIn("TBD[notebook-lightcurve-validation]", self.guide)


if __name__ == "__main__":
    unittest.main()
