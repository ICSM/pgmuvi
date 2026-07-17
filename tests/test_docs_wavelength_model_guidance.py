"""Regression checks for wavelength-dependent model-family guidance."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestWavelengthModelGuidanceDocs(unittest.TestCase):
    def _read(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def _read_normalized(self, relative_path):
        return " ".join(self._read(relative_path).split())

    def test_howto_index_links_model_guidance(self):
        text = self._read("docs/source/howto/index.rst")
        self.assertIn("wavelength_models", text)

    def test_page_covers_expected_model_families_without_prioritizing_achromatic(self):
        text = self._read("docs/source/howto/wavelength_models.rst")
        for model in [
            "2D",
            "2DSeparable",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
        ]:
            with self.subTest(model=model):
                self.assertIn(model, text)
        self.assertNotIn("2DAchromatic", text)

    def test_page_distinguishes_mean_and_covariance(self):
        text = self._read("docs/source/howto/wavelength_models.rst")
        required = [
            "mean function",
            "covariance function",
            "k_time(t,t') * k_wavelength(lambda,lambda')",
            "amplitude * exp(-tau * lambda**(-alpha)) + offset",
            "offset + weight * lambda**exponent",
            "Non-separable two-dimensional spectral-mixture kernel",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_page_documents_coordinate_and_transform_cautions(self):
        text = self._read("docs/source/howto/wavelength_models.rst")
        required = [
            "physical, strictly positive wavelengths",
            "Integer band codes",
            "coordinate system seen by the GP",
            "wavelength transform",
            "raw-micron physical quantities",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_page_documents_try_first_and_consensus_guidance(self):
        text = self._read_normalized("docs/source/howto/wavelength_models.rst")
        required = [
            "What should I try first?",
            'time_kernel_type="quasi_periodic"',
            'time_kernel_type="spectral_mixture"',
            'fit_strategy="consensus"',
            'constraint_set="LPV"',
            "period_length",
            "independent light-curve instances",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_page_preserves_advisory_non_selection_contract(self):
        text = self._read_normalized("docs/source/howto/wavelength_models.rst")
        required = [
            "advisory_only=True",
            "selected_model=None",
            "training-space residual",
            "not held-out predictive performance",
            "does not make the advisory ranking automatic model selection",
            "triage rules, not statistical model selection",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_page_marks_known_future_work(self):
        text = self._read("docs/source/howto/wavelength_models.rst")
        for marker in [
            "TBD[multi-periodic-wavelength-models]",
            "TBD[non-monotonic-wavelength-kernels]",
            "TBD[automatic-model-selection]",
            "TBD[physical-wavelength-kernels]",
            "TBD[wavelength-dependent-lags]",
        ]:
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_related_guides_link_to_model_guidance(self):
        for relative_path in [
            "docs/source/howto/consensus_fitting.rst",
            "docs/source/howto/model_selection.rst",
            "docs/source/howto/wavelength_advisory.rst",
            "docs/source/howto/legacy_wavelength_candidates.rst",
        ]:
            text = self._read(relative_path)
            with self.subTest(path=relative_path):
                self.assertIn("wavelength_models", text)


if __name__ == "__main__":
    unittest.main()
