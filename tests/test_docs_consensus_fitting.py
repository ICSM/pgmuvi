from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestConsensusFittingDocumentation(unittest.TestCase):
    def _read(self, relative_path: str) -> str:
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_howto_index_links_consensus_fitting(self):
        text = self._read("docs/source/howto/index.rst")
        self.assertIn("consensus_fitting", text)

    def test_consensus_howto_documents_current_fit_call(self):
        text = self._read("docs/source/howto/consensus_fitting.rst")
        required = [
            "fit_strategy=\"consensus\"",
            "learn_additional_noise=True",
            "ConsensusFitError",
            "band=",
            "accepted bands",
            "rejected bands",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            "time_kernel_type=\"quasi_periodic\"",
            "run_period_independent_wavelength_advisory_workflow",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_readme_points_to_consensus_fitting_docs(self):
        text = self._read("README.md")
        required = [
            "fit_strategy=\"consensus\"",
            "learn_additional_noise=True",
            "docs/source/howto/consensus_fitting.rst",
            "examples/consensus_multiband_fit.py",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_example_script_contains_flagship_call(self):
        text = self._read("examples/consensus_multiband_fit.py")
        required = [
            "fit_strategy",
            "consensus",
            "learn_additional_noise",
            "Lightcurve.from_csv",
            "band=",
            "get_period_summary",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_glossary_has_consensus_terms(self):
        text = self._read("docs/source/glossary.rst")
        required = [
            "Consensus fitting",
            "Consensus period",
            "ConsensusFitError",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)


class TestConsensusFittingPeriodSummaryDocumentation(unittest.TestCase):
    def _read(self, relative_path: str) -> str:
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_consensus_howto_documents_period_summary_result_access(self):
        text = self._read("docs/source/howto/consensus_fitting.rst")
        required = [
            "PeriodSummaryResult",
            "dominant_period",
            "dominant_frequency",
            "get_primary_peak()",
            "as_dict()",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("print(period_summary)", text)
        self.assertNotIn("print(summary)", text)

    def test_consensus_example_formats_period_summary_result(self):
        text = self._read("examples/consensus_multiband_fit.py")
        required = [
            "print_period_summary",
            "get_primary_peak",
            "as_dict",
            "dominant_period",
            "dominant_frequency",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("print(period_summary)", text)


if __name__ == "__main__":
    unittest.main()
