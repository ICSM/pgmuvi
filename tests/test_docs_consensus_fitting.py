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

    def test_guide_covers_baseline_and_consensus_decision_path(self):
        text = self._read("docs/source/howto/consensus_fitting.rst")
        required = [
            'model="1D"',
            'model="2D"',
            'fit_strategy="consensus"',
            "direct fit",
            "2DSeparable",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            'time_kernel_type="quasi_periodic"',
            'time_kernel_type="spectral_mixture"',
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("2DAchromatic", text)

    def test_guide_documents_validation_noise_and_numerical_defaults(self):
        text = self._read("docs/source/howto/consensus_fitting.rst")
        required = [
            "band=",
            "accepted and rejected bands",
            "variance=False",
            "learn_additional_noise=True",
            "torch.float64",
            'center_time="auto"',
            "use_parameter_workflow=True",
            'constraint_set="LPV"',
            "training_iter=500",
            "miniter=100",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_guide_documents_two_band_frequency_agreement_guard(self):
        text = self._read("docs/source/howto/consensus_fitting.rst")
        required = [
            "two_band_max_fractional_frequency_difference",
            "default ``0.10``",
            "midpoint period supported by neither band",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_guide_documents_failure_and_multicomponent_limits(self):
        text = self._read("docs/source/howto/consensus_fitting.rst")
        required = [
            "ConsensusFitError",
            "failure_diagnostics",
            "failure_summary",
            'fit_strategy="consensus_multicomp"',
            "component-specific constraints",
            "TBD: multi-periodic support",
            ":doc:`../notebooks/pgmuvi_tutorial_2d`",
            "run_period_independent_wavelength_advisory_workflow",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_guide_documents_example_execution_and_artifacts(self):
        text = self._read("docs/source/howto/consensus_fitting.rst")
        required = [
            "examples/consensus_multiband_fit.py",
            "--dry-run",
            "--output-dir",
            "run_configuration.json",
            "period_summary.json",
            "consensus_diagnostics.json",
            "fit_history.json",
            "failure.json",
            "exit status 2",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_readme_points_to_consensus_fitting_docs(self):
        text = self._read("README.md")
        required = [
            'fit_strategy="consensus"',
            "learn_additional_noise=True",
            "docs/source/howto/consensus_fitting.rst",
            "examples/consensus_multiband_fit.py",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_example_script_contains_maintained_execution_contract(self):
        text = self._read("examples/consensus_multiband_fit.py")
        required = [
            "ConsensusFitError",
            "validate_configuration",
            "build_fit_kwargs",
            "LPV_CONFIGURABLE_MODELS",
            "--dry-run",
            "--output-dir",
            "run_configuration.json",
            "period_summary.json",
            "failure.json",
            "return 2",
            "get_period_summary",
            "get_fit_history",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("2DAchromatic", text)

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
