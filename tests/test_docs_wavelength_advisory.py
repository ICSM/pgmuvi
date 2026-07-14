"""Documentation coverage checks for wavelength advisory workflows."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestWavelengthAdvisoryDocs(unittest.TestCase):
    def _read(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_single_source_advisory_doc_mentions_public_methods(self):
        text = self._read("docs/source/howto/wavelength_advisory.rst")
        required = [
            "diagnose_period_independent_wavelength_structure",
            "build_period_independent_wavelength_parameter_plan",
            "build_period_independent_wavelength_model_kernel_configs",
            "run_period_independent_wavelength_model_kernel_configs",
            "score_period_independent_wavelength_model_kernel_config_quality",
            "run_period_independent_wavelength_advisory_workflow",
            "export_period_independent_wavelength_advisory_workflow",
        ]
        for name in required:
            with self.subTest(name=name):
                self.assertIn(name, text)

    def test_batch_doc_mentions_output_artifacts_and_schema_terms(self):
        text = self._read("docs/source/howto/wavelength_advisory_batch.rst")
        required = [
            "run_period_independent_wavelength_advisory_workflow_batch",
            "examples/run_wavelength_advisory_batch.py",
            "<prefix>_summary.json",
            "<prefix>_summary.csv",
            "<prefix>_model_kernel_configs.csv",
            "<prefix>_model_kernel_config_summary.csv",
            "<prefix>_report.md",
            "n_model_kernel_configs",
            "n_successful_model_kernel_configs",
            "n_failed_model_kernel_configs",
            "top_ranked_fraction",
            "positive_data_filter_kwargs",
            "n_rows_dropped_positive_filter",
            "source_output_dir",
            "source_output_prefix",
            "--allow-nonpositive-flux",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_howto_index_links_current_advisory_pages(self):
        text = self._read("docs/source/howto/index.rst")
        self.assertIn("wavelength_advisory", text)
        self.assertIn("wavelength_advisory_batch", text)

    def test_readme_points_to_current_advisory_docs(self):
        text = self._read("README.md")
        required = [
            "docs/source/howto/wavelength_advisory.rst",
            "docs/source/howto/wavelength_advisory_batch.rst",
            "model/kernel",
            "run_period_independent_wavelength_advisory_workflow",
            "run_period_independent_wavelength_advisory_workflow_batch",
            "n_model_kernel_configs",
            "advisory_only",
            "selected_model",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_glossary_contains_current_advisory_terms(self):
        text = self._read("docs/source/glossary.rst")
        required = [
            "Model/kernel config",
            "Advisory workflow",
            "Top-ranked model",
            "Selected model",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
