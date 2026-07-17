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
            "raw_half_amplitude_q02_5_q97_5",
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
            "fit_quality_ranking_status",
            "fit_quality_ranking_available",
            "only_valid_model",
            "n_sources_with_comparative_ranking",
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


    def test_docs_explain_ranking_availability_and_single_candidate_semantics(self):
        single = self._read("docs/source/howto/wavelength_advisory.rst")
        batch = self._read("docs/source/howto/wavelength_advisory_batch.rst")

        for token in [
            "fit_quality_ranking_status",
            "single_valid_candidate",
            "only_valid_model",
            "at least two candidates",
        ]:
            with self.subTest(document="single", token=token):
                self.assertIn(token, single)

        for token in [
            "fit_quality_ranking_available",
            "n_sources_with_fit_quality",
            "n_sources_with_comparative_ranking",
            "n_top_ranked_sources / n_sources_with_comparative_ranking",
        ]:
            with self.subTest(document="batch", token=token):
                self.assertIn(token, batch)

    def test_docs_explain_retained_state_marginal_likelihood(self):
        single = self._read("docs/source/howto/wavelength_advisory.rst")
        batch = self._read("docs/source/howto/wavelength_advisory_batch.rst")
        single = " ".join(single.split())
        batch = " ".join(batch.split())

        for token in [
            "training_log_marginal_likelihood",
            "data log marginal likelihood per observation",
            "excluding registered priors",
            "training_map_objective",
            "retained_current_state",
        ]:
            with self.subTest(document="single", token=token):
                self.assertIn(token, single)

        for token in [
            "training_log_marginal_likelihood_total",
            "training_map_objective_total",
            "training_registered_log_prior_total",
            "training_marginal_likelihood_evaluation_mode",
            "retained_current_state",
        ]:
            with self.subTest(document="batch", token=token):
                self.assertIn(token, batch)

    def test_docs_explain_training_residual_noise_semantics(self):
        single = self._read("docs/source/howto/wavelength_advisory.rst")
        batch = self._read("docs/source/howto/wavelength_advisory_batch.rst")
        single = " ".join(single.split())
        batch = " ".join(batch.split())

        for token in [
            "likelihood(model(x_train))",
            "not added to it a second time",
            "predictive variance is unavailable",
        ]:
            with self.subTest(document="single", token=token):
                self.assertIn(token, single)

        for token in [
            "training_standardization_sigma_source",
            "already includes likelihood noise",
            "not combined with ``yerr`` again",
        ]:
            with self.subTest(document="batch", token=token):
                self.assertIn(token, batch)

    def test_single_source_doc_describes_retained_high_level_reports_truthfully(self):
        text = self._read("docs/source/howto/wavelength_advisory.rst")
        self.assertIn(
            "does **not** preserve the complete period-independent diagnostic report",
            text,
        )
        self.assertIn("parameter-plan report", text)
        self.assertNotIn(
            "The high-level helper returns all of these pieces in one dictionary",
            text,
        )

    def test_batch_doc_does_not_call_rank_labels_stable_configuration_ids(self):
        text = self._read("docs/source/howto/wavelength_advisory_batch.rst")
        self.assertIn("rank-dependent evaluation label", text)
        self.assertIn("not a stable configuration identity", text)
        self.assertNotIn(
            "Stable row identifier such as ``rank1_2DWavelengthDependent``",
            text,
        )

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
