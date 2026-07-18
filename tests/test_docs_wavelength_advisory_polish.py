"""Regression checks for the final wavelength advisory docs polish."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestWavelengthAdvisoryDocsPolish(unittest.TestCase):
    def _read(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_docs_status_markers_track_the_current_documented_scope(self):
        single_text = self._read("docs/source/howto/wavelength_advisory.rst")
        batch_text = self._read("docs/source/howto/wavelength_advisory_batch.rst")

        self.assertIn("current through PR94", single_text)
        self.assertIn("current through PR101", batch_text)
        self.assertNotIn("current through PR71", single_text)
        self.assertNotIn("current through PR71", batch_text)

    def test_single_source_doc_has_coherent_current_workflow_map(self):
        text = self._read("docs/source/howto/wavelength_advisory.rst")
        required = [
            "Current workflow map",
            "Period-independent diagnostics",
            "Model/kernel-config runs",
            "Advisory ranking",
            "Failure fallback diagnostics",
            "raw_half_amplitude_q02_5_q97_5",
            "fallback_diagnostics_available",
            "fallback_report",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_batch_doc_integrates_filtering_outputs_failures_and_ard_fields(self):
        text = self._read("docs/source/howto/wavelength_advisory_batch.rst")
        required = [
            "Input hygiene and per-source outputs",
            "n_rows_before_positive_filter",
            "n_rows_dropped_positive_filter",
            "source_output_dir",
            "source_output_prefix",
            "failure_stage",
            "failure_stage_reason",
            "is_consensus_failure",
            "is_numerical_failure",
            "is_input_validation_failure",
            "n_sm_ard_boundary_hits",
            "sm_ard_boundary_pressure_scope",
            "n_sm_temporal_boundary_components",
            "n_sm_wavelength_boundary_components",
            "sm_ard_boundary_hits",
            "sm_num_mixtures_fixed_at_one",
            "n_constrained_sm_ard_components",
            "n_constrained_sm_time_components",
            "n_constrained_sm_wavelength_components",
            "constrained_sm_ard_components",
            "fallback_diagnostics_available",
            "fallback_report",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_batch_doc_no_longer_has_duplicate_failure_artifact_section(self):
        text = self._read("docs/source/howto/wavelength_advisory_batch.rst")
        self.assertEqual(text.count("Per-source failure artifacts"), 0)
        self.assertEqual(text.count("Failure handling"), 1)
        self.assertEqual(text.count("Advisory failure fallback reporting"), 1)


if __name__ == "__main__":
    unittest.main()
