"""Regression checks for the PR101 batch advisory walkthrough docs."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BATCH_DOC = ROOT / "docs/source/howto/wavelength_advisory_batch.rst"
WALKTHROUGH = ROOT / "examples/wavelength_advisory_batch_walkthrough.py"


class TestBatchAdvisoryWalkthroughDocs(unittest.TestCase):
    def setUp(self):
        self.text = BATCH_DOC.read_text(encoding="utf-8")

    def test_batch_doc_is_marked_current_through_pr101(self):
        self.assertIn("current through PR101", self.text)

    def test_batch_doc_links_toy_walkthrough(self):
        required = [
            "examples/wavelength_advisory_batch_walkthrough.py",
            "Toy batch walkthrough",
            "--run-batch",
            "--model-kernel-config-limit 1",
            "PREPARE ONLY",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.text)

    def test_batch_doc_covers_supported_input_layouts(self):
        required = [
            "Direct command-line tokens",
            "Newline-delimited text",
            "Headered CSV",
            "JSON list",
            "source_id=csv_path",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.text)

    def test_batch_doc_explains_output_tree_and_triage_order(self):
        required = [
            "Output directory anatomy",
            "toy_batch_summary.json",
            "toy_batch_model_kernel_configs.csv",
            "toy_batch_model_kernel_config_summary.csv",
            "toy_batch_report.md",
            "Triage failures in this order",
            "source-level failure",
            "per-config failure",
            "all-config fallback",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.text)

    def test_batch_doc_explains_exit_status_policy(self):
        required = [
            "Exit status and continuation policy",
            "--stop-on-error",
            "--allow-source-failures",
            "status 1",
            "status 0",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.text)

    def test_batch_doc_keeps_advisory_boundary_and_future_markers(self):
        required = [
            "do not validate scientific",
            "automatic_model_selection_applied",
            "selected_model",
            "TBD[batch-validation]",
            "TBD[batch-notebook]",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.text)

    def test_walkthrough_script_exists(self):
        self.assertTrue(WALKTHROUGH.is_file())


if __name__ == "__main__":
    unittest.main()
