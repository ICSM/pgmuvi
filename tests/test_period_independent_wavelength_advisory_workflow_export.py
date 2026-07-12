import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.wavelength_diagnostics import (
    export_period_independent_wavelength_advisory_workflow,
)


def _workflow_with_outputs():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 2])
    return {
        "kind": "period_independent_wavelength_advisory_workflow",
        "advisory_only": True,
        "runs_fits": True,
        "candidate_fit_state_isolated": True,
        "mutates_input_lightcurve": False,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "automatic_constraints_applied": False,
        "automatic_initialization_applied": False,
        "top_ranked_model": "2DDustMean",
        "score_kind": "training_residual_fit_quality",
        "text_report": "Workflow-level report\nworkflow_runs_candidate_fits: True\n",
        "comparison_text_report": "Nested comparison report\nquality_score_report_runs_fits: False\n",
        "figures": {"quality_scores": fig},
    }


class TestPeriodIndependentWavelengthAdvisoryWorkflowExport(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_export_writes_json_text_comparison_and_figures(self):
        workflow = _workflow_with_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = export_period_independent_wavelength_advisory_workflow(
                workflow,
                tmp,
                prefix="source_a",
            )

            self.assertEqual(
                manifest["kind"],
                "period_independent_wavelength_advisory_workflow_export",
            )
            self.assertTrue(manifest["wrote_json"])
            self.assertTrue(manifest["wrote_text_report"])
            self.assertTrue(manifest["wrote_comparison_text_report"])
            self.assertTrue(manifest["wrote_figures"])

            json_path = Path(manifest["json_path"])
            text_path = Path(manifest["text_report_path"])
            comparison_path = Path(manifest["comparison_text_report_path"])
            figure_path = Path(manifest["figure_paths"]["quality_scores"])

            self.assertTrue(json_path.exists())
            self.assertTrue(text_path.exists())
            self.assertTrue(comparison_path.exists())
            self.assertTrue(figure_path.exists())

            payload = json.loads(json_path.read_text())
            self.assertNotIn("figures", payload)
            self.assertEqual(payload["top_ranked_model"], "2DDustMean")
            self.assertIn("workflow_runs_candidate_fits", text_path.read_text())
            self.assertIn("quality_score_report_runs_fits", comparison_path.read_text())

    def test_export_manifest_is_advisory_and_does_not_run_fits(self):
        workflow = _workflow_with_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = export_period_independent_wavelength_advisory_workflow(
                workflow,
                tmp,
                save_figures=False,
            )

        self.assertTrue(manifest["advisory_only"])
        self.assertFalse(manifest["runs_fits"])
        self.assertFalse(manifest["applies_to_fit"])
        self.assertFalse(manifest["mutates_input_lightcurve"])
        self.assertFalse(manifest["automatic_model_selection_applied"])
        self.assertIsNone(manifest["selected_model"])
        self.assertFalse(manifest["automatic_constraints_applied"])
        self.assertFalse(manifest["automatic_initialization_applied"])

    def test_export_can_disable_json_text_and_figures(self):
        workflow = _workflow_with_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = export_period_independent_wavelength_advisory_workflow(
                workflow,
                tmp,
                save_json=False,
                save_text=False,
                save_figures=False,
            )
            self.assertFalse(manifest["wrote_json"])
            self.assertFalse(manifest["wrote_text_report"])
            self.assertFalse(manifest["wrote_figures"])
            self.assertEqual(manifest["exported_files"], [])

    def test_rejects_non_workflow_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "advisory workflow"):
                export_period_independent_wavelength_advisory_workflow(
                    {"kind": "period_independent_wavelength_fit_candidate_scores"},
                    tmp,
                )

    def test_lightcurve_method_delegates_for_precomputed_workflow(self):
        lc = Lightcurve(
            xdata=np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 2.0], [1.0, 2.0]]),
            ydata=np.array([1.0, 1.1, 2.0, 2.1]),
            yerr=np.array([0.1, 0.1, 0.1, 0.1]),
        )
        workflow = _workflow_with_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = lc.export_period_independent_wavelength_advisory_workflow(
                workflow=workflow,
                output_dir=tmp,
                prefix="lc_export",
                save_figures=False,
            )
            self.assertTrue(Path(manifest["json_path"]).exists())
            self.assertTrue(Path(manifest["text_report_path"]).exists())


if __name__ == "__main__":
    unittest.main()
