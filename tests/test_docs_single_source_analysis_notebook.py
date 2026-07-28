"""Regression contract for the complete real-source analysis notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "docs"
    / "source"
    / "notebooks"
    / "tutorial_single_source_analysis.ipynb"
)
REPRESENTATIVE_CSV = ROOT / "examples" / "data" / "10131+3049.csv"
INDEX = ROOT / "docs" / "source" / "index.rst"
STATUS = ROOT / "docs" / "source" / "notebook_status.rst"
FUTURE = ROOT / "docs" / "source" / "future_work.rst"
CONSENSUS = ROOT / "docs" / "source" / "howto" / "consensus_fitting.rst"
MODELS = ROOT / "docs" / "source" / "howto" / "wavelength_models.rst"
PRIORS = ROOT / "docs" / "source" / "howto" / "priors_constraints.rst"
INTERPRET = ROOT / "docs" / "source" / "howto" / "interpreting_results.rst"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells() -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in load_notebook().get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def notebook_text() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook().get("cells", [])
    )


class TestSingleSourceAnalysisNotebook(unittest.TestCase):
    def test_notebook_and_public_source_are_present_and_linked(self):
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(REPRESENTATIVE_CSV.is_file())
        self.assertIn(
            "notebooks/tutorial_single_source_analysis",
            INDEX.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "tutorial_single_source_analysis.ipynb",
            STATUS.read_text(encoding="utf-8"),
        )

    def test_notebook_contains_the_complete_required_workflow(self):
        text = notebook_text()
        required_tokens = [
            "examples/data/10131+3049.csv",
            "RUN_REQUIRED_FIT = True",
            "MAX_SAMPLES_PER_OBSERVATIONAL_CHANNEL = 50",
            "TRAINING_ITER = 200",
            "MINITER = 100",
            "summarize_single_source_observational_channels",
            "build_per_observational_channel_period_evidence",
            'model": "2DWavelengthDependent"',
            'fit_strategy": "consensus"',
            'time_kernel_type": "quasi_periodic"',
            'wavelength_kernel_type": "rbf"',
            'constraint_set": "LPV"',
            'learn_additional_noise": True',
            'use_parameter_workflow": True',
            'duplicate_wavelength_policy": "select"',
            "DUPLICATE_WAVELENGTH_SELECTION",
            "KELT/OSN_Johnson.Cousins_R3_0",
            "KELT/OSN_Johnson.Cousins_R3_1",
            "build_wavelength_constraint_position_rows",
            "training_point_prediction_summary",
            "lightcurve.plot(",
            "summarize_observational_channel_residuals",
            "build_phase_folded_prediction_summary",
            "summarize_single_source_fit_quality",
            "summarize_single_source_noise_provenance",
            "build_single_source_analysis_report",
            "validate_single_source_stage_records",
            "SINGLE_SOURCE_ANALYSIS_STAGE_ORDER",
            "write_single_source_analysis_report",
            "constraint_then_value",
            "fractional_position_within_bounds",
            "minimum_distance_to_bound",
            "additional_noise_is_not_measurement_error_replacement",
            "gp_training_scope",
            "consensus_input_scope",
            "channels_requiring_review",
            "fit_quality_warning_messages",
        ]
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, text)


    def test_first_code_cell_bootstraps_google_colab(self):
        cells = code_cells()
        self.assertTrue(cells)
        bootstrap = cells[0]
        required_tokens = [
            'importlib.util.find_spec("google.colab")',
            'PGMUVI_COLAB_CHECKOUT = Path("/content/pgmuvi-repo")',
            '"git",',
            '"clone",',
            '"--branch",',
            "PGMUVI_COLAB_GIT_REF",
            "sys.executable",
            '"--editable",',
            "for module_name in list(sys.modules):",
            "sys.path.insert(0, repository_string)",
            "importlib.invalidate_caches()",
            'REPOSITORY_ROOT / "pgmuvi/single_source_analysis.py"',
            "PGMUVI was imported from the wrong location",
            'REPOSITORY_ROOT / "examples/data/10131+3049.csv"',
        ]
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, bootstrap)

        self.assertNotIn('Path("/content/pgmuvi")', bootstrap)
        self.assertNotIn("!pip", bootstrap)
        self.assertNotIn("%pip", bootstrap)
        self.assertLess(
            bootstrap.index("REPOSITORY_ROOT = _prepare_repository_root()"),
            bootstrap.index(
                "from pgmuvi.single_source_analysis import"
            ),
        )

    def test_default_path_cannot_silently_skip_training(self):
        text = notebook_text()
        self.assertNotIn("training_iter=0", text)
        self.assertNotIn('"training_iter": 0', text)
        self.assertNotIn("RUN_REQUIRED_FIT = False", text)
        self.assertIn(
            'assert RUN_REQUIRED_FIT, "The default Run All path must execute the GP fit."',
            text,
        )
        self.assertIn('assert FIT_KWARGS["training_iter"] > 0', text)
        self.assertIn('assert FIT_KWARGS["miniter"] > 0', text)
        self.assertIn('assert lightcurve.is_fitted', text)
        self.assertIn('assert latest_fit["success"] is True', text)

    def test_scientific_order_is_explicit_in_separate_cells(self):
        cells = code_cells()
        load_cell = next(
            index
            for index, source in enumerate(cells)
            if "load_wavelength_constraint_tutorial_lightcurve(" in source
        )
        evidence_cell = next(
            index
            for index, source in enumerate(cells)
            if "period_evidence = build_per_observational_channel_period_evidence"
            in source
        )
        fit_cell = next(
            index
            for index, source in enumerate(cells)
            if "fit_result = lightcurve.fit(**FIT_KWARGS)" in source
        )
        constraint_cell = next(
            index
            for index, source in enumerate(cells)
            if "constraint_rows = build_wavelength_constraint_position_rows"
            in source
        )
        prediction_cell = next(
            index
            for index, source in enumerate(cells)
            if "predictions = training_point_prediction_summary" in source
        )
        plot_cell = next(
            index
            for index, source in enumerate(cells)
            if "fitted_lightcurve_figures = lightcurve.plot(" in source
        )
        residual_cell = next(
            index
            for index, source in enumerate(cells)
            if "channel_residual_rows = summarize_observational_channel_residuals"
            in source
        )
        report_cell = next(
            index
            for index, source in enumerate(cells)
            if "analysis_report = build_single_source_analysis_report(" in source
        )
        self.assertLess(load_cell, evidence_cell)
        self.assertLess(evidence_cell, fit_cell)
        self.assertLess(fit_cell, constraint_cell)
        self.assertLess(constraint_cell, prediction_cell)
        self.assertLess(prediction_cell, plot_cell)
        self.assertLess(plot_cell, residual_cell)
        self.assertLess(residual_cell, report_cell)

    def test_consensus_and_parameter_workflow_order_is_documented_and_verified(self):
        text = notebook_text()
        self.assertIn(
            "Before optimizer training, PGMUVI",
            text,
        )
        self.assertIn(
            "builds and applies the model's data-derived parameter workflow",
            text,
        )
        self.assertIn('consensus_diagnostics["consensus_success"] is True', text)
        self.assertIn("parameter_workflow_report", text)
        self.assertIn('row["application_order"] == "constraint_then_value"', text)
        self.assertIn('row["constraint_registered"]', text)
        self.assertIn('row["value_initialized"]', text)

    def test_state_changes_are_scoped_and_restored(self):
        text = notebook_text()
        self.assertGreaterEqual(
            text.count("with preserve_single_source_analysis_state("),
            8,
        )
        self.assertNotIn("torch.set_default_dtype(", text)
        self.assertNotIn("np.random.seed(", text)
        self.assertNotIn("torch.manual_seed(", text)
        self.assertNotIn("os.chdir(", text)

    def test_only_the_bundled_real_source_is_used(self):
        text = notebook_text()
        unavailable_source = "07454" + "-7112"
        unsupported_suffix = "." + "parquet"
        self.assertNotIn(unavailable_source, text)
        self.assertNotIn("NESS", text)
        self.assertNotIn(unsupported_suffix, text)
        self.assertNotRegex(
            text,
            re.compile(r"/(?:Users|Volumes|home)/", re.IGNORECASE),
        )
        self.assertIn("strictly positive", text)
        self.assertIn("linear flux", text.lower())
        self.assertIn(
            '["time", "physical_wavelength"]',
            text,
        )

    def test_observational_channel_and_calibration_boundaries_are_explicit(self):
        text = notebook_text()
        normalized = " ".join(text.split())
        self.assertIn("observational channel", normalized)
        self.assertIn("physical wavelength", normalized)
        self.assertIn("not an instrument-channel calibration", normalized)
        self.assertIn("both KELT channels were", normalized)
        self.assertNotIn("2DAchromatic", normalized)
        self.assertIn(
            "not evidence that the source is achromatic",
            normalized,
        )

    def test_fit_quality_warnings_are_promoted_to_the_final_report(self):
        text = notebook_text()
        self.assertIn(
            "fit_quality = summarize_single_source_fit_quality(",
            text,
        )
        self.assertIn(
            'fit_quality["warning_messages"]',
            text,
        )
        self.assertIn(
            "fit_warning_messages + plot_warning_messages "
            "+ fit_quality_warning_messages",
            " ".join(text.split()),
        )
        self.assertIn(
            '"fit_quality": fit_quality',
            text,
        )
        self.assertIn(
            '"fit_quality_status": fit_quality["status"]',
            text,
        )

    def test_all_code_cells_compile_and_parse(self):
        for index, source in enumerate(code_cells()):
            compile(
                source,
                f"tutorial_single_source_analysis.ipynb:cell-{index}",
                "exec",
            )
            ast.parse(source)

    def test_every_cell_has_a_stable_nbformat_id(self):
        notebook = load_notebook()
        cell_ids = [cell.get("id") for cell in notebook.get("cells", [])]
        self.assertTrue(all(isinstance(value, str) and value for value in cell_ids))
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        self.assertTrue(all(value.startswith("pr174-cell-") for value in cell_ids))

    def test_saved_execution_is_complete_when_outputs_are_present(self):
        notebook = load_notebook()
        code = [
            cell
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        execution_counts = [cell.get("execution_count") for cell in code]
        if all(value is None for value in execution_counts):
            return
        self.assertTrue(all(isinstance(value, int) for value in execution_counts))
        self.assertEqual(
            execution_counts,
            sorted(execution_counts),
        )
        self.assertEqual(
            len(execution_counts),
            len(set(execution_counts)),
        )
        fit_cell = next(
            cell
            for cell in code
            if "fit_result = lightcurve.fit(**FIT_KWARGS)"
            in "".join(cell.get("source", []))
        )
        report_cell = next(
            cell
            for cell in code
            if "analysis_report = build_single_source_analysis_report("
            in "".join(cell.get("source", []))
        )
        self.assertTrue(fit_cell.get("outputs"))
        self.assertTrue(report_cell.get("outputs"))

    def test_companion_documentation_links_the_complete_notebook(self):
        for document in [CONSENSUS, MODELS, PRIORS, INTERPRET]:
            text = document.read_text(encoding="utf-8")
            self.assertIn(
                ":doc:`../notebooks/tutorial_single_source_analysis`",
                text,
            )
        future = FUTURE.read_text(encoding="utf-8")
        self.assertIn("Completed complete single-source notebook (PR174)", future)
        self.assertNotIn("TBD[result-interpretation-notebook]", future)


if __name__ == "__main__":
    unittest.main()
