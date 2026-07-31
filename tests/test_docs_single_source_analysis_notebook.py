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
            'model="2DWavelengthDependent"',
            'fit_strategy=RESOLVED_FIT_STRATEGY',
            'time_kernel_type=RESOLVED_TIME_KERNEL_TYPE',
            'wavelength_kernel_type="rbf"',
            'constraint_set": "LPV"',
            'learn_additional_noise": True',
            'use_parameter_workflow": True',
            'duplicate_wavelength_policy="select"',
            '"default exact-GP selection"',
            "expected_default_duplicate_selection",
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
        self.assertIn('assert TRAINING_ITER > 0', text)
        self.assertIn('assert MINITER > 0', text)
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
            if "workflow_enabled_result = workflow_enabled_lightcurve.fit(" in source
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
        self.assertIn(
            (
                "All retained channels in each duplicate group are "
                "analysed independently for period evidence and remain "
                "available to consensus"
            ),
            normalized,
        )
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
            if "workflow_enabled_result = workflow_enabled_lightcurve.fit("
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



class TestSingleSourceNotebookSourceGenerality(unittest.TestCase):
    @staticmethod
    def _source_text():
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "docs/source/notebooks/tutorial_single_source_analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )

    def test_notebook_source_contains_no_example_specific_logic(self):
        source_text = self._source_text()
        forbidden = [
            "KELT/OSN_Johnson.Cousins_R3_0",
            "KELT/OSN_Johnson.Cousins_R3_1",
            "0.6561154962791801",
            'source_id="10131+3049"',
            'source_path="examples/data/10131+3049.csv"',
            "all 17 observational channels",
            "both KELT channels",
        ]
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source_text)

    def test_source_and_report_metadata_are_derived(self):
        source_text = self._source_text()
        required = [
            'SOURCE_CSV = REPOSITORY_ROOT / '
            '"examples/data/10131+3049.csv"',
            "SOURCE_ID = SOURCE_CSV.stem",
            "SOURCE_PATH_FOR_REPORT",
            'name=f"{SOURCE_ID} complete single-source analysis"',
            "source_id=SOURCE_ID",
            "source_path=SOURCE_PATH_FOR_REPORT",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, source_text)

    def test_duplicate_groups_and_output_counts_are_dynamic(self):
        source_text = self._source_text()
        required = [
            "duplicate_wavelength_groups",
            "duplicate_group_wavelength",
            '"default exact-GP selection"',
            "expected_default_duplicate_selection",
            '"ignored observational channels"',
            "expected_gp_figure_count",
            "expected_training_channel_count",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, source_text)



class TestSingleSourceNotebookGeneralInvariants(unittest.TestCase):
    @staticmethod
    def _code_text():
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "docs/source/notebooks/tutorial_single_source_analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_executable_assertions_are_not_fixture_specific(self):
        code_text = self._code_text()
        forbidden = [
            'assert sampling_summary["n_rows_original"] == 10815',
            'assert input_summary["n_observational_channels"] == 17',
            'assert input_summary["n_physical_wavelengths"] == 16',
            "KELT/OSN_Johnson.Cousins_R3_0",
            "KELT/OSN_Johnson.Cousins_R3_1",
            "0.6561154962791801",
        ]
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, code_text)

    def test_retained_input_invariants_are_general(self):
        code_text = self._code_text()
        required = [
            "retained_row_count = retained_time.size",
            'assert sampling_summary["n_rows_retained"] == '
            "retained_row_count",
            "assert np.all(np.isfinite(retained_time))",
            "assert np.all(np.isfinite(retained_flux))",
            "assert np.all(np.isfinite(retained_flux_error))",
            "assert np.all(retained_flux > 0.0)",
            "assert np.all(retained_flux_error > 0.0)",
            "derived_duplicate_groups",
            "assert reported_channels == "
            "derived_duplicate_groups[wavelength]",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, code_text)

    def test_result_scope_invariants_are_dynamic(self):
        code_text = self._code_text()
        required = [
            'period_evidence["n_available"]',
            'period_evidence["n_unavailable"]',
            "prediction_array_lengths",
            "assert len(set(prediction_array_lengths.values())) == 1",
            "training_observational_channels",
            "residual_observational_channels",
            "phase_observational_channels",
            ") == n_predictions",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, code_text)



class TestSingleSourceNotebookSamplingQualityDefault(unittest.TestCase):
    @staticmethod
    def _notebook_texts():
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "docs/source/notebooks/tutorial_single_source_analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        return code_text, markdown_text

    def test_sampling_quality_check_is_enabled(self):
        code_text, _ = self._notebook_texts()
        required = [
            "CHECK_SAMPLING = True",
            "SAMPLING_KWARGS = None",
            "check_sampling=CHECK_SAMPLING",
            "sampling_kwargs=SAMPLING_KWARGS",
            'assert sampling_summary["check_sampling"] is CHECK_SAMPLING',
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, code_text)
        self.assertNotIn("check_sampling=False", code_text)

    def test_sampling_quality_defaults_are_documented(self):
        _, markdown_text = self._notebook_texts()
        required = [
            "`check_sampling=True`",
            "`min_points=15`",
            "`max_gap_fraction=0.3`",
            "`min_baseline_factor=3.0`",
            "`min_snr=3.0`",
            "`min_fraction_good_snr=0.5`",
            "physical wavelength",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, markdown_text)



class TestSingleSourceNotebookChannelPlotStyles(unittest.TestCase):
    @staticmethod
    def _code_text():
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "docs/source/notebooks/tutorial_single_source_analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_style_builder_is_dynamic_and_deterministic(self):
        code_text = self._code_text()
        required = [
            "OBSERVATIONAL_CHANNEL_MARKERS = (",
            "def build_observational_channel_styles(",
            "dict.fromkeys(str(channel) for channel in "
            "observational_channels)",
            "style_combinations = [",
            "observational_channel_styles = (",
            "build_observational_channel_styles(",
            "set(retained_observational_channels)",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, code_text)

    def test_retained_flux_plot_uses_log_scale_and_shared_styles(self):
        code_text = self._code_text()
        required = [
            'ax.set_yscale("log")',
            "Sampling-quality-retained observations by "
            "observational channel",
            'style = observational_channel_styles[observational_channel]',
            'color=style["color"]',
            'marker=style["marker"]',
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, code_text)
        self.assertNotIn(
            "Deterministically retained observations by "
            "observational channel",
            code_text,
        )

    def test_shared_styles_are_reused_in_residual_and_phase_plots(self):
        code_text = self._code_text()
        self.assertGreaterEqual(
            code_text.count(
                "style = "
                "observational_channel_styles[observational_channel]"
            ),
            4,
        )
        self.assertGreaterEqual(
            code_text.count('color=style["color"]'),
            4,
        )
        self.assertGreaterEqual(
            code_text.count('marker=style["marker"]'),
            4,
        )
        self.assertIn(
            "Standardized phase-folded residual by observational channel",
            code_text,
        )
        self.assertIn(
            "phase_values.size == prediction_channels.size",
            code_text,
        )
        self.assertIn(
            "standardized_phase_residual_values = np.asarray(",
            code_text,
        )
        self.assertIn(
            'phase_diagnostics["standardized_residual"]',
            code_text,
        )
        self.assertNotIn(
            'phase_diagnostics["residual"]',
            code_text,
        )



class TestSingleSourceNotebookPeriodDiagnosticComparison(unittest.TestCase):
    @staticmethod
    def _notebook():
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "docs/source/notebooks/tutorial_single_source_analysis.ipynb"
        )
        return json.loads(notebook_path.read_text(encoding="utf-8"))

    def test_period_evidence_is_registered_on_parent_lightcurve(self):
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in self._notebook()["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn(
            "lightcurve.register_period_diagnostic_evidence(\n"
            "    period_evidence\n"
            ")",
            code_text,
        )

    def test_comparison_method_runs_after_period_summary(self):
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in self._notebook()["cells"]
            if cell.get("cell_type") == "code"
        )
        summary_index = code_text.index(
            "period_summary = lightcurve.get_period_summary("
        )
        comparison_index = code_text.index(
            "lightcurve.plot_period_diagnostic_comparison("
        )
        self.assertGreater(comparison_index, summary_index)
        self.assertIn("strict=True", code_text)

    def test_bespoke_prefit_periodogram_plot_is_removed(self):
        notebook = self._notebook()
        text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertNotIn(
            "Representative channel-specific Lomb–Scargle periodograms",
            text,
        )
        self.assertNotIn("plot_rows = sorted(", text)
        self.assertIn(
            "The multi-method comparison is generated only after",
            text,
        )


class TestSingleSourceNotebookDefaultDuplicatePolicy(unittest.TestCase):
    @staticmethod
    def _notebook_code():
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "docs/source/notebooks/tutorial_single_source_analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_fit_uses_package_default_duplicate_policy(self):
        code_text = self._notebook_code()
        active_code = "\n".join(
            line
            for line in code_text.splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertNotIn(
            "DUPLICATE_WAVELENGTH_SELECTION",
            active_code,
        )
        self.assertNotIn(
            '"duplicate_wavelength_policy"',
            active_code,
        )
        self.assertNotIn(
            '"duplicate_wavelength_selection"',
            active_code,
        )
        self.assertIn("workflow_enabled_result = workflow_enabled_lightcurve.fit(", active_code)
        self.assertIn(
            "workflow_disabled_result = workflow_disabled_lightcurve.fit(",
            active_code,
        )

    def test_generic_override_is_comment_only(self):
        code_text = self._notebook_code()
        required = [
            '# duplicate_wavelength_policy="first".',
            '# duplicate_wavelength_policy="select",',
            '# duplicate_wavelength_selection={',
            (
                "#     duplicated_wavelength: "
                "preferred_observational_channel,"
            ),
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, code_text)

    def test_default_resolution_is_verified_dynamically(self):
        code_text = self._notebook_code()
        required = [
            'duplicate_wavelength_resolution["policy"] == "first"',
            "expected_default_duplicate_selection",
            "actual_default_duplicate_selection",
            'group["observational_channels"][0]',
            'group["selected_observational_channel"]',
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, code_text)


class TestSingleSourceNotebookPeriodComponentControls(unittest.TestCase):
    @staticmethod
    def _texts():
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "docs/source/notebooks/tutorial_single_source_analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        return code, markdown

    def test_user_controls_ls_and_gp_component_counts(self):
        code, _ = self._texts()
        for token in (
            "LS_NUM_COMPONENTS = 3",
            "GP_NUM_COMPONENTS = ",
            "resolve_single_source_period_component_configuration(",
            "num_peaks=LS_NUM_COMPONENTS",
            'fit_strategy=RESOLVED_FIT_STRATEGY',
            'time_kernel_type=RESOLVED_TIME_KERNEL_TYPE',
            'num_mixtures=RESOLVED_NUM_MIXTURES',
            "n_peaks=GP_NUM_COMPONENTS",
            "prefer_fitted_psd=True",
        ):
            with self.subTest(token=token):
                self.assertIn(token, code)

    def test_acf_is_curve_only(self):
        code, markdown = self._texts()
        self.assertNotIn("strongest_positive_lag", code)
        self.assertNotIn("strongest positive ACF lag", code)
        self.assertIn("ACF curve available", code)
        self.assertIn("does not identify peaks", markdown)

    def test_gp_summary_uses_period_axis_and_components(self):
        code, _ = self._texts()
        for token in (
            '"x_axis": "period"',
            '"log_x": True',
            '"show_components": True',
            "period_summary=period_summary",
        ):
            with self.subTest(token=token):
                self.assertIn(token, code)

class TestComment8ParameterWorkflowComparison(unittest.TestCase):
    def setUp(self):
        repository = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (
                repository
                / "docs/source/notebooks/"
                "tutorial_single_source_analysis.ipynb"
            ).read_text(encoding="utf-8")
        )
        self.notebook_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )

    def test_explicit_enabled_and_disabled_fit_calls_are_visible(self):
        source = self.notebook_source
        self.assertIn(
            "workflow_enabled_result = workflow_enabled_lightcurve.fit(",
            source,
        )
        self.assertIn(
            "workflow_disabled_result = workflow_disabled_lightcurve.fit(",
            source,
        )
        self.assertIn("use_parameter_workflow=True", source)
        self.assertIn("use_parameter_workflow=False", source)
        self.assertNotIn("fit_result = lightcurve.fit(**FIT_KWARGS)", source)

    def test_comparison_is_documented_without_overclaiming(self):
        source = self.notebook_source
        self.assertIn(
            "parameter workflow enabled versus disabled",
            source,
        )
        self.assertIn(
            "The disabled fit is **not unconstrained**",
            source,
        )
        self.assertIn(
            "enabled − parameter workflow disabled",
            source,
        )
        self.assertIn(
            "same temporal-consensus pipeline",
            source,
        )

    def test_numerical_and_graphical_comparison_contract(self):
        source = self.notebook_source
        for token in (
            "final objective",
            "best objective",
            "dominant period",
            "fitted wavelength value",
            "standardized residual RMS",
            "empirical 95% coverage",
            "channels requiring review",
            "parameter_comparison_rows",
            "workflow_comparison_channel_rows",
            "all fitted-light-curve panels",
            "parameter_workflow_comparison",
        ):
            self.assertIn(token, source)

    def test_control_reload_replays_identical_numeric_analysis_state(self):
        source = self.notebook_source
        loader_token = (
            "workflow_disabled_lightcurve, "
            "workflow_disabled_sampling_summary = ("
        )
        loader_index = source.index(loader_token)
        fit_index = source.index(
            "# FIT A: schema-driven parameter initialization",
            loader_index,
        )
        context_index = source.rfind(
            "with preserve_single_source_analysis_state(",
            0,
            loader_index,
        )
        self.assertGreaterEqual(context_index, 0)
        control_block = source[context_index:fit_index]
        self.assertIn("seed=SEED", control_block)
        self.assertIn(
            "default_dtype=torch.float64",
            control_block,
        )
        self.assertIn(
            "working_directory=REPOSITORY_ROOT",
            control_block,
        )
        self.assertIn(
            "controlled_sampling_summary_keys",
            control_block,
        )
        self.assertIn(
            "workflow_enabled_lightcurve.xdata.dtype "
            "== torch.float64",
            control_block,
        )
        self.assertIn(
            "workflow_disabled_lightcurve.xdata.dtype "
            "== torch.float64",
            control_block,
        )
        self.assertNotIn(
            "assert workflow_disabled_sampling_summary "
            "== sampling_summary",
            source,
        )

    def test_comparison_resolves_constraints_on_fitted_model_raw_names(self):
        source = self.notebook_source
        self.assertIn(
            "lightcurve_object.model.named_parameters()",
            source,
        )
        self.assertIn(
            'component.removeprefix("raw_")',
            source,
        )
        self.assertIn(
            "lightcurve_object.model.constraint_for_parameter_name(",
            source,
        )
        self.assertIn(
            "matched_model_parameter",
            source,
        )
        self.assertNotIn(
            "lightcurve_object.constraint_for_parameter_name(",
            source,
        )

    def test_section4_interprets_constraints_without_overclaiming(self):
        source = self.notebook_source
        for token in (
            "Constraint registered",
            "Initialization valid",
            "Fitted value valid",
            "Parameter identified",
            "technically satisfactory",
            "near-bound",
            "at-bound",
            "not_established_by_constraint_diagnostics",
            "Parameter-by-parameter interpretation",
            "value_origin",
            "interval_origin",
            "parameter_constraint_diagnostics",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertIn(
            "does not mean that the parameter is well identified",
            source,
        )

    def test_notebook_tables_use_export_safe_markdown(self):
        source = self.notebook_source
        self.assertIn(
            "from IPython.display import Markdown, display",
            source,
        )
        self.assertIn(
            "display(Markdown(markdown_text))",
            source,
        )
        self.assertIn(
            '"| " + " | ".join',
            source,
        )
        self.assertNotIn("HTML(", source)
        self.assertNotIn(
            "from IPython.display import HTML",
            source,
        )

    def test_constraint_interpretation_is_preserved_in_report(self):
        source = self.notebook_source
        self.assertIn(
            '"interpretation": parameter_constraint_diagnostics',
            source,
        )
        self.assertIn(
            "summarize_single_source_constraint_diagnostics(",
            source,
        )


class TestComment10CompleteFitInterpretation(unittest.TestCase):
    @staticmethod
    def _source():
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "docs/source/notebooks/tutorial_single_source_analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )

    def test_all_maintained_fit_panels_are_displayed(self):
        source = self._source()
        self.assertNotIn("selected_figure_indices", source)
        self.assertIn("for figure in fitted_lightcurve_figures:", source)
        self.assertIn("display(figure)", source)
        self.assertIn("every fitted physical-wavelength panel", source)

    def test_pairwise_scientific_explanations_are_reported(self):
        source = self._source()
        for token in (
            "summarize_single_source_fit_explanations(",
            "Per-channel fit explanation diagnostics",
            "Channel pairs requiring scientific interpretation",
            "normalized_phase_shape_correlation",
            "fitted_wavelength_kernel_correlation",
            "phase_aligned_temporal_kernel_median_absolute_correlation",
            '"fit_explanations": fit_explanations',
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_notebook_does_not_overclaim_calibration_or_coherence(self):
        source = self._source()
        self.assertIn(
            "passband/SED or calibration incompatibility candidate",
            source,
        )
        self.assertIn(
            "cannot distinguish those explanations",
            source,
        )
        self.assertIn("the active fit uses a", source)
        self.assertIn("spectral-mixture temporal kernel", source)
        self.assertNotIn("poorly sampled channels were retained", source)


class TestComment11ResidualDynamicRange(unittest.TestCase):
    @staticmethod
    def _source():
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "docs/source/notebooks/tutorial_single_source_analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )

    def test_primary_cross_channel_plot_is_dimensionless(self):
        source = self._source()
        self.assertIn(
            'row["standardized_residual_rms"]',
            source,
        )
        self.assertIn("Standardized residual RMS", source)
        self.assertIn(
            "Primary scale-independent residual summary by channel",
            source,
        )
        self.assertIn(
            '"primary_cross_channel_metric": "standardized_residual_rms"',
            source,
        )

    def test_raw_rmse_is_secondary_and_log_scaled(self):
        source = self._source()
        self.assertIn('ax.set_yscale("log")', source)
        self.assertIn(
            "Secondary raw-RMSE summary (logarithmic y-axis)",
            source,
        )
        self.assertIn(
            "Training residual RMSE [channel flux units]",
            source,
        )

    def test_combined_time_and_phase_plots_use_standardized_residuals(self):
        source = self._source()
        self.assertIn(
            'predictions["standardized_residual"][channel_mask]',
            source,
        )
        self.assertIn(
            'phase_diagnostics["standardized_residual"]',
            source,
        )
        self.assertIn(
            "Standardized phase-folded residual by observational channel",
            source,
        )
        self.assertNotIn(
            'phase_diagnostics["residual"]',
            source,
        )

    def test_raw_signed_residuals_use_per_channel_small_multiples(self):
        source = self._source()
        for token in (
            "sharey=False",
            "Raw residual [channel flux units]",
            "per-channel panels with independent y-ranges",
            '"signed_log_transform_applied": False',
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertNotIn('ax.set_yscale("symlog")', source)

    def test_dimensionless_metrics_and_scale_policy_are_reported(self):
        source = self._source()
        for token in (
            "fractional_rmse_over_abs_median_flux",
            "normalized_rmse_over_robust_amplitude",
            "empirical_95_percent_coverage",
            '"scale_policy": residual_scale_policy',
            '"overall_standardized_residual_rms"',
            '"overall_empirical_95_percent_coverage"',
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
