"""Regression tests for the real-data wavelength-constraint tutorial."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/source/notebooks/tutorial_wavelength_constraints.ipynb"
REPRESENTATIVE_CSV = ROOT / "examples/data/10131+3049.csv"
INDEX = ROOT / "docs/source/index.rst"
STATUS = ROOT / "docs/source/notebook_status.rst"
FUTURE = ROOT / "docs/source/future_work.rst"
PRIORS = ROOT / "docs/source/howto/priors_constraints.rst"
MODELS = ROOT / "docs/source/howto/wavelength_models.rst"
CALIBRATION = ROOT / (
    "docs/source/pgmuvi.wavelength_validation_robustness_calibration.rst"
)


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


class TestWavelengthConstraintNotebook(unittest.TestCase):
    def test_notebook_and_public_data_are_present(self):
        self.assertTrue(NOTEBOOK.is_file())
        self.assertTrue(REPRESENTATIVE_CSV.is_file())
        self.assertIn(
            "notebooks/tutorial_wavelength_constraints",
            INDEX.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "TBD[wavelength-constraint-notebook]",
            STATUS.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "TBD[wavelength-constraint-notebook]",
            FUTURE.read_text(encoding="utf-8"),
        )

    def test_notebook_has_substantive_real_data_workflow(self):
        text = notebook_text()
        required_tokens = [
            "examples/data/10131+3049.csv",
            "load_wavelength_constraint_tutorial_lightcurve",
            "RUN_REQUIRED_FITS = True",
            "MAX_SAMPLES_PER_OBSERVATIONAL_CHANNEL",
            'model="2DWavelengthDependent"',
            'model="2DSeparable"',
            "primary_lightcurve.fit(",
            "control_lightcurve.fit(",
            'duplicate_wavelength_policy": "select"',
            "DUPLICATE_WAVELENGTH_SELECTION",
            "primary_lightcurve.plot(",
            "training_point_prediction_summary",
            "summarize_observational_channel_residuals",
            "standardized_residual",
            "build_wavelength_constraint_position_rows",
            "constraint_then_value",
            "minimum_distance_to_bound",
            "build_tutorial_fit_summary",
            "objective_improved",
            "nonfinite_prediction_count",
            "negative_variance_count",
            "residual_rmse",
            "TBD[instrument-channel-calibration]",
        ]
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_default_path_does_not_disable_or_skip_required_fits(self):
        text = notebook_text()
        for stale in [
            "RUN_REDUCED_SYNTHETIC_FITS = False",
            "does not train a GP",
            "default notebook path must not fit",
            'fit_results = {"status": "disabled"',
        ]:
            with self.subTest(stale=stale):
                self.assertNotIn(stale, text)
        self.assertIn("RUN_OPTIONAL_EXPENSIVE_EXTENSIONS = False", text)
        self.assertIn(
            "all required fits above did run",
            text,
        )

    def test_fit_prediction_plot_and_residual_order_is_semantic(self):
        cells = code_cells()
        primary_fit_cell = next(
            index for index, source in enumerate(cells)
            if "primary_lightcurve.fit(" in source
        )
        prediction_cell = next(
            index for index, source in enumerate(cells)
            if "primary_predictions = training_point_prediction_summary" in source
        )
        plot_cell = next(
            index for index, source in enumerate(cells)
            if "fitted_lightcurve_figures = primary_lightcurve.plot(" in source
        )
        residual_cell = next(
            index for index, source in enumerate(cells)
            if "primary_channel_residual_rows" in source
        )
        summary_cell = next(
            index for index, source in enumerate(cells)
            if "required_summary_fields" in source
        )
        self.assertLess(primary_fit_cell, prediction_cell)
        self.assertLess(primary_fit_cell, plot_cell)
        self.assertLess(prediction_cell, residual_cell)
        self.assertLess(residual_cell, summary_cell)

    def test_all_code_cells_compile_and_parse(self):
        for index, source in enumerate(code_cells()):
            compile(
                source,
                f"tutorial_wavelength_constraints.ipynb:cell-{index}",
                "exec",
            )
            ast.parse(source)

    def test_observational_channel_terminology_and_calibration_boundary(self):
        text = notebook_text()
        self.assertIn("observational channel", text)
        self.assertIn("physical wavelength", text)
        self.assertIn("shared-wavelength channels", text)
        self.assertIn("does not cross-calibrate", text)
        self.assertNotIn("NESS", text)
        self.assertIn(
            "TBD[instrument-channel-calibration]",
            FUTURE.read_text(encoding="utf-8"),
        )

    def test_companion_documentation_links_notebook(self):
        for document in [PRIORS, MODELS, CALIBRATION]:
            text = document.read_text(encoding="utf-8")
            expected = (
                ":doc:`../notebooks/tutorial_wavelength_constraints`"
                if document in {PRIORS, MODELS}
                else ":doc:`notebooks/tutorial_wavelength_constraints`"
            )
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
