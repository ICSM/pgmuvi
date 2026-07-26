"""Regression tests for the executed public 2-D consensus notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/source/notebooks/pgmuvi_tutorial_2d.ipynb"
INDEX = ROOT / "docs/source/index.rst"
CONF = ROOT / "docs/source/conf.py"
CONSENSUS = ROOT / "docs/source/howto/consensus_fitting.rst"
STATUS = ROOT / "docs/source/notebook_status.rst"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells() -> list[dict]:
    return [
        cell
        for cell in load_notebook().get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def code_sources() -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in code_cells()
    ]


def notebook_text() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook().get("cells", [])
    )


def output_text() -> str:
    chunks: list[str] = []

    for cell in code_cells():
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                chunks.append(
                    "".join(text)
                    if isinstance(text, list)
                    else str(text)
                )
                continue

            data = output.get("data", {})
            plain = data.get("text/plain", "")
            chunks.append(
                "".join(plain)
                if isinstance(plain, list)
                else str(plain)
            )

    return "\n".join(chunks)


class TestExecuted2DConsensusNotebook(unittest.TestCase):
    def test_notebook_is_public_and_not_excluded(self):
        self.assertIn(
            "notebooks/pgmuvi_tutorial_2d",
            INDEX.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "notebooks/pgmuvi_tutorial_2d.ipynb",
            CONF.read_text(encoding="utf-8"),
        )

    def test_notebook_has_executed_current_structure(self):
        notebook = load_notebook()

        self.assertEqual(notebook.get("nbformat"), 4)
        self.assertEqual(
            notebook.get("metadata", {})
            .get("kernelspec", {})
            .get("name"),
            "python3",
        )
        self.assertGreaterEqual(
            len(notebook.get("cells", [])),
            12,
        )

        for cell in load_notebook().get("cells", []):
            self.assertTrue(cell.get("id"))

        for cell in code_cells():
            self.assertIsNotNone(
                cell.get("execution_count")
            )
            self.assertFalse(
                any(
                    output.get("output_type") == "error"
                    for output in cell.get("outputs", [])
                )
            )

    def test_required_end_to_end_contract(self):
        text = notebook_text()

        for token in [
            "RUN_REQUIRED_FIT = True",
            '"model": "2DWavelengthDependent"',
            '"fit_strategy": "consensus"',
            '"time_kernel_type": "quasi_periodic"',
            '"wavelength_kernel_type": "rbf"',
            '"constraint_set": "LPV"',
            '"learn_additional_noise": True',
            "warnings.catch_warnings(record=True)",
            "fit_result = lc.fit(**fit_kwargs)",
            "fit_history = lc.get_fit_history()",
            '"fit_warning_count"',
            '"plot_warning_count"',
            'latest_fit["success"] is True',
            'latest_fit["training_iter"] == TRAINING_ITER',
            'period_summary["dominant_period"]',
            "fitted_lightcurve_figures = lc.plot(",
            '"figure_count"',
            "2DDustMean",
            "2DPowerLawMean",
            "2DSeparable",
            "automatic model selection",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_default_path_cannot_skip_required_fit(self):
        text = notebook_text()

        for stale in [
            "RUN_FIT = False",
            "RUN_FITS = False",
            "PREPARE ONLY",
            "training was not started",
            "No fit diagnostics exist",
            "if RUN_FIT:",
        ]:
            with self.subTest(stale=stale):
                self.assertNotIn(stale, text)

    def test_fit_and_plot_order_is_semantic(self):
        sources = code_sources()

        fit_cell = next(
            index
            for index, source in enumerate(sources)
            if "fit_result = lc.fit(**fit_kwargs)" in source
        )
        history_cell = next(
            index
            for index, source in enumerate(sources)
            if "fit_history = lc.get_fit_history()" in source
        )
        plot_cell = next(
            index
            for index, source in enumerate(sources)
            if "fitted_lightcurve_figures = lc.plot("
            in source
        )

        self.assertEqual(fit_cell, history_cell)
        self.assertLess(fit_cell, plot_cell)

    def test_all_code_cells_compile_and_parse(self):
        for index, source in enumerate(code_sources()):
            compile(
                source,
                f"pgmuvi_tutorial_2d.ipynb:cell-{index}",
                "exec",
            )
            ast.parse(source)

    def test_saved_outputs_prove_fit_and_plot_completed(self):
        text = output_text()

        self.assertIn("fit_success", text)
        self.assertIn("figure_count", text)
        self.assertIn("dominant_period", text)
        self.assertIn("fit_warning_count", text)
        self.assertIn("plot_warning_count", text)

        for machine_local_prefix in [
            "/Volumes/",
            "/Users/",
            "/private/var/",
        ]:
            self.assertNotIn(
                machine_local_prefix,
                text,
            )

        image_outputs = [
            output
            for cell in code_cells()
            for output in cell.get("outputs", [])
            if "image/png" in output.get("data", {})
        ]
        self.assertGreaterEqual(len(image_outputs), 1)

    def test_status_and_consensus_guide(self):
        self.assertIn(
            ":doc:`../notebooks/pgmuvi_tutorial_2d`",
            CONSENSUS.read_text(encoding="utf-8"),
        )

        status = STATUS.read_text(encoding="utf-8")
        self.assertIn(
            "required ``2DWavelengthDependent`` consensus fit",
            status,
        )
        self.assertIn(
            "calls ``Lightcurve.plot()``",
            status,
        )
        self.assertNotIn(
            "explicit no-training default",
            status,
        )


if __name__ == "__main__":
    unittest.main()
