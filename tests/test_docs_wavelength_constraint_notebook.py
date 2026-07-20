"""Regression tests for the maintained wavelength-constraint notebook."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs/source/notebooks/tutorial_wavelength_constraints.ipynb"
INDEX = ROOT / "docs/source/index.rst"
CONF = ROOT / "docs/source/conf.py"
STATUS = ROOT / "docs/source/notebook_status.rst"
FUTURE = ROOT / "docs/source/future_work.rst"
PRIORS = ROOT / "docs/source/howto/priors_constraints.rst"
MODELS = ROOT / "docs/source/howto/wavelength_models.rst"
CALIBRATION = ROOT / (
    "docs/source/pgmuvi.wavelength_validation_robustness_calibration.rst"
)


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_text() -> str:
    chunks = []
    for cell in load_notebook().get("cells", []):
        source = cell.get("source", [])
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


class TestWavelengthConstraintNotebook(unittest.TestCase):
    def test_notebook_is_public_and_status_marker_is_closed(self):
        self.assertTrue(NOTEBOOK.is_file())
        self.assertIn(
            "notebooks/tutorial_wavelength_constraints",
            INDEX.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "notebooks/tutorial_wavelength_constraints.ipynb",
            CONF.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "TBD[wavelength-constraint-notebook]",
            STATUS.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "TBD[wavelength-constraint-notebook]",
            FUTURE.read_text(encoding="utf-8"),
        )

    def test_notebook_has_current_structure_and_clean_outputs(self):
        notebook = load_notebook()
        self.assertEqual(notebook.get("nbformat"), 4)
        self.assertEqual(
            notebook.get("metadata", {}).get("kernelspec", {}).get("name"),
            "python3",
        )
        self.assertGreaterEqual(len(notebook.get("cells", [])), 28)
        self.assertTrue(
            any(cell.get("cell_type") == "markdown" for cell in notebook["cells"])
        )
        self.assertTrue(
            any(cell.get("cell_type") == "code" for cell in notebook["cells"])
        )
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_notebook_covers_constraint_and_interpretation_contract(self):
        text = notebook_text()
        for token in [
            "build_wavelength_estimation_context",
            "build_wavelength_mean_estimation_context",
            "build_dimension_aware_sm_ard_estimates",
            "diagnose_spectral_mixture_ard",
            "get_parameter_workflow_report",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            "2DSeparable",
            "joint non-separable",
            "ARD index 0",
            "ARD index 1",
            "temporal_frequency",
            "wavelength_frequency",
            "physical wavelength",
            "round_trip_max_relative_error",
            "register the final constraint",
            "near-bound",
            "at-bound",
            "does not perform automatic model selection",
            "technical failure",
            "diagnostic unavailability",
            "scientific incompatibility",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, text)
        self.assertNotIn("2DAchromatic", text)

    def test_notebook_records_quasi_periodic_scope_and_empirical_results(self):
        text = notebook_text()
        for token in [
            'RUN_REDUCED_SYNTHETIC_FITS = False',
            '"fit_strategy": "consensus"',
            '"time_kernel_type": "quasi_periodic"',
            '"learn_additional_noise": True',
            "tested default hypothesis",
            "not a claim that every LPV is quasi-periodic",
            '"canonical_runs": 420',
            '"completed_fits": 393',
            '"expected_failures_matched": 20',
            '"structured_unexpected_consensus_rejections": 7',
            "d2-uneven-band-counts",
            "d2-longer-sparse-baseline",
            "d2-large-wavelength-gap",
            "d2-insufficient-per-band-sampling",
            "d2-reference-wavelength-quadratic-strong-turning",
            "d2-reference-joint-sm-ard-moderate",
            "d2-joint-sm-sparse-independent",
            "mean_structure_case_ids",
            "REDUCED_FIT_MODELS",
            "approximately 23-day",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_notebook_avoids_stale_or_nonportable_patterns(self):
        text = notebook_text()
        for token in [
            "TODO",
            "%pip",
            "!pip install",
            "dr2_pr133_canonical_10_seed_calibration",
            "auto_select_model",
            "best model",
        ]:
            with self.subTest(token=token):
                self.assertNotIn(token, text)

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(load_notebook().get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(
                source,
                f"tutorial_wavelength_constraints.ipynb:cell-{index}",
                "exec",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("gpytorch") is not None,
        "gpytorch is required for the notebook smoke test",
    )
    def test_default_path_executes_without_fitting_or_files(self):
        from pgmuvi.lightcurve import Lightcurve

        namespace = {"__name__": "__main__"}
        old_cwd = Path.cwd()
        old_default_dtype = torch.get_default_dtype()
        old_numpy_random_state = np.random.get_state()
        old_torch_random_state = torch.random.get_rng_state()
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            Lightcurve,
            "fit",
            side_effect=AssertionError("default notebook path must not fit"),
        ):
            os.chdir(tmpdir)
            try:
                for index, cell in enumerate(load_notebook().get("cells", [])):
                    if cell.get("cell_type") != "code":
                        continue
                    source = "".join(cell.get("source", []))
                    exec(
                        compile(
                            source,
                            f"tutorial_wavelength_constraints.ipynb:cell-{index}",
                            "exec",
                        ),
                        namespace,
                    )
            finally:
                os.chdir(old_cwd)
                torch.set_default_dtype(old_default_dtype)
                np.random.set_state(old_numpy_random_state)
                torch.random.set_rng_state(old_torch_random_state)
            self.assertEqual(list(Path(tmpdir).iterdir()), [])

        self.assertEqual(torch.get_default_dtype(), old_default_dtype)
        self.assertFalse(namespace["RUN_REDUCED_SYNTHETIC_FITS"])
        self.assertLess(namespace["round_trip_max_relative_error"], 1.0e-12)
        self.assertTrue(namespace["wavelength_diagnostics"].available)
        self.assertEqual(
            tuple(namespace["ARD_COORDINATE_ORDER"]),
            ("temporal_frequency", "wavelength_frequency"),
        )
        self.assertEqual(
            tuple(namespace["REDUCED_FIT_MODELS"]),
            ("2DWavelengthDependent", "2D"),
        )
        self.assertEqual(
            {row["scenario"] for row in namespace["mean_structure_rows"]},
            {
                "d2-reference-wavelength-quadratic-moderate",
                "d2-reference-wavelength-quadratic-strong-turning",
            },
        )
        self.assertEqual(namespace["fit_results"]["status"], "disabled")
        self.assertEqual(namespace["pr133_calibration_summary"]["canonical_runs"], 420)

    def test_notebook_links_relevant_guides_and_api_pages(self):
        text = notebook_text()
        for target in [
            "../pgmuvi.wavelength_estimation.rst",
            "../pgmuvi.spectral_mixture_ard.rst",
            "../pgmuvi.spectral_mixture_ard_diagnostics.rst",
            "../howto/wavelength_models.rst",
            "../howto/priors_constraints.rst",
            "../howto/consensus_fitting.rst",
            "../howto/interpreting_results.rst",
            "../pgmuvi.wavelength_validation_robustness_calibration.rst",
        ]:
            with self.subTest(target=target):
                self.assertIn(target, text)
                self.assertTrue((NOTEBOOK.parent / target).resolve().is_file())

    def test_companion_documentation_links_notebook(self):
        for path in [PRIORS, MODELS, CALIBRATION]:
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                ":doc:`../notebooks/tutorial_wavelength_constraints`"
                if path in {PRIORS, MODELS}
                else ":doc:`notebooks/tutorial_wavelength_constraints`",
                text,
            )


if __name__ == "__main__":
    unittest.main()
