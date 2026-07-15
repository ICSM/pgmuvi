"""Regression contracts for completing the public notebook refresh."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "source"
NOTEBOOKS = SOURCE / "notebooks"


class TestNotebookRefreshCompletion(unittest.TestCase):
    def test_dead_mcmc_notebook_file_is_removed(self) -> None:
        self.assertFalse((NOTEBOOKS / "pgmuvi_tutorial_mcmc.ipynb").exists())

    def test_docs_do_not_reference_deleted_notebook_path(self) -> None:
        stale = "notebooks/pgmuvi_tutorial_mcmc"
        offenders = []
        for path in SOURCE.rglob("*"):
            if path.suffix not in {".rst", ".ipynb", ".py"}:
                continue
            if stale in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_general_tutorial_states_current_mcmc_boundary(self) -> None:
        path = NOTEBOOKS / "pgmuvi_tutorial.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertIn("Lightcurve.mcmc", text)
        self.assertIn("NotImplementedError", text)
        self.assertNotIn("how to use MCMC to sample", text)

    def test_roadmap_marks_notebook_refresh_complete(self) -> None:
        text = (SOURCE / "documentation_roadmap.rst").read_text(encoding="utf-8")
        self.assertIn("Completed in PR103--PR108", text)
        self.assertIn("unavailable MCMC notebook was deleted", text)

    def test_status_retains_future_work_without_nonfunctional_notebook(self) -> None:
        text = (SOURCE / "notebook_status.rst").read_text(encoding="utf-8")
        self.assertIn("TBD[mcmc-implementation]", text)
        self.assertIn("Do not restore the deleted notebook verbatim", text)
        self.assertIn("No quarantined notebook files remain", text)


if __name__ == "__main__":
    unittest.main()
