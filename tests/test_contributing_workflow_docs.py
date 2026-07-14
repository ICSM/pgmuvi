from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTING = ROOT / "CONTRIBUTING.md"


class TestContributingWorkflowDocs(unittest.TestCase):
    def test_contributing_mentions_strict_docs_contract(self) -> None:
        text = CONTRIBUTING.read_text(encoding="utf-8")
        self.assertIn("Documentation build and review workflow", text)
        self.assertIn("docs/source/requirements.txt", text)
        self.assertIn("make html-strict", text)
        self.assertIn("warnings as errors", text)

    def test_contributing_points_to_clean_snapshot_helper(self) -> None:
        text = CONTRIBUTING.read_text(encoding="utf-8")
        self.assertIn("scripts/create_source_snapshot.py", text)
        self.assertIn("pgmuvi_current.zip", text)
        self.assertIn("tracked files only", text)

    def test_contributing_warns_against_committing_local_artifacts(self) -> None:
        text = CONTRIBUTING.read_text(encoding="utf-8")
        for expected in [
            "debug_pr*.txt",
            "docs/build/",
            "pgmuvi_current*.zip",
            "Python cache directories",
        ]:
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
