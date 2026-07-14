"""Regression tests for the strict documentation CI workflow."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "docs.yml"
MAINTENANCE_PAGE = ROOT / "docs" / "source" / "docs_maintenance.rst"


class TestStrictDocsCIWorkflow(unittest.TestCase):
    def test_workflow_exists(self) -> None:
        self.assertTrue(WORKFLOW.exists())

    def test_workflow_runs_strict_docs_target(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r"make(?:\s+-C\s+docs)?\s+html-strict",
        )
        self.assertIn("docs/source/requirements.txt", text)
        self.assertIn("actions/setup-python@v5", text)
        self.assertIn("actions/checkout@v4", text)

    def test_workflow_runs_on_push_and_pull_request(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^on:\s*$")
        self.assertRegex(text, r"(?m)^  push:\s*$")
        self.assertRegex(text, r"(?m)^  pull_request:\s*$")

    def test_workflow_does_not_use_non_strict_docs_target(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotRegex(text, re.compile(r"make\s+html(?!-strict)"))

    def test_maintenance_page_mentions_ci_contract(self) -> None:
        text = MAINTENANCE_PAGE.read_text(encoding="utf-8")
        self.assertIn("html-strict", text)
        self.assertIn(".github/workflows/docs.yml", text)


if __name__ == "__main__":
    unittest.main()
