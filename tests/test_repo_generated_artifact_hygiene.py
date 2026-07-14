"""Regression tests for local generated-artifact hygiene."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = ROOT / ".gitignore"
MAINTENANCE_PAGE = ROOT / "docs" / "source" / "docs_maintenance.rst"


class TestGeneratedArtifactHygiene(unittest.TestCase):
    def test_generated_artifact_patterns_are_ignored(self) -> None:
        text = GITIGNORE.read_text(encoding="utf-8")
        for pattern in (
            ".ruff_cache/",
            "docs/build/",
            "debug*.txt",
            "*.orig",
            "*.rej",
            "pgmuvi_current*.zip",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, text)

    def test_docs_explain_clean_source_snapshots(self) -> None:
        text = MAINTENANCE_PAGE.read_text(encoding="utf-8")
        self.assertIn("git archive --format=zip -o pgmuvi_current.zip HEAD", text)
        self.assertIn("Generated artifacts and source snapshots", text)
        self.assertIn("docs/build/", text)
        self.assertIn("debug*.txt", text)


if __name__ == "__main__":
    unittest.main()
