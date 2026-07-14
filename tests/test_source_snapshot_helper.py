"""Regression tests for the clean source snapshot helper."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_SCRIPT = ROOT / "scripts" / "create_source_snapshot.py"
MAINTENANCE_PAGE = ROOT / "docs" / "source" / "docs_maintenance.rst"
GITIGNORE = ROOT / ".gitignore"


class TestSourceSnapshotHelper(unittest.TestCase):
    def test_helper_exists_and_is_valid_python(self) -> None:
        text = SNAPSHOT_SCRIPT.read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn("git", text)
        self.assertIn("archive", text)
        self.assertIn("--format=zip", text)
        self.assertIn("pgmuvi_current.zip", text)

    def test_helper_uses_tracked_git_tree_not_working_directory_zip(self) -> None:
        text = SNAPSHOT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("git", text)
        self.assertIn("archive", text)
        self.assertNotIn("shutil.make_archive", text)
        self.assertNotIn("zipfile.ZipFile", text)

    def test_maintenance_docs_reference_helper_and_git_archive(self) -> None:
        text = MAINTENANCE_PAGE.read_text(encoding="utf-8")
        self.assertIn("scripts/create_source_snapshot.py", text)
        self.assertIn("git archive --format=zip", text)
        self.assertIn("tracked files only", text)

    def test_default_snapshot_output_is_ignored(self) -> None:
        text = GITIGNORE.read_text(encoding="utf-8")
        self.assertIn("pgmuvi_current*.zip", text)


if __name__ == "__main__":
    unittest.main()
