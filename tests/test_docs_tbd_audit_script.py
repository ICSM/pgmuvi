"""Tests for the documentation TBD-marker audit script."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_docs_tbd_markers.py"


class TestDocumentationTbdAuditScript(unittest.TestCase):
    def run_audit(self, source_root: Path, registry: Path):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--source-root",
                str(source_root),
                "--registry",
                str(registry),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_repository_registry_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TBD marker audit passed", result.stdout)
        self.assertIn("TBD[automatic-model-selection]", result.stdout)

    def test_unregistered_marker_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            registry = source / "future_work.rst"
            registry.write_text("TBD[registered-item]\n", encoding="utf-8")
            (source / "owner.rst").write_text(
                "TBD[registered-item]\nTBD[unregistered-item]\n",
                encoding="utf-8",
            )

            result = self.run_audit(source, registry)
            self.assertEqual(result.returncode, 1)
            self.assertIn("unregistered marker", result.stderr)

    def test_malformed_marker_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            registry = source / "future_work.rst"
            registry.write_text("TBD[registered-item]\n", encoding="utf-8")
            (source / "owner.rst").write_text(
                "TBD[registered-item]\nTBD[Bad Marker]\n",
                encoding="utf-8",
            )

            result = self.run_audit(source, registry)
            self.assertEqual(result.returncode, 1)
            self.assertIn("malformed marker", result.stderr)

    def test_registered_marker_without_owner_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            registry = source / "future_work.rst"
            registry.write_text("TBD[orphaned-item]\n", encoding="utf-8")

            result = self.run_audit(source, registry)
            self.assertEqual(result.returncode, 1)
            self.assertIn("has no user-facing owner", result.stderr)


if __name__ == "__main__":
    unittest.main()
