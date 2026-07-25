"""Regression checks for repository-local reference hygiene."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MACHINE_LOCAL_PATH = re.compile(
    r"(/Users/|/Volumes/|/home/|[A-Za-z]:\\Users\\|file://)"
)


class TestRepositoryReferenceHygiene(unittest.TestCase):
    def test_removed_test_script_is_not_referenced(self):
        text = "\n".join(
            [
                (ROOT / ".github/copilot-instructions.md").read_text(
                    encoding="utf-8"
                ),
                (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
            ]
        )
        self.assertNotIn("pgmuvi/test_script.py", text)
        self.assertFalse((ROOT / "pgmuvi/test_script.py").exists())

    def test_batch_docs_use_existing_public_source(self):
        text = (
            ROOT / "docs/source/howto/wavelength_advisory_batch.rst"
        ).read_text(encoding="utf-8")
        self.assertNotIn("07454-7112.csv", text)
        self.assertIn("examples/data/10131+3049.csv", text)
        self.assertTrue(
            (ROOT / "examples/data/10131+3049.csv").is_file()
        )

    def test_notebook_outputs_have_no_machine_local_paths(self):
        violations = []

        for path in sorted(
            (ROOT / "docs/source/notebooks").glob("*.ipynb")
        ):
            payload = json.loads(path.read_text(encoding="utf-8"))
            relative = path.relative_to(ROOT).as_posix()

            sections = [
                ("metadata", payload.get("metadata", {})),
            ]

            for cell_index, cell in enumerate(payload.get("cells", [])):
                sections.append(
                    (
                        f"cell[{cell_index}].metadata",
                        cell.get("metadata", {}),
                    )
                )
                for output_index, output in enumerate(
                    cell.get("outputs", [])
                ):
                    sections.append(
                        (
                            f"cell[{cell_index}].outputs[{output_index}]",
                            output,
                        )
                    )

            for location, content in sections:
                serialized = json.dumps(content, ensure_ascii=False)
                if MACHINE_LOCAL_PATH.search(serialized):
                    violations.append(f"{relative}:{location}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
