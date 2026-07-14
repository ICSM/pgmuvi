"""Regression checks for the documentation expansion roadmap."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "source"


class TestDocumentationExpansionRoadmap(unittest.TestCase):
    def test_roadmap_page_exists_and_defines_scope(self):
        page = (SOURCE / "documentation_roadmap.rst").read_text(encoding="utf-8")

        self.assertIn("Documentation expansion roadmap", page)
        self.assertIn("separate", page)
        self.assertIn("documentation build cleanup", page)
        self.assertIn("wavelength-advisory synchronization", page)
        self.assertIn("self-contained chunks", page)

    def test_roadmap_lists_required_documentation_layers(self):
        page = (SOURCE / "documentation_roadmap.rst").read_text(encoding="utf-8")

        required = [
            "Concept background",
            "Runnable Python script",
            "Notebook tutorial",
            "API reference",
            "TBD markers",
            "make html-strict",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, page)

    def test_roadmap_covers_remaining_workflow_areas(self):
        page = (SOURCE / "documentation_roadmap.rst").read_text(encoding="utf-8")

        required = [
            "Core package orientation",
            "Lightcurve creation and validation",
            "Single-source fitting",
            "Wavelength-dependent model guidance",
            "Batch advisory workflows",
            "Result interpretation",
            "Notebook refresh",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, page)

    def test_roadmap_is_linked_from_public_indices(self):
        root_index = (SOURCE / "index.rst").read_text(encoding="utf-8")
        howto_index = (SOURCE / "howto" / "index.rst").read_text(encoding="utf-8")

        self.assertIn("documentation_roadmap", root_index)
        self.assertIn("documentation_roadmap", howto_index)


if __name__ == "__main__":
    unittest.main()
