from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SOURCE = DOCS / "source"


class TestDocsStrictBuildContract(unittest.TestCase):
    def test_makefile_has_warning_as_error_html_target(self):
        makefile = (DOCS / "Makefile").read_text()

        self.assertIn("html-strict:", makefile)
        self.assertIn("-W --keep-going", makefile)
        self.assertIn(
            '$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"',
            makefile,
        )

    def test_maintenance_page_is_linked_from_root_index(self):
        index = (SOURCE / "index.rst").read_text()

        self.assertIn("Developer documentation", index)
        self.assertIn("docs_maintenance", index)

    def test_maintenance_page_documents_clean_strict_build(self):
        page = (SOURCE / "docs_maintenance.rst").read_text()

        self.assertIn("make clean", page)
        self.assertIn("make html-strict", page)
        self.assertIn("warnings as errors", page)
        self.assertIn("notebook_status.rst", page)


if __name__ == "__main__":
    unittest.main()
