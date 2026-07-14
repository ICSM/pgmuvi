from pathlib import Path
import unittest


class TestDocsRequirementsContract(unittest.TestCase):
    def test_docs_requirements_declares_build_dependencies(self):
        text = Path("docs/source/requirements.txt").read_text(encoding="utf-8")

        required_lines = {
            "-e .",
            "sphinx>=5.3",
            "nbsphinx",
            "ipykernel",
        }
        lines = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}

        missing = sorted(required_lines - lines)
        self.assertEqual(missing, [])

    def test_readthedocs_uses_docs_requirements(self):
        text = Path(".readthedocs.yaml").read_text(encoding="utf-8")

        self.assertIn("requirements: docs/source/requirements.txt", text)

    def test_docs_ci_uses_docs_requirements_file(self):
        text = Path(".github/workflows/docs.yml").read_text(encoding="utf-8")

        self.assertIn("python -m pip install -r docs/source/requirements.txt", text)
        self.assertIn("make -C docs html-strict", text)
        self.assertNotIn("python -m pip install sphinx nbsphinx ipykernel", text)

    def test_docs_maintenance_points_to_requirements_file(self):
        text = Path("docs/source/docs_maintenance.rst").read_text(encoding="utf-8")

        self.assertIn("Documentation dependencies", text)
        self.assertIn("docs/source/requirements.txt", text)
        self.assertIn("Read the Docs", text)
        self.assertIn("GitHub Actions", text)


if __name__ == "__main__":
    unittest.main()
