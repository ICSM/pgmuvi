from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".github" / "pull_request_template.md"


class TestPullRequestTemplate(unittest.TestCase):
    def test_template_exists(self):
        self.assertTrue(TEMPLATE.exists())

    def test_template_mentions_validation_commands(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("PYTHONPATH=.", text)
        self.assertIn("make clean && make html-strict", text)
        self.assertIn("git diff --check", text)
        self.assertIn("git status --short", text)

    def test_template_mentions_generated_artifact_hygiene(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        for phrase in [
            "Generated artifacts were not committed",
            "review ZIPs",
            "debug logs",
            "Sphinx builds",
            "patch backups",
        ]:
            self.assertIn(phrase, text)

    def test_template_points_to_snapshot_helper(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("scripts/create_source_snapshot.py", text)
        self.assertIn("pgmuvi_current.zip", text)

    def test_template_has_documentation_impact_section(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("## Documentation impact", text)
        self.assertIn("API reference or docstring coverage", text)
        self.assertIn("Contributor or maintenance documentation", text)


if __name__ == "__main__":
    unittest.main()
