from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = ROOT / "docs" / "source"

DOC_ROLE = re.compile(r":doc:`(?:[^`<]*<)?([^`>]+)>?`")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
LOCAL_ACTION_USE = re.compile(
    r"^\s*uses:\s*(\./[^\s#]+)",
    re.MULTILINE,
)
MACHINE_LOCAL_REFERENCE = re.compile(
    r"(?:"
    r"/Users/"
    r"|/Volumes/"
    r"|/home/[A-Za-z0-9_.-]+/"
    r"|file:///"
    r"|[A-Za-z]:\\Users\\"
    r")"
)

FORBIDDEN_PUBLIC_DATASET = "07454-7112.csv"
PUBLIC_PREFIXES = (
    ".github/",
    "docs/",
    "examples/",
    "paper/",
    "pgmuvi/",
)
PUBLIC_ROOT_FILES = {
    "CITATION.cff",
    "README.md",
    "paper.md",
    "pyproject.toml",
}


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        ROOT / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def _strip_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _clean_target(raw_target: str) -> str:
    target = unquote(raw_target.strip())
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    elif " " in target:
        target = target.split(" ", 1)[0]
    target = target.strip().strip("<>")
    target = target.split("#", 1)[0]
    target = target.split("?", 1)[0]
    return target.strip()


def _skip_target(target: str) -> bool:
    lowered = target.lower()
    return (
        not target
        or target.startswith("#")
        or target.startswith("|")
        or lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("mailto:")
        or lowered.startswith("data:")
        or lowered.startswith("javascript:")
        or "://" in target
        or "{" in target
        or "}" in target
        or "*" in target
        or ("[" in target and "]" in target)
    )


def _inside(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve(strict=False).relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _document_candidates(source_file: Path, target: str) -> list[Path]:
    if target.startswith("/"):
        candidate = DOCS_SOURCE / target.lstrip("/")
    else:
        candidate = source_file.parent / target
    if candidate.suffix:
        return [candidate]
    return [
        candidate.with_suffix(".rst"),
        candidate.with_suffix(".ipynb"),
        candidate.with_suffix(".md"),
    ]


class TestRepositoryReferenceIntegrity(unittest.TestCase):
    def test_all_sphinx_doc_targets_exist(self) -> None:
        failures: list[str] = []
        for source_file in sorted(DOCS_SOURCE.rglob("*.rst")):
            text = source_file.read_text(encoding="utf-8")
            for match in DOC_ROLE.finditer(text):
                target = _clean_target(match.group(1))
                if _skip_target(target):
                    continue
                candidates = _document_candidates(source_file, target)
                if any(
                    _inside(candidate, DOCS_SOURCE)
                    and candidate.is_file()
                    for candidate in candidates
                ):
                    continue
                line_number = text.count("\n", 0, match.start(1)) + 1
                failures.append(
                    f"{source_file.relative_to(ROOT)}:"
                    f"{line_number}: {target}"
                )
        self.assertEqual(
            failures,
            [],
            "Unresolved :doc: targets:\n" + "\n".join(failures),
        )

    def test_local_markdown_links_exist(self) -> None:
        failures: list[str] = []
        for source_file in [
            path
            for path in _tracked_files()
            if path.suffix.lower() == ".md"
        ]:
            text = _strip_html_comments(
                source_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )
            for match in MARKDOWN_LINK.finditer(text):
                target = _clean_target(match.group(1))
                if _skip_target(target):
                    continue
                candidate = (source_file.parent / target).resolve(
                    strict=False
                )
                if _inside(candidate, ROOT) and candidate.exists():
                    continue
                line_number = text.count("\n", 0, match.start(1)) + 1
                failures.append(
                    f"{source_file.relative_to(ROOT)}:"
                    f"{line_number}: {target}"
                )
        self.assertEqual(
            failures,
            [],
            "Broken local Markdown links:\n" + "\n".join(failures),
        )

    def test_local_github_actions_exist(self) -> None:
        failures: list[str] = []
        workflow_files = [
            *ROOT.glob(".github/workflows/*.yml"),
            *ROOT.glob(".github/workflows/*.yaml"),
        ]
        for source_file in sorted(workflow_files):
            text = source_file.read_text(encoding="utf-8")
            for match in LOCAL_ACTION_USE.finditer(text):
                target = match.group(1)
                candidate = ROOT / target.removeprefix("./")
                if candidate.exists():
                    continue
                line_number = text.count("\n", 0, match.start(1)) + 1
                failures.append(
                    f"{source_file.relative_to(ROOT)}:"
                    f"{line_number}: {target}"
                )
        self.assertEqual(
            failures,
            [],
            "Missing local GitHub actions:\n" + "\n".join(failures),
        )

    def test_public_files_have_no_machine_paths(self) -> None:
        failures: list[str] = []
        for source_file in _tracked_files():
            relative = str(source_file.relative_to(ROOT))
            is_public = (
                relative in PUBLIC_ROOT_FILES
                or relative.startswith(PUBLIC_PREFIXES)
            )
            if not is_public or relative.startswith("tests/"):
                continue
            data = source_file.read_bytes()
            if b"\0" in data:
                continue
            text = data.decode("utf-8", errors="replace")
            for match in MACHINE_LOCAL_REFERENCE.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{relative}:{line_number}: {match.group(0)}"
                )
        self.assertEqual(
            failures,
            [],
            "Machine-local paths in public files:\n"
            + "\n".join(failures),
        )

    def test_forbidden_dataset_is_absent_from_public_files(self) -> None:
        failures: list[str] = []
        for source_file in _tracked_files():
            relative = str(source_file.relative_to(ROOT))
            is_public = (
                relative in PUBLIC_ROOT_FILES
                or relative.startswith(PUBLIC_PREFIXES)
            )
            if not is_public or relative.startswith("tests/"):
                continue
            data = source_file.read_bytes()
            if b"\0" in data:
                continue
            text = data.decode("utf-8", errors="replace")
            if FORBIDDEN_PUBLIC_DATASET in text:
                failures.append(relative)
        self.assertEqual(
            failures,
            [],
            "Forbidden dataset references:\n" + "\n".join(failures),
        )

    def test_2d_guides_match_executed_workflow(self) -> None:
        consensus = (
            DOCS_SOURCE / "howto" / "consensus_fitting.rst"
        ).read_text(encoding="utf-8")
        multiband = (
            DOCS_SOURCE / "howto" / "multiband.rst"
        ).read_text(encoding="utf-8")
        for text in (consensus, multiband):
            self.assertIn("2DWavelengthDependent", text)
            self.assertIn("Lightcurve.plot()", text)
        self.assertNotIn(
            "explicit no-training default",
            consensus,
        )
        self.assertNotIn(
            "prepares the baseline ``2D``",
            multiband,
        )

    def test_notebook_json_is_parseable(self) -> None:
        for notebook_file in sorted(DOCS_SOURCE.rglob("*.ipynb")):
            with self.subTest(
                notebook=str(notebook_file.relative_to(ROOT))
            ):
                json.loads(
                    notebook_file.read_text(encoding="utf-8")
                )


if __name__ == "__main__":
    unittest.main()
