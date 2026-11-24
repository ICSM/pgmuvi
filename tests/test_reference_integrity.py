"""Cross-references in the repository point at things that exist.

Three checks, each over the tracked files: Sphinx ``:doc:`` roles resolve to a
document, local Markdown links resolve to a file, and no public file carries a
path from someone's machine. Rewritten small from the archived stack's
``test_repository_reference_integrity.py`` (roadmap 1.3).

Deliberately small: inline Markdown links only (reference-style links are not
checked and fenced code is not excluded), ``:doc:`` roles in ``.rst`` files only,
and machine paths in four public directories (not README.md or docs/dev).
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "source"
DOC_ROLE = re.compile(r":doc:`(?:[^`<]*<)?([^`>]+)>?`")
MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)")
MACHINE_PATH = re.compile(r"/Users/|/Volumes/|/home/[\w.-]+/|file:///|[A-Za-z]:[\\/]+Users[\\/]")
EXTERNAL = ("http://", "https://", "mailto:", "#")


def _tracked(suffix=None):
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    paths = [ROOT / p.decode() for p in out.split(b"\0") if p]
    return [p for p in paths if suffix is None or p.suffix == suffix]


def _lineno(text, pos):
    return text.count("\n", 0, pos) + 1


def test_sphinx_doc_roles_resolve():
    bad = []
    for rst in _tracked(".rst"):
        text = rst.read_text(encoding="utf-8")
        for m in DOC_ROLE.finditer(text):
            target = m.group(1).strip()
            base = DOCS / target.lstrip("/") if target.startswith("/") else rst.parent / target
            if not any(
                (base.parent / (base.name + s)).is_file() for s in (".rst", ".ipynb", ".md")
            ):
                bad.append(f"{rst.relative_to(ROOT)}:{_lineno(text, m.start())}: {target}")
    assert not bad, "Unresolved :doc: targets:\n" + "\n".join(bad)


def test_local_markdown_links_resolve():
    bad = []
    for md in _tracked(".md"):
        text = re.sub(r"<!--.*?-->", "", md.read_text(encoding="utf-8"), flags=re.S)
        for m in MD_LINK.finditer(text):
            target = m.group(1).split("#")[0].strip("<>")
            if not target or target.startswith(EXTERNAL) or "{" in target:
                continue
            if not (md.parent / target).exists():
                bad.append(f"{md.relative_to(ROOT)}:{_lineno(text, m.start())}: {target}")
    assert not bad, "Broken local Markdown links:\n" + "\n".join(bad)


@pytest.mark.parametrize("prefix", ["pgmuvi/", "docs/source/", "examples/", "paper/"])
def test_no_machine_local_paths(prefix):
    bad = []
    for path in _tracked():
        rel = path.relative_to(ROOT).as_posix()
        if not rel.startswith(prefix):
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        for m in MACHINE_PATH.finditer(text):
            bad.append(f"{rel}:{_lineno(text, m.start())}: {m.group(0)}")
    assert not bad, "Machine-local paths in public files:\n" + "\n".join(bad)
