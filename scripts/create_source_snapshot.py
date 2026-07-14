#!/usr/bin/env python3
"""Create a clean source snapshot from a committed Git tree.

This helper intentionally uses ``git archive`` instead of zipping the working
copy. The resulting archive contains only files tracked at the requested Git
reference, so it excludes local caches, documentation builds, debug logs,
patch backups, and the ``.git`` directory.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _repository_root() -> Path:
    return Path(_run_git(["rev-parse", "--show-toplevel"]))


def create_snapshot(output: Path, ref: str) -> None:
    root = _repository_root()
    output = output.expanduser()
    if not output.is_absolute():
        output = root / output

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "archive", "--format=zip", "-o", str(output), ref],
        cwd=root,
        check=True,
    )

    commit = _run_git(["rev-parse", "--short", ref])
    print(f"Wrote {output}")
    print(f"Source reference: {ref} ({commit})")
    print("Archive was created from tracked files only; local generated artifacts are excluded.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clean pgmuvi source snapshot from a committed Git tree."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="pgmuvi_current.zip",
        help="Output ZIP path, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Git reference to archive; defaults to HEAD.",
    )
    args = parser.parse_args()
    create_snapshot(Path(args.output), args.ref)


if __name__ == "__main__":
    main()
