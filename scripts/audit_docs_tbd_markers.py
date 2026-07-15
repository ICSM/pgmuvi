#!/usr/bin/env python3
"""Audit registered documentation TBD markers."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


VALID_MARKER = re.compile(r"TBD\[([a-z0-9]+(?:-[a-z0-9]+)*)\]")
BRACKETED_CANDIDATE = re.compile(r"TBD\[[^\]\r\n]*\]")
TEXT_SUFFIXES = {".rst", ".md", ".txt", ".py", ".ipynb"}


def iter_text_files(source_root: Path):
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def relative_label(path: Path, source_root: Path) -> str:
    try:
        return str(path.relative_to(source_root))
    except ValueError:
        return str(path)


def audit(source_root: Path, registry: Path) -> tuple[list[str], dict[str, list[str]]]:
    errors: list[str] = []
    locations: dict[str, list[str]] = defaultdict(list)

    if not source_root.is_dir():
        return [f"source root does not exist: {source_root}"], locations
    if not registry.is_file():
        return [f"registry does not exist: {registry}"], locations

    registry_text = registry.read_text(encoding="utf-8")
    registry_markers = set(VALID_MARKER.findall(registry_text))
    if not registry_markers:
        errors.append(f"registry contains no valid TBD markers: {registry}")

    for path in iter_text_files(source_root):
        text = path.read_text(encoding="utf-8")
        label = relative_label(path, source_root)

        valid_spans = {match.span() for match in VALID_MARKER.finditer(text)}
        for candidate in BRACKETED_CANDIDATE.finditer(text):
            if candidate.span() not in valid_spans:
                errors.append(
                    f"malformed marker in {label}: {candidate.group(0)!r}"
                )

        for match in VALID_MARKER.finditer(text):
            marker = match.group(1)
            locations[marker].append(label)

    used_markers = set(locations)
    unregistered = sorted(used_markers - registry_markers)
    for marker in unregistered:
        errors.append(f"unregistered marker: TBD[{marker}]")

    registry_label = relative_label(registry, source_root)
    for marker in sorted(registry_markers):
        owners = [label for label in locations.get(marker, []) if label != registry_label]
        if not owners:
            errors.append(
                f"registered marker has no user-facing owner: TBD[{marker}]"
            )

    return errors, locations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit documentation TBD markers against the registry."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("docs/source"),
        help="documentation source directory (default: docs/source)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("docs/source/future_work.rst"),
        help="future-work registry file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors, locations = audit(args.source_root, args.registry)

    if errors:
        print("TBD marker audit failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("TBD marker audit passed.")
    print(f"registered markers: {len(locations)}")
    for marker in sorted(locations):
        unique_locations = sorted(set(locations[marker]))
        print(f"  TBD[{marker}]: {', '.join(unique_locations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
