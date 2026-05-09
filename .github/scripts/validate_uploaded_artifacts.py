#!/usr/bin/env python3
"""Validate uploaded distribution artifacts against stale-version ambiguity."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

from pgmuvi.upload_validation import (
    UPLOAD_VALIDATION_SENTINEL,
    assert_no_duplicate_uploaded_files,
    assert_selected_files_are_latest,
)

_DIST_VERSION_RE = re.compile(r"^pgmuvi-(?P<version>[^-]+)")


def _extract_version(path):
    base = os.path.basename(path)
    match = _DIST_VERSION_RE.match(base)
    if match is None:
        raise RuntimeError(f"Could not extract package version from {base!r}.")
    return match.group("version")


def _build_records(paths, logical_name, sentinel):
    records = []
    for path in sorted(paths):
        stat_result = os.stat(path)
        records.append({
            "logical_name": logical_name,
            "file_path": path,
            "revision": int(stat_result.st_mtime_ns),
            "uploaded_at": int(stat_result.st_mtime_ns),
            "sentinel": sentinel,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-sentinel", required=True)
    args = parser.parse_args()
    expected_version = args.expected_version.lstrip("v")

    sentinel_path = os.path.join(args.dist_dir, ".upload_validation_sentinel")
    if not os.path.exists(sentinel_path):
        raise RuntimeError("Missing upload-validation sentinel file in dist directory.")
    sentinel_value = open(sentinel_path, encoding="utf-8").read().strip()
    if sentinel_value != args.expected_sentinel:
        raise RuntimeError(
            "Upload-validation sentinel mismatch; artifact may come from a stale run."
        )
    if not sentinel_value.endswith(UPLOAD_VALIDATION_SENTINEL):
        raise RuntimeError("Sentinel version token mismatch in uploaded artifacts.")

    wheel_paths = glob.glob(os.path.join(args.dist_dir, "*.whl"))
    sdist_paths = glob.glob(os.path.join(args.dist_dir, "*.tar.gz"))
    if not wheel_paths or not sdist_paths:
        raise RuntimeError("Expected both wheel and sdist artifacts in dist directory.")

    records = []
    records.extend(_build_records(wheel_paths, "wheel", sentinel_value))
    records.extend(_build_records(sdist_paths, "sdist", sentinel_value))
    assert_no_duplicate_uploaded_files(records)

    selected = {
        "wheel": max(wheel_paths, key=lambda p: os.stat(p).st_mtime_ns),
        "sdist": max(sdist_paths, key=lambda p: os.stat(p).st_mtime_ns),
    }
    assert_selected_files_are_latest(records, selected)

    for path in selected.values():
        version = _extract_version(path)
        if version != expected_version:
            raise RuntimeError(
                f"Artifact version mismatch for {path!r}: expected "
                f"{expected_version!r}, got {version!r}."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
