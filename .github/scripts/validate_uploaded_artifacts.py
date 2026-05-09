#!/usr/bin/env python3
"""Validate uploaded distribution artifacts against stale-version ambiguity."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

_DIST_VERSION_RE = re.compile(r"^pgmuvi-(?P<version>[^-]+)")
UPLOAD_VALIDATION_SENTINEL = "PGMUVI_UPLOAD_VALIDATION_V1"


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


def _resolve_latest_uploaded_files(uploaded_files):
    grouped = {}
    for index, rec in enumerate(uploaded_files):
        logical_name = rec["logical_name"]
        current = grouped.get(logical_name)
        candidate_key = (rec["revision"], rec["uploaded_at"], index)
        if current is None or candidate_key > current[0]:
            grouped[logical_name] = (candidate_key, rec)
    return {logical_name: entry[1] for logical_name, entry in grouped.items()}


def _assert_no_duplicate_uploaded_files(uploaded_files):
    grouped = {}
    for rec in uploaded_files:
        grouped.setdefault(rec["logical_name"], []).append(rec["file_path"])
    duplicate_map = {
        logical_name: sorted(paths)
        for logical_name, paths in grouped.items()
        if len(paths) > 1
    }
    if duplicate_map:
        raise RuntimeError(
            f"Detected duplicate uploaded files for logical names: {duplicate_map}"
        )


def _assert_selected_files_are_latest(uploaded_files, selected_files):
    latest = _resolve_latest_uploaded_files(uploaded_files)
    for logical_name, latest_record in latest.items():
        selected_path = selected_files.get(logical_name)
        if selected_path is None:
            raise RuntimeError(
                f"Missing selected file for logical name {logical_name!r}."
            )
        if str(selected_path) != latest_record["file_path"]:
            raise RuntimeError(
                f"Selected uploaded file for logical name {logical_name!r} "
                f"is stale: expected {latest_record['file_path']!r}, "
                f"got {selected_path!r}."
            )


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
    with open(sentinel_path, encoding="utf-8") as sentinel_file:
        sentinel_value = sentinel_file.read().strip()
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
    _assert_no_duplicate_uploaded_files(records)

    selected = {
        "wheel": max(wheel_paths, key=lambda p: os.stat(p).st_mtime_ns),
        "sdist": max(sdist_paths, key=lambda p: os.stat(p).st_mtime_ns),
    }
    _assert_selected_files_are_latest(records, selected)

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
