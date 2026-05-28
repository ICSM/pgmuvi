"""Helpers to validate uploaded file-version selection deterministically.

Stale uploaded files are dangerous in validation/review workflows because they
can make checks pass against outdated artifacts instead of the newest output.
This module enforces explicit revision/sentinel checks so ambiguous filename
matching cannot silently select old files.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict


UPLOAD_VALIDATION_SENTINEL = "PGMUVI_UPLOAD_VALIDATION_V1"


@dataclass(frozen=True)
class UploadedFileVersion:
    """Typed uploaded-file record used by validation helpers."""

    logical_name: str
    file_path: str
    revision: int
    uploaded_at: int
    sentinel: str = UPLOAD_VALIDATION_SENTINEL


def _coerce_uploaded_file_record(record):
    """Return *record* as :class:`UploadedFileVersion`."""
    if isinstance(record, UploadedFileVersion):
        return record
    if not isinstance(record, dict):
        raise TypeError("uploaded file records must be dicts or UploadedFileVersion.")
    return UploadedFileVersion(
        logical_name=str(record["logical_name"]),
        file_path=str(record["file_path"]),
        revision=int(record["revision"]),
        uploaded_at=int(record["uploaded_at"]),
        sentinel=str(record.get("sentinel", UPLOAD_VALIDATION_SENTINEL)),
    )


def resolve_latest_uploaded_files(uploaded_files):
    """Resolve latest uploaded files per logical name with deterministic ties."""
    grouped = defaultdict(list)
    for raw in uploaded_files:
        rec = _coerce_uploaded_file_record(raw)
        if rec.sentinel != UPLOAD_VALIDATION_SENTINEL:
            raise RuntimeError(
                "Uploaded file record has an unexpected sentinel and may be stale."
            )
        grouped[rec.logical_name].append(rec)

    latest = {}
    for logical_name, records in grouped.items():
        if not records:
            continue
        latest[logical_name] = max(
            records,
            key=lambda rec: (rec.revision, rec.uploaded_at, rec.file_path),
        )
    return latest


def assert_no_duplicate_uploaded_files(uploaded_files):
    """Fail loudly when multiple uploaded copies share one logical name."""
    grouped = defaultdict(list)
    for raw in uploaded_files:
        rec = _coerce_uploaded_file_record(raw)
        grouped[rec.logical_name].append(rec.file_path)
    duplicate_map = {
        logical_name: sorted(paths)
        for logical_name, paths in grouped.items()
        if len(paths) > 1
    }
    if duplicate_map:
        raise RuntimeError(
            f"Detected duplicate uploaded files for logical names: {duplicate_map}"
        )


def assert_selected_files_are_latest(uploaded_files, selected_files):
    """Fail when selected files are not the deterministic latest versions."""
    latest = resolve_latest_uploaded_files(uploaded_files)
    for logical_name, latest_record in latest.items():
        selected_path = selected_files.get(logical_name)
        if selected_path is None:
            raise RuntimeError(
                f"Missing selected file for logical name {logical_name!r}; expected "
                f"latest path {latest_record.file_path!r}."
            )
        if str(selected_path) != latest_record.file_path:
            raise RuntimeError(
                f"Selected uploaded file for logical name {logical_name!r} is stale: "
                f"expected {latest_record.file_path!r}, got {selected_path!r}."
            )


def validate_uploaded_file_selection(
    uploaded_files,
    selected_files,
    *,
    allow_duplicates=False,
):
    """Validate deterministic latest-file selection for review/validation steps.

    Stale uploaded artifacts are dangerous during review because checks may pass
    against old outputs rather than the newest upload. This helper enforces a
    deterministic latest-file resolution and prevents ambiguous filename-only
    matching from silently selecting outdated files.
    """
    latest = resolve_latest_uploaded_files(uploaded_files)
    if not allow_duplicates:
        assert_no_duplicate_uploaded_files(uploaded_files)
    assert_selected_files_are_latest(uploaded_files, selected_files)
    return latest
