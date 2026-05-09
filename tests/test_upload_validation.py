"""Regression tests for deterministic uploaded-file version validation."""

import pytest

from pgmuvi.upload_validation import (
    UPLOAD_VALIDATION_SENTINEL,
    assert_no_duplicate_uploaded_files,
    assert_selected_files_are_latest,
    resolve_latest_uploaded_files,
    validate_uploaded_file_selection,
)


def _record(logical_name, file_path, revision, uploaded_at, sentinel=None):
    return {
        "logical_name": logical_name,
        "file_path": file_path,
        "revision": revision,
        "uploaded_at": uploaded_at,
        "sentinel": sentinel or UPLOAD_VALIDATION_SENTINEL,
    }


def test_resolve_latest_single_record_returns_only_file():
    uploaded_files = [_record("artifact", "/tmp/artifact-v1.txt", 1, 10)]
    latest = resolve_latest_uploaded_files(uploaded_files)
    assert latest["artifact"].file_path == "/tmp/artifact-v1.txt"


def test_resolve_latest_selects_highest_revision():
    uploaded_files = [
        _record("artifact", "/tmp/artifact-v1.txt", 1, 20),
        _record("artifact", "/tmp/artifact-v2.txt", 2, 10),
    ]
    latest = resolve_latest_uploaded_files(uploaded_files)
    assert latest["artifact"].file_path == "/tmp/artifact-v2.txt"


def test_resolve_latest_uses_uploaded_at_tie_breaker():
    uploaded_files = [
        _record("artifact", "/tmp/artifact-a.txt", 5, 100),
        _record("artifact", "/tmp/artifact-b.txt", 5, 200),
    ]
    latest = resolve_latest_uploaded_files(uploaded_files)
    assert latest["artifact"].file_path == "/tmp/artifact-b.txt"


def test_resolve_latest_uses_input_order_final_tie_breaker():
    uploaded_files = [
        _record("artifact", "/tmp/artifact-a.txt", 5, 200),
        _record("artifact", "/tmp/artifact-b.txt", 5, 200),
    ]
    latest = resolve_latest_uploaded_files(uploaded_files)
    assert latest["artifact"].file_path == "/tmp/artifact-b.txt"


def test_resolve_latest_rejects_unexpected_sentinel():
    uploaded_files = [
        _record(
            "artifact",
            "/tmp/artifact-a.txt",
            1,
            1,
            sentinel="STALE_SENTINEL",
        )
    ]
    with pytest.raises(RuntimeError, match="unexpected sentinel"):
        resolve_latest_uploaded_files(uploaded_files)


def test_assert_no_duplicate_uploaded_files_passes_for_unique_names():
    uploaded_files = [
        _record("wheel", "/tmp/pkg-1.whl", 1, 1),
        _record("sdist", "/tmp/pkg-1.tar.gz", 1, 1),
    ]
    assert_no_duplicate_uploaded_files(uploaded_files)


def test_assert_no_duplicate_uploaded_files_raises_for_duplicate_name():
    uploaded_files = [
        _record("wheel", "/tmp/pkg-1.whl", 1, 1),
        _record("wheel", "/tmp/pkg-2.whl", 2, 2),
    ]
    with pytest.raises(RuntimeError, match="wheel"):
        assert_no_duplicate_uploaded_files(uploaded_files)


def test_assert_selected_files_are_latest_passes_for_latest_mapping():
    uploaded_files = [
        _record("report", "/tmp/report-v1.json", 1, 10),
        _record("report", "/tmp/report-v2.json", 2, 20),
    ]
    assert_selected_files_are_latest(uploaded_files, {"report": "/tmp/report-v2.json"})


def test_assert_selected_files_are_latest_raises_for_stale_path():
    uploaded_files = [
        _record("report", "/tmp/report-v1.json", 1, 10),
        _record("report", "/tmp/report-v2.json", 2, 20),
    ]
    with pytest.raises(RuntimeError, match="report.*report-v2.json"):
        assert_selected_files_are_latest(
            uploaded_files,
            {"report": "/tmp/report-v1.json"},
        )


def test_assert_selected_files_are_latest_raises_for_missing_logical_name():
    uploaded_files = [_record("report", "/tmp/report-v1.json", 1, 10)]
    with pytest.raises(RuntimeError, match="report.*report-v1.json"):
        assert_selected_files_are_latest(uploaded_files, {})


def test_validate_uploaded_file_selection_succeeds_with_unique_records():
    uploaded_files = [
        _record("wheel", "/tmp/pkg-1.whl", 1, 1),
        _record("sdist", "/tmp/pkg-1.tar.gz", 1, 1),
    ]
    latest = validate_uploaded_file_selection(
        uploaded_files,
        {"wheel": "/tmp/pkg-1.whl", "sdist": "/tmp/pkg-1.tar.gz"},
    )
    assert latest["wheel"].file_path == "/tmp/pkg-1.whl"
    assert latest["sdist"].file_path == "/tmp/pkg-1.tar.gz"


def test_validate_uploaded_file_selection_fails_on_duplicates_when_disallowed():
    uploaded_files = [
        _record("wheel", "/tmp/pkg-1.whl", 1, 1),
        _record("wheel", "/tmp/pkg-2.whl", 2, 2),
    ]
    with pytest.raises(RuntimeError, match="wheel"):
        validate_uploaded_file_selection(
            uploaded_files,
            {"wheel": "/tmp/pkg-2.whl"},
            allow_duplicates=False,
        )


def test_validate_uploaded_file_selection_allows_duplicates_when_requested():
    uploaded_files = [
        _record("wheel", "/tmp/pkg-1.whl", 1, 1),
        _record("wheel", "/tmp/pkg-2.whl", 2, 2),
    ]
    latest = validate_uploaded_file_selection(
        uploaded_files,
        {"wheel": "/tmp/pkg-2.whl"},
        allow_duplicates=True,
    )
    assert latest["wheel"].file_path == "/tmp/pkg-2.whl"


def test_validate_uploaded_file_selection_rejects_stale_when_duplicates_allowed():
    uploaded_files = [
        _record("wheel", "/tmp/pkg-1.whl", 1, 1),
        _record("wheel", "/tmp/pkg-2.whl", 2, 2),
    ]
    with pytest.raises(RuntimeError, match="wheel.*pkg-2.whl"):
        validate_uploaded_file_selection(
            uploaded_files,
            {"wheel": "/tmp/pkg-1.whl"},
            allow_duplicates=True,
        )
