"""Regression tests for deterministic uploaded-file version validation."""

import unittest

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


class TestUploadValidation(unittest.TestCase):
    def test_resolve_latest_single_record_returns_only_file(self):
        uploaded_files = [_record("artifact", "/tmp/artifact-v1.txt", 1, 10)]
        latest = resolve_latest_uploaded_files(uploaded_files)
        self.assertEqual(latest["artifact"].file_path, "/tmp/artifact-v1.txt")

    def test_resolve_latest_selects_highest_revision(self):
        uploaded_files = [
            _record("artifact", "/tmp/artifact-v1.txt", 1, 20),
            _record("artifact", "/tmp/artifact-v2.txt", 2, 10),
        ]
        latest = resolve_latest_uploaded_files(uploaded_files)
        self.assertEqual(latest["artifact"].file_path, "/tmp/artifact-v2.txt")

    def test_resolve_latest_uses_uploaded_at_tie_breaker(self):
        uploaded_files = [
            _record("artifact", "/tmp/artifact-a.txt", 5, 100),
            _record("artifact", "/tmp/artifact-b.txt", 5, 200),
        ]
        latest = resolve_latest_uploaded_files(uploaded_files)
        self.assertEqual(latest["artifact"].file_path, "/tmp/artifact-b.txt")

    def test_resolve_latest_uses_input_order_final_tie_breaker(self):
        uploaded_files = [
            _record("artifact", "/tmp/artifact-a.txt", 5, 200),
            _record("artifact", "/tmp/artifact-b.txt", 5, 200),
        ]
        latest = resolve_latest_uploaded_files(uploaded_files)
        self.assertEqual(latest["artifact"].file_path, "/tmp/artifact-b.txt")

    def test_resolve_latest_rejects_unexpected_sentinel(self):
        uploaded_files = [
            _record(
                "artifact",
                "/tmp/artifact-a.txt",
                1,
                1,
                sentinel="STALE_SENTINEL",
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "unexpected sentinel"):
            resolve_latest_uploaded_files(uploaded_files)

    def test_assert_no_duplicate_uploaded_files_passes_for_unique_names(self):
        uploaded_files = [
            _record("wheel", "/tmp/pkg-1.whl", 1, 1),
            _record("sdist", "/tmp/pkg-1.tar.gz", 1, 1),
        ]
        assert_no_duplicate_uploaded_files(uploaded_files)

    def test_assert_no_duplicate_uploaded_files_raises_for_duplicate_name(self):
        uploaded_files = [
            _record("wheel", "/tmp/pkg-1.whl", 1, 1),
            _record("wheel", "/tmp/pkg-2.whl", 2, 2),
        ]
        with self.assertRaisesRegex(RuntimeError, "wheel"):
            assert_no_duplicate_uploaded_files(uploaded_files)

    def test_assert_selected_files_are_latest_passes_for_latest_mapping(self):
        uploaded_files = [
            _record("report", "/tmp/report-v1.json", 1, 10),
            _record("report", "/tmp/report-v2.json", 2, 20),
        ]
        assert_selected_files_are_latest(uploaded_files, {"report": "/tmp/report-v2.json"})

    def test_assert_selected_files_are_latest_raises_for_stale_path(self):
        uploaded_files = [
            _record("report", "/tmp/report-v1.json", 1, 10),
            _record("report", "/tmp/report-v2.json", 2, 20),
        ]
        with self.assertRaisesRegex(RuntimeError, "report.*report-v2.json"):
            assert_selected_files_are_latest(
                uploaded_files,
                {"report": "/tmp/report-v1.json"},
            )

    def test_assert_selected_files_are_latest_raises_for_missing_logical_name(self):
        uploaded_files = [_record("report", "/tmp/report-v1.json", 1, 10)]
        with self.assertRaisesRegex(RuntimeError, "report.*report-v1.json"):
            assert_selected_files_are_latest(uploaded_files, {})

    def test_validate_uploaded_file_selection_succeeds_with_unique_records(self):
        uploaded_files = [
            _record("wheel", "/tmp/pkg-1.whl", 1, 1),
            _record("sdist", "/tmp/pkg-1.tar.gz", 1, 1),
        ]
        latest = validate_uploaded_file_selection(
            uploaded_files,
            {"wheel": "/tmp/pkg-1.whl", "sdist": "/tmp/pkg-1.tar.gz"},
        )
        self.assertEqual(latest["wheel"].file_path, "/tmp/pkg-1.whl")
        self.assertEqual(latest["sdist"].file_path, "/tmp/pkg-1.tar.gz")

    def test_validate_uploaded_file_selection_fails_on_duplicates_when_disallowed(self):
        uploaded_files = [
            _record("wheel", "/tmp/pkg-1.whl", 1, 1),
            _record("wheel", "/tmp/pkg-2.whl", 2, 2),
        ]
        with self.assertRaisesRegex(RuntimeError, "wheel"):
            validate_uploaded_file_selection(
                uploaded_files,
                {"wheel": "/tmp/pkg-2.whl"},
                allow_duplicates=False,
            )

    def test_validate_uploaded_file_selection_allows_duplicates_when_requested(self):
        uploaded_files = [
            _record("wheel", "/tmp/pkg-1.whl", 1, 1),
            _record("wheel", "/tmp/pkg-2.whl", 2, 2),
        ]
        latest = validate_uploaded_file_selection(
            uploaded_files,
            {"wheel": "/tmp/pkg-2.whl"},
            allow_duplicates=True,
        )
        self.assertEqual(latest["wheel"].file_path, "/tmp/pkg-2.whl")

    def test_validate_uploaded_file_selection_rejects_stale_when_duplicates_allowed(self):
        uploaded_files = [
            _record("wheel", "/tmp/pkg-1.whl", 1, 1),
            _record("wheel", "/tmp/pkg-2.whl", 2, 2),
        ]
        with self.assertRaisesRegex(RuntimeError, "wheel.*pkg-2.whl"):
            validate_uploaded_file_selection(
                uploaded_files,
                {"wheel": "/tmp/pkg-1.whl"},
                allow_duplicates=True,
            )


if __name__ == "__main__":
    unittest.main()
