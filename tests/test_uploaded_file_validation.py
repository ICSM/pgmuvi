"""Tests for deterministic uploaded-file version selection safeguards."""

import unittest

from pgmuvi.upload_validation import (
    UPLOAD_VALIDATION_SENTINEL,
    assert_no_duplicate_uploaded_files,
    assert_selected_files_are_latest,
    resolve_latest_uploaded_files,
)


class TestUploadedFileValidation(unittest.TestCase):
    """Regression tests for stale-upload selection and duplicate protection."""

    def test_duplicate_uploaded_files_raise(self):
        records = [
            {
                "logical_name": "paper.pdf",
                "file_path": "/tmp/paper-v1.pdf",
                "revision": 1,
                "uploaded_at": 10,
                "sentinel": UPLOAD_VALIDATION_SENTINEL,
            },
            {
                "logical_name": "paper.pdf",
                "file_path": "/tmp/paper-v2.pdf",
                "revision": 2,
                "uploaded_at": 20,
                "sentinel": UPLOAD_VALIDATION_SENTINEL,
            },
        ]
        with self.assertRaises(RuntimeError):
            assert_no_duplicate_uploaded_files(records)

    def test_stale_selected_file_is_rejected(self):
        records = [
            {
                "logical_name": "report.json",
                "file_path": "/tmp/report-old.json",
                "revision": 1,
                "uploaded_at": 100,
                "sentinel": UPLOAD_VALIDATION_SENTINEL,
            },
            {
                "logical_name": "report.json",
                "file_path": "/tmp/report-new.json",
                "revision": 2,
                "uploaded_at": 200,
                "sentinel": UPLOAD_VALIDATION_SENTINEL,
            },
        ]
        with self.assertRaises(RuntimeError):
            assert_selected_files_are_latest(
                records,
                {"report.json": "/tmp/report-old.json"},
            )

    def test_latest_selection_is_deterministic(self):
        records = [
            {
                "logical_name": "artifact.txt",
                "file_path": "/tmp/artifact-v1.txt",
                "revision": 1,
                "uploaded_at": 100,
                "sentinel": UPLOAD_VALIDATION_SENTINEL,
            },
            {
                "logical_name": "artifact.txt",
                "file_path": "/tmp/artifact-v2.txt",
                "revision": 2,
                "uploaded_at": 150,
                "sentinel": UPLOAD_VALIDATION_SENTINEL,
            },
            {
                "logical_name": "artifact.txt",
                "file_path": "/tmp/artifact-v2-b.txt",
                "revision": 2,
                "uploaded_at": 200,
                "sentinel": UPLOAD_VALIDATION_SENTINEL,
            },
        ]
        latest = resolve_latest_uploaded_files(records)
        self.assertEqual(latest["artifact.txt"].file_path, "/tmp/artifact-v2-b.txt")

    def test_single_file_is_accepted(self):
        records = [
            {
                "logical_name": "single.txt",
                "file_path": "/tmp/single.txt",
                "revision": 5,
                "uploaded_at": 500,
                "sentinel": UPLOAD_VALIDATION_SENTINEL,
            }
        ]
        latest = resolve_latest_uploaded_files(records)
        self.assertEqual(latest["single.txt"].file_path, "/tmp/single.txt")
        assert_selected_files_are_latest(records, {"single.txt": "/tmp/single.txt"})

    def test_unexpected_sentinel_is_rejected(self):
        records = [
            {
                "logical_name": "bad.txt",
                "file_path": "/tmp/bad.txt",
                "revision": 1,
                "uploaded_at": 1,
                "sentinel": "STALE_UPLOAD_VERSION",
            }
        ]
        with self.assertRaises(RuntimeError):
            resolve_latest_uploaded_files(records)


if __name__ == "__main__":
    unittest.main()
