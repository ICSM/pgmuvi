"""Integration tests for .github/scripts/validate_uploaded_artifacts.py."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pgmuvi.upload_validation import UPLOAD_VALIDATION_SENTINEL

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / ".github" / "scripts" / "validate_uploaded_artifacts.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "validate_uploaded_artifacts_script",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load validate_uploaded_artifacts script module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_artifact(path: Path, *, mtime_ns: int):
    path.write_text("artifact", encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _write_sentinel(dist_dir: Path, value: str):
    sentinel_path = dist_dir / ".upload_validation_sentinel"
    sentinel_path.write_text(value, encoding="utf-8")


def _run_main(
    module,
    *,
    dist_dir: Path,
    expected_version: str,
    sentinel: str,
):
    with mock.patch.object(
        sys,
        "argv",
        [
            str(_SCRIPT_PATH),
            "--dist-dir",
            str(dist_dir),
            "--expected-version",
            expected_version,
            "--expected-sentinel",
            sentinel,
        ],
    ):
        return module.main()


def _valid_expected_sentinel():
    return f"sha:run:attempt:{UPLOAD_VALIDATION_SENTINEL}"


class TestValidateUploadedArtifactsScript(unittest.TestCase):
    def test_validate_uploaded_artifacts_valid_wheel_and_sdist_pass(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir) / "dist"
            dist_dir.mkdir()
            _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=200)
            _write_artifact(dist_dir / "pgmuvi-1.2.3-src.tar.gz", mtime_ns=210)
            sentinel = _valid_expected_sentinel()
            _write_sentinel(dist_dir, sentinel)

            self.assertEqual(
                _run_main(
                    module,
                    dist_dir=dist_dir,
                    expected_version="1.2.3",
                    sentinel=sentinel,
                ),
                0,
            )

    def test_validate_uploaded_artifacts_stale_older_wheel_exists(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir) / "dist"
            dist_dir.mkdir()
            _write_artifact(dist_dir / "pgmuvi-0.0.1-py3-none-any.whl", mtime_ns=100)
            _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=300)
            _write_artifact(dist_dir / "pgmuvi-1.2.3-src.tar.gz", mtime_ns=250)
            sentinel = _valid_expected_sentinel()
            _write_sentinel(dist_dir, sentinel)

            self.assertEqual(
                _run_main(
                    module,
                    dist_dir=dist_dir,
                    expected_version="1.2.3",
                    sentinel=sentinel,
                ),
                0,
            )

    def test_validate_uploaded_artifacts_stale_older_sdist_exists(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir) / "dist"
            dist_dir.mkdir()
            _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=250)
            _write_artifact(dist_dir / "pgmuvi-0.0.1-src.tar.gz", mtime_ns=100)
            _write_artifact(dist_dir / "pgmuvi-1.2.3-src.tar.gz", mtime_ns=300)
            sentinel = _valid_expected_sentinel()
            _write_sentinel(dist_dir, sentinel)

            self.assertEqual(
                _run_main(
                    module,
                    dist_dir=dist_dir,
                    expected_version="1.2.3",
                    sentinel=sentinel,
                ),
                0,
            )

    def test_validate_uploaded_artifacts_wrong_sentinel_fails(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir) / "dist"
            dist_dir.mkdir()
            _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=200)
            _write_artifact(dist_dir / "pgmuvi-1.2.3-src.tar.gz", mtime_ns=210)
            _write_sentinel(dist_dir, "sha:run:attempt:WRONG_SENTINEL_TOKEN")

            with self.assertRaisesRegex(RuntimeError, "Sentinel version token mismatch"):
                _run_main(
                    module,
                    dist_dir=dist_dir,
                    expected_version="1.2.3",
                    sentinel="sha:run:attempt:WRONG_SENTINEL_TOKEN",
                )

    def test_validate_uploaded_artifacts_version_mismatch_fails(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir) / "dist"
            dist_dir.mkdir()
            _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=200)
            _write_artifact(dist_dir / "pgmuvi-9.9.9-src.tar.gz", mtime_ns=210)
            sentinel = _valid_expected_sentinel()
            _write_sentinel(dist_dir, sentinel)

            with self.assertRaisesRegex(RuntimeError, "Artifact version mismatch"):
                _run_main(
                    module,
                    dist_dir=dist_dir,
                    expected_version="1.2.3",
                    sentinel=sentinel,
                )

    def test_validate_uploaded_artifacts_missing_required_artifact_fails(self):
        module = _load_script_module()
        cases = [
            ("wheel", None, "pgmuvi-1.2.3.tar.gz"),
            ("wheel", None, "pgmuvi-1.2.3-src.tar.gz"),
            ("sdist", "pgmuvi-1.2.3-py3-none-any.whl", None),
        ]
        for missing_kind, wheel_name, sdist_name in cases:
            with self.subTest(missing_kind=missing_kind):
                with tempfile.TemporaryDirectory() as tmpdir:
                    dist_dir = Path(tmpdir) / "dist"
                    dist_dir.mkdir()
                    if wheel_name is not None:
                        _write_artifact(dist_dir / wheel_name, mtime_ns=200)
                    if sdist_name is not None:
                        _write_artifact(dist_dir / sdist_name, mtime_ns=210)
                    sentinel = _valid_expected_sentinel()
                    _write_sentinel(dist_dir, sentinel)

                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Expected both wheel and sdist artifacts",
                    ):
                        _run_main(
                            module,
                            dist_dir=dist_dir,
                            expected_version="1.2.3",
                            sentinel=sentinel,
                        )

    def test_validate_uploaded_artifacts_allows_duplicate_names_in_workflow(self):
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir) / "dist"
            dist_dir.mkdir()
            _write_artifact(dist_dir / "pgmuvi-0.0.1-py3-none-any.whl", mtime_ns=100)
            _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=300)
            _write_artifact(dist_dir / "pgmuvi-0.0.1-src.tar.gz", mtime_ns=120)
            _write_artifact(dist_dir / "pgmuvi-1.2.3-src.tar.gz", mtime_ns=320)
            sentinel = _valid_expected_sentinel()
            _write_sentinel(dist_dir, sentinel)

            call_args = {}
            original_validate = module.validate_uploaded_file_selection

            def _capturing_validate(
                uploaded_files,
                selected_files,
                *,
                allow_duplicates=False,
            ):
                call_args["allow_duplicates"] = allow_duplicates
                return original_validate(
                    uploaded_files,
                    selected_files,
                    allow_duplicates=allow_duplicates,
                )

            with mock.patch.object(
                module,
                "validate_uploaded_file_selection",
                _capturing_validate,
            ):
                self.assertEqual(
                    _run_main(
                        module,
                        dist_dir=dist_dir,
                        expected_version="1.2.3",
                        sentinel=sentinel,
                    ),
                    0,
                )
            self.assertTrue(call_args["allow_duplicates"])


if __name__ == "__main__":
    unittest.main()
