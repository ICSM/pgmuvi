"""Integration tests for .github/scripts/validate_uploaded_artifacts.py."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from pgmuvi.upload_validation import UPLOAD_VALIDATION_SENTINEL

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / ".github" / "scripts" / "validate_uploaded_artifacts.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "validate_uploaded_artifacts_script",
        _SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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
    monkeypatch,
    *,
    dist_dir: Path,
    expected_version: str,
    sentinel: str,
):
    monkeypatch.setattr(
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
    )
    return module.main()


def _valid_expected_sentinel():
    return f"sha:run:attempt:{UPLOAD_VALIDATION_SENTINEL}"


def test_validate_uploaded_artifacts_valid_wheel_and_sdist_pass(tmp_path, monkeypatch):
    module = _load_script_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=200)
    _write_artifact(dist_dir / "pgmuvi-1.2.3-src.tar.gz", mtime_ns=210)
    sentinel = _valid_expected_sentinel()
    _write_sentinel(dist_dir, sentinel)

    assert (
        _run_main(
            module,
            monkeypatch,
            dist_dir=dist_dir,
            expected_version="1.2.3",
            sentinel=sentinel,
        )
        == 0
    )


def test_validate_uploaded_artifacts_stale_older_wheel_exists(tmp_path, monkeypatch):
    module = _load_script_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_artifact(dist_dir / "pgmuvi-0.0.1-py3-none-any.whl", mtime_ns=100)
    _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=300)
    _write_artifact(dist_dir / "pgmuvi-1.2.3-src.tar.gz", mtime_ns=250)
    sentinel = _valid_expected_sentinel()
    _write_sentinel(dist_dir, sentinel)

    assert (
        _run_main(
            module,
            monkeypatch,
            dist_dir=dist_dir,
            expected_version="1.2.3",
            sentinel=sentinel,
        )
        == 0
    )


def test_validate_uploaded_artifacts_stale_older_sdist_exists(tmp_path, monkeypatch):
    module = _load_script_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=250)
    _write_artifact(dist_dir / "pgmuvi-0.0.1-src.tar.gz", mtime_ns=100)
    _write_artifact(dist_dir / "pgmuvi-1.2.3-src.tar.gz", mtime_ns=300)
    sentinel = _valid_expected_sentinel()
    _write_sentinel(dist_dir, sentinel)

    assert (
        _run_main(
            module,
            monkeypatch,
            dist_dir=dist_dir,
            expected_version="1.2.3",
            sentinel=sentinel,
        )
        == 0
    )


def test_validate_uploaded_artifacts_wrong_sentinel_fails(tmp_path, monkeypatch):
    module = _load_script_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=200)
    _write_artifact(dist_dir / "pgmuvi-1.2.3-src.tar.gz", mtime_ns=210)
    _write_sentinel(dist_dir, "sha:run:attempt:WRONG_SENTINEL_TOKEN")

    with pytest.raises(RuntimeError, match="Sentinel version token mismatch"):
        _run_main(
            module,
            monkeypatch,
            dist_dir=dist_dir,
            expected_version="1.2.3",
            sentinel="sha:run:attempt:WRONG_SENTINEL_TOKEN",
        )


def test_validate_uploaded_artifacts_version_mismatch_fails(tmp_path, monkeypatch):
    module = _load_script_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=200)
    _write_artifact(dist_dir / "pgmuvi-9.9.9-src.tar.gz", mtime_ns=210)
    sentinel = _valid_expected_sentinel()
    _write_sentinel(dist_dir, sentinel)

    with pytest.raises(RuntimeError, match="Artifact version mismatch"):
        _run_main(
            module,
            monkeypatch,
            dist_dir=dist_dir,
            expected_version="1.2.3",
            sentinel=sentinel,
        )


@pytest.mark.parametrize(
    "missing_kind, wheel_name, sdist_name",
    [
        ("wheel", None, "pgmuvi-1.2.3.tar.gz"),
        ("wheel", None, "pgmuvi-1.2.3-src.tar.gz"),
        ("sdist", "pgmuvi-1.2.3-py3-none-any.whl", None),
    ],
)
def test_validate_uploaded_artifacts_missing_required_artifact_fails(
    tmp_path,
    monkeypatch,
    missing_kind,
    wheel_name,
    sdist_name,
):
    module = _load_script_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    if wheel_name is not None:
        _write_artifact(dist_dir / wheel_name, mtime_ns=200)
    if sdist_name is not None:
        _write_artifact(dist_dir / sdist_name, mtime_ns=210)
    sentinel = _valid_expected_sentinel()
    _write_sentinel(dist_dir, sentinel)

    with pytest.raises(RuntimeError, match="Expected both wheel and sdist artifacts"):
        _run_main(
            module,
            monkeypatch,
            dist_dir=dist_dir,
            expected_version="1.2.3",
            sentinel=sentinel,
        )


def test_validate_uploaded_artifacts_allows_duplicate_logical_names_in_workflow(
    tmp_path,
    monkeypatch,
):
    module = _load_script_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_artifact(dist_dir / "pgmuvi-0.0.1-py3-none-any.whl", mtime_ns=100)
    _write_artifact(dist_dir / "pgmuvi-1.2.3-py3-none-any.whl", mtime_ns=300)
    _write_artifact(dist_dir / "pgmuvi-0.0.1-src.tar.gz", mtime_ns=120)
    _write_artifact(dist_dir / "pgmuvi-1.2.3-src.tar.gz", mtime_ns=320)
    sentinel = _valid_expected_sentinel()
    _write_sentinel(dist_dir, sentinel)

    call_args = {}
    original_validate = module.validate_uploaded_file_selection

    def _capturing_validate(uploaded_files, selected_files, *, allow_duplicates=False):
        call_args["allow_duplicates"] = allow_duplicates
        return original_validate(
            uploaded_files,
            selected_files,
            allow_duplicates=allow_duplicates,
        )

    monkeypatch.setattr(module, "validate_uploaded_file_selection", _capturing_validate)
    assert (
        _run_main(
            module,
            monkeypatch,
            dist_dir=dist_dir,
            expected_version="1.2.3",
            sentinel=sentinel,
        )
        == 0
    )
    assert call_args["allow_duplicates"] is True
