#!/usr/bin/env python3
"""Execute the frozen representative instrument-channel validation protocol."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pgmuvi.instrument_channel_calibration_validation_execution import (
    DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_PROTOCOL_REFERENCE,
    DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_REPORT_REFERENCE,
    DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_REFERENCE,
    execute_instrument_channel_calibration_validation_protocol,
    write_instrument_channel_calibration_validation_artifacts,
)


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _package_version() -> str:
    try:
        return importlib.metadata.version("pgmuvi")
    except importlib.metadata.PackageNotFoundError:
        return "repository-checkout"


def main() -> int:
    parser = argparse.ArgumentParser()
    repository_root = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_root,
    )
    parser.add_argument(
        "--protocol",
        default=(
            DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_PROTOCOL_REFERENCE
        ),
    )
    parser.add_argument(
        "--result-output",
        default=(
            DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_REFERENCE
        ),
    )
    parser.add_argument(
        "--report-output",
        default=(
            DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_REPORT_REFERENCE
        ),
    )
    parser.add_argument("--package-version", default=None)
    parser.add_argument("--executed-at-utc", default=None)
    arguments = parser.parse_args()

    root = arguments.repository_root.resolve()
    tracked_changes = _git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_changes:
        parser.error(
            "Tracked repository state must be clean before maintained execution."
        )

    package_commit = _git_output(root, "rev-parse", "HEAD")
    package_version = arguments.package_version or _package_version()
    executed_at = arguments.executed_at_utc or (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    result, report = (
        execute_instrument_channel_calibration_validation_protocol(
            repository_root=root,
            protocol_reference=arguments.protocol,
            execution_reference=arguments.result_output,
            package_version=package_version,
            package_commit=package_commit,
            executed_at_utc=executed_at,
        )
    )

    result_path = (root / arguments.result_output).resolve()
    report_path = (root / arguments.report_output).resolve()
    write_instrument_channel_calibration_validation_artifacts(
        result=result,
        report=report,
        result_path=result_path,
        report_path=report_path,
    )

    source = result.source_results[0]
    print(
        f"result={result_path.relative_to(root)} "
        f"report={report_path.relative_to(root)}"
    )
    print(
        f"pairs={source.n_matched_pairs} "
        f"folds={source.n_temporal_folds} "
        f"successful_folds={source.n_successful_temporal_folds}"
    )
    print(
        f"disposition={report.disposition.value} "
        f"independent_sources="
        f"{report.independent_astrophysical_source_count} "
        f"reasons={','.join(report.reasons)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
