#!/usr/bin/env python3
"""Run frozen maintainer-private calibration validation on one Parquet file.

This script is intentionally outside the installed ``pgmuvi`` package.  It
requires ``pyarrow`` only when Parquet input is read.  The private detailed
report may contain raw ``object_id`` values; the redacted summary does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pgmuvi import __version__
from pgmuvi.instrument_channel_calibration_multisource_execution import (
    InstrumentChannelCalibrationMultiSourceChannelData,
    InstrumentChannelCalibrationMultiSourceExecutionResult,
    InstrumentChannelCalibrationMultiSourceSourceData,
    execute_instrument_channel_calibration_multisource_validation,
)
from pgmuvi.instrument_channel_calibration_multisource_validation import (
    InstrumentChannelCalibrationMultiSourceValidationProtocol,
)
from pgmuvi.instrument_channel_calibration_validation import (
    InstrumentChannelCalibrationValidationProtocol,
)

REQUIRED_COLUMNS = (
    "object_id",
    "time",
    "flux",
    "flux_error",
    "wavelength",
    "band",
)
DEFAULT_MULTISOURCE_PROTOCOL = (
    "examples/validation/"
    "kelt_r3_maintainer_multisource_validation_protocol_v1.json"
)
DEFAULT_CANDIDATE_PROTOCOL = (
    "examples/validation/kelt_r3_pairing_validation_protocol_v1.json"
)
DEFAULT_PRIVATE_REPORT = (
    "validation_outputs/"
    "private_kelt_r3_multisource_validation_report.json"
)
DEFAULT_REDACTED_SUMMARY = (
    "validation_outputs/"
    "private_kelt_r3_multisource_validation_summary.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_parquet_columns(path: Path) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        parquet = None

    if parquet is not None:
        parquet_file = parquet.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_COLUMNS) - available)
        if missing:
            raise ValueError(
                "Private Parquet input is missing required columns: "
                + ", ".join(missing)
                + "."
            )
        table = parquet_file.read(columns=list(REQUIRED_COLUMNS))
        return {
            name: table.column(name).to_pylist()
            for name in REQUIRED_COLUMNS
        }

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "The maintainer-only Parquet runner requires pyarrow, or pandas "
            "with an installed Parquet engine, in the private validation "
            "environment."
        ) from exc

    try:
        frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    except Exception as exc:
        raise RuntimeError(
            "Could not read the private Parquet input with pandas. Install "
            "pyarrow or another pandas-compatible Parquet engine."
        ) from exc
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            "Private Parquet input is missing required columns: "
            + ", ".join(missing)
            + "."
        )
    return {
        name: frame[name].tolist()
        for name in REQUIRED_COLUMNS
    }


def _normalize_object_id(value: Any) -> str:
    if value is None:
        raise ValueError("object_id values must be non-null.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("object_id values must be finite when numeric.")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("object_id values must be non-empty.")
    return normalized


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def build_source_data_from_columns(
    columns: dict[str, list[Any]],
    *,
    protocol: InstrumentChannelCalibrationMultiSourceValidationProtocol,
) -> tuple[
    tuple[InstrumentChannelCalibrationMultiSourceSourceData, ...],
    dict[str, int],
]:
    """Group one private table by object_id and split the exact channel pair."""

    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(
            "Input columns are missing: " + ", ".join(missing) + "."
        )

    lengths = {len(columns[name]) for name in REQUIRED_COLUMNS}
    if len(lengths) != 1:
        raise ValueError("All input columns must have the same row count.")

    grouped: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {
            "reference_time": [],
            "reference_flux": [],
            "reference_error": [],
            "reference_wavelength": [],
            "channel_time": [],
            "channel_flux": [],
            "channel_error": [],
            "channel_wavelength": [],
        }
    )
    ignored_non_candidate_rows = 0
    for index in range(next(iter(lengths), 0)):
        source_id = _normalize_object_id(columns["object_id"][index])
        group = grouped[source_id]
        band_value = columns["band"][index]
        if isinstance(band_value, bytes):
            band_value = band_value.decode("utf-8")
        band = "" if band_value is None else str(band_value).strip()

        if band == protocol.reference_channel:
            prefix = "reference"
        elif band == protocol.channel:
            prefix = "channel"
        else:
            ignored_non_candidate_rows += 1
            continue

        group[f"{prefix}_time"].append(
            _coerce_float(columns["time"][index])
        )
        group[f"{prefix}_flux"].append(
            _coerce_float(columns["flux"][index])
        )
        group[f"{prefix}_error"].append(
            _coerce_float(columns["flux_error"][index])
        )
        group[f"{prefix}_wavelength"].append(
            _coerce_float(columns["wavelength"][index])
        )

    sources = tuple(
        InstrumentChannelCalibrationMultiSourceSourceData(
            astrophysical_source_id=source_id,
            reference=InstrumentChannelCalibrationMultiSourceChannelData(
                channel=protocol.reference_channel,
                time=tuple(values["reference_time"]),
                flux=tuple(values["reference_flux"]),
                flux_error=tuple(values["reference_error"]),
                wavelength=tuple(values["reference_wavelength"]),
            ),
            channel=InstrumentChannelCalibrationMultiSourceChannelData(
                channel=protocol.channel,
                time=tuple(values["channel_time"]),
                flux=tuple(values["channel_flux"]),
                flux_error=tuple(values["channel_error"]),
                wavelength=tuple(values["channel_wavelength"]),
            ),
            is_primary_source=True,
            derivation_parent_source_id=None,
        )
        for source_id, values in sorted(grouped.items())
    )
    ingestion = {
        "input_row_count": next(iter(lengths), 0),
        "grouped_source_count": len(sources),
        "ignored_non_candidate_channel_row_count": (
            ignored_non_candidate_rows
        ),
    }
    return sources, ingestion


def _git_commit(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("Could not determine a full package Git commit.")
    return commit


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def execute_private_parquet_validation(
    *,
    parquet_path: Path,
    repository_root: Path,
    multisource_protocol_path: Path,
    candidate_protocol_path: Path,
    selection_seed: int,
    n_sources: int | str,
    private_report_output: Path,
    redacted_summary_output: Path,
    package_version: str | None = None,
    package_commit: str | None = None,
    executed_at_utc: str | None = None,
    parquet_loader: Callable[[Path], dict[str, list[Any]]] = (
        _load_parquet_columns
    ),
) -> InstrumentChannelCalibrationMultiSourceExecutionResult:
    """Load private Parquet rows and execute the public array-based engine."""

    parquet_path = parquet_path.expanduser().resolve()
    if not parquet_path.is_file():
        raise ValueError("parquet_path must identify an existing file.")

    protocol = InstrumentChannelCalibrationMultiSourceValidationProtocol.from_dict(
        _load_json(multisource_protocol_path)
    )
    candidate_protocol = (
        InstrumentChannelCalibrationValidationProtocol.from_dict(
            _load_json(candidate_protocol_path)
        )
    )
    columns = parquet_loader(parquet_path)
    sources, ingestion = build_source_data_from_columns(
        columns,
        protocol=protocol,
    )

    resolved_version = package_version or __version__
    resolved_commit = package_commit or _git_commit(repository_root)
    resolved_time = executed_at_utc or _utc_now()
    input_sha256 = _sha256_file(parquet_path)

    result = execute_instrument_channel_calibration_multisource_validation(
        sources,
        protocol=protocol,
        candidate_protocol=candidate_protocol,
        private_input_sha256=input_sha256,
        selection_seed=selection_seed,
        n_sources=n_sources,
        package_version=resolved_version,
        package_commit=resolved_commit,
        executed_at_utc=resolved_time,
    )

    private_payload = {
        "schema_version": (
            "pgmuvi-maintainer-private-parquet-"
            "multisource-validation-report-v1"
        ),
        "private_input_path": str(parquet_path),
        "private_input_sha256": input_sha256,
        "required_columns": list(REQUIRED_COLUMNS),
        "source_independence_assertion": (
            "Each unique object_id is treated as one independent primary "
            "astrophysical source, as asserted by the maintainer."
        ),
        "ingestion": ingestion,
        "multisource_protocol": protocol.to_dict(),
        "candidate_protocol": candidate_protocol.to_dict(),
        "execution": result.to_private_dict(),
    }
    _write_json(private_report_output, private_payload)
    _write_json(redacted_summary_output, result.summary.to_dict())
    return result


def _parse_source_count(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "all":
        return "all"
    try:
        count = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--n-sources must be an integer or 'all'."
        ) from exc
    if count < 2:
        raise argparse.ArgumentTypeError("--n-sources must be at least 2.")
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen maintainer-private multi-source calibration "
            "validation on one Parquet catalogue."
        )
    )
    parser.add_argument("parquet_path", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--multisource-protocol",
        type=Path,
        default=Path(DEFAULT_MULTISOURCE_PROTOCOL),
    )
    parser.add_argument(
        "--candidate-protocol",
        type=Path,
        default=Path(DEFAULT_CANDIDATE_PROTOCOL),
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=20260726,
    )
    parser.add_argument(
        "--n-sources",
        type=_parse_source_count,
        default=5,
    )
    parser.add_argument(
        "--private-report-output",
        type=Path,
        default=Path(DEFAULT_PRIVATE_REPORT),
    )
    parser.add_argument(
        "--redacted-summary-output",
        type=Path,
        default=Path(DEFAULT_REDACTED_SUMMARY),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.expanduser().resolve()

    def resolve_from_repository(path: Path) -> Path:
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (
            repository_root / path
        ).resolve()

    result = execute_private_parquet_validation(
        parquet_path=args.parquet_path,
        repository_root=repository_root,
        multisource_protocol_path=resolve_from_repository(
            args.multisource_protocol
        ),
        candidate_protocol_path=resolve_from_repository(
            args.candidate_protocol
        ),
        selection_seed=args.selection_seed,
        n_sources=args.n_sources,
        private_report_output=resolve_from_repository(
            args.private_report_output
        ),
        redacted_summary_output=resolve_from_repository(
            args.redacted_summary_output
        ),
    )

    print("PGMUVI maintainer-private multi-source validation")
    print(f"disposition: {result.disposition.value}")
    print(f"eligible sources: {len(result.eligible_source_ids)}")
    print(f"selected sources: {len(result.selected_source_ids)}")
    print(
        "successful holdouts: "
        f"{result.summary.successful_source_holdout_count}"
    )
    print(
        "median held-out normalized RMSE: "
        f"{result.summary.median_source_holdout_normalized_rmse}"
    )
    print(
        "worst held-out normalized RMSE: "
        f"{result.summary.worst_source_holdout_normalized_rmse}"
    )
    print(
        "median held-out normalized bias: "
        f"{result.summary.median_source_holdout_bias_normalized}"
    )
    print(f"private report: {args.private_report_output}")
    print(f"redacted summary: {args.redacted_summary_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
