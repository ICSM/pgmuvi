"""Maintained execution of frozen instrument-channel validation protocols.

This module executes a prospectively frozen protocol against repository-contained
data.  It records fold-level evidence and delegates scientific disposition to
:func:`assess_instrument_channel_calibration_validation_result`.

It does not populate a pairing-rule catalogue or activate calibration in normal
fitting workflows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    construct_instrument_channel_pairing,
    fit_instrument_channel_calibration,
)
from pgmuvi.instrument_channel_calibration_validation import (
    InstrumentChannelCalibrationValidationDataset,
    InstrumentChannelCalibrationValidationDatasetManifest,
    InstrumentChannelCalibrationValidationFoldResult,
    InstrumentChannelCalibrationValidationProtocol,
    InstrumentChannelCalibrationValidationReport,
    InstrumentChannelCalibrationValidationResult,
    InstrumentChannelCalibrationValidationSourceResult,
    assess_instrument_channel_calibration_validation_result,
)


DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_PROTOCOL_REFERENCE = (
    "examples/validation/kelt_r3_pairing_validation_protocol_v1.json"
)
DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_REFERENCE = (
    "examples/validation/kelt_r3_pairing_validation_result_v1.json"
)
DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_REPORT_REFERENCE = (
    "examples/validation/kelt_r3_pairing_validation_report_v1.json"
)
DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_ID = (
    "kelt-osn-r3-pairing-validation-execution-v1"
)

__all__ = [
    "DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_PROTOCOL_REFERENCE",
    "DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_REPORT_REFERENCE",
    "DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_ID",
    "DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_REFERENCE",
    "execute_instrument_channel_calibration_validation_dataset_manifest",
    "execute_instrument_channel_calibration_validation_protocol",
    "write_instrument_channel_calibration_validation_artifacts",
]


_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_UTC_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)


def _repository_path(
    repository_root: Path,
    reference: str,
    *,
    name: str,
) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{name} must be a non-empty repository-relative path.")

    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError(f"{name} must be repository-relative.")

    resolved = (repository_root / relative).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(
            f"{name} must not escape the repository root."
        ) from exc
    return resolved


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_protocol(
    repository_root: Path,
    protocol_reference: str,
) -> InstrumentChannelCalibrationValidationProtocol:
    path = _repository_path(
        repository_root,
        protocol_reference,
        name="protocol_reference",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return InstrumentChannelCalibrationValidationProtocol.from_dict(payload)


def _load_dataset_manifest(
    repository_root: Path,
    manifest_reference: str,
) -> InstrumentChannelCalibrationValidationDatasetManifest:
    path = _repository_path(
        repository_root,
        manifest_reference,
        name="dataset_manifest_reference",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return InstrumentChannelCalibrationValidationDatasetManifest.from_dict(
        payload
    )


def _load_dataset_channel_rows(
    repository_root: Path,
    protocol: InstrumentChannelCalibrationValidationProtocol,
    dataset: InstrumentChannelCalibrationValidationDataset,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = _repository_path(
        repository_root,
        dataset.dataset_reference,
        name="dataset.dataset_reference",
    )

    actual_sha256 = _file_sha256(dataset_path)
    if actual_sha256 != dataset.dataset_sha256:
        raise ValueError(
            "Validation dataset SHA-256 does not match its manifest identity."
        )

    reference_rows: list[tuple[float, float, float]] = []
    channel_rows: list[tuple[float, float, float]] = []

    with dataset_path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        required = {
            "time",
            "flux",
            "flux_error",
            "wavelength",
            "band",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                "Validation dataset must contain time, flux, flux_error, "
                "wavelength, and band columns."
            )

        for row in reader:
            try:
                time_value = float(row["time"])
                flux_value = float(row["flux"])
                error_value = float(row["flux_error"])
                wavelength = float(row["wavelength"])
            except (TypeError, ValueError):
                continue

            if (
                not math.isfinite(time_value)
                or not math.isfinite(flux_value)
                or not math.isfinite(error_value)
                or not math.isfinite(wavelength)
                or flux_value <= 0.0
                or error_value <= 0.0
                or not math.isclose(
                    wavelength,
                    protocol.physical_wavelength,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
            ):
                continue

            record = (time_value, flux_value, error_value)
            if row["band"] == protocol.reference_channel:
                reference_rows.append(record)
            elif row["band"] == protocol.channel:
                channel_rows.append(record)

    if not reference_rows:
        raise ValueError(
            "Validation dataset contains no usable reference-channel rows."
        )
    if not channel_rows:
        raise ValueError(
            "Validation dataset contains no usable target-channel rows."
        )

    return (
        np.asarray(reference_rows, dtype=float),
        np.asarray(channel_rows, dtype=float),
    )


def _load_anchor_channel_rows(
    repository_root: Path,
    protocol: InstrumentChannelCalibrationValidationProtocol,
) -> tuple[np.ndarray, np.ndarray]:
    return _load_dataset_channel_rows(
        repository_root,
        protocol,
        protocol.anchor_dataset,
    )


def _construct_partitioned_pair_positions(
    reference_times: np.ndarray,
    channel_times: np.ndarray,
    *,
    protocol: InstrumentChannelCalibrationValidationProtocol,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact maintained pairs while avoiding one unnecessary huge grid.

    Combined timestamps are partitioned whenever the adjacent gap exceeds the
    pairing tolerance.  No eligible cross-partition pair can then exist, so
    invoking the maintained pairing primitive independently in each partition
    preserves its maximum-cardinality/minimum-separation result.
    """

    tolerance = float(protocol.maximum_time_separation or 0.0)
    combined = sorted(
        (
            (float(time), 0, index)
            for index, time in enumerate(reference_times)
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    combined.extend(
        (float(time), 1, index)
        for index, time in enumerate(channel_times)
    )
    combined.sort(key=lambda item: (item[0], item[1], item[2]))

    components: list[list[tuple[float, int, int]]] = []
    current: list[tuple[float, int, int]] = []

    for record in combined:
        if current and record[0] - current[-1][0] > tolerance:
            components.append(current)
            current = []
        current.append(record)

    if current:
        components.append(current)

    reference_positions: list[int] = []
    channel_positions: list[int] = []

    for component in components:
        local_reference = np.asarray(
            [item[2] for item in component if item[1] == 0],
            dtype=int,
        )
        local_channel = np.asarray(
            [item[2] for item in component if item[1] == 1],
            dtype=int,
        )

        if local_reference.size == 0 or local_channel.size == 0:
            continue

        pairing = construct_instrument_channel_pairing(
            reference_times[local_reference],
            channel_times[local_channel],
            reference_channel=protocol.reference_channel,
            channel=protocol.channel,
            wavelength=protocol.physical_wavelength,
            time_unit=protocol.time_unit,
            method=protocol.pairing_method,
            maximum_time_separation=protocol.maximum_time_separation,
            reference_row_indices=local_reference,
            channel_row_indices=local_channel,
        )
        reference_positions.extend(pairing.reference_row_indices)
        channel_positions.extend(pairing.channel_row_indices)

    if not reference_positions:
        raise ValueError(
            "No eligible one-to-one pairs were found for the frozen protocol."
        )

    reference_array = np.asarray(reference_positions, dtype=int)
    channel_array = np.asarray(channel_positions, dtype=int)
    midpoint = 0.5 * (
        reference_times[reference_array]
        + channel_times[channel_array]
    )
    ordering = np.lexsort(
        (channel_array, reference_array, midpoint)
    )
    return reference_array[ordering], channel_array[ordering]


def _execute_fold(
    *,
    fold_index: int,
    holdout_positions: np.ndarray,
    reference_flux: np.ndarray,
    channel_flux: np.ndarray,
    reference_error: np.ndarray,
    channel_error: np.ndarray,
    time_separation: np.ndarray,
    protocol: InstrumentChannelCalibrationValidationProtocol,
) -> InstrumentChannelCalibrationValidationFoldResult:
    n_pairs = int(reference_flux.size)
    training_mask = np.ones(n_pairs, dtype=bool)
    training_mask[holdout_positions] = False
    training_positions = np.flatnonzero(training_mask)

    amplitude = float(
        np.quantile(
            reference_flux[holdout_positions],
            0.95,
            method="linear",
        )
        - np.quantile(
            reference_flux[holdout_positions],
            0.05,
            method="linear",
        )
    )
    median_reference_error = float(
        np.median(reference_error[holdout_positions])
    )

    try:
        calibration = fit_instrument_channel_calibration(
            reference_flux[training_positions],
            channel_flux[training_positions],
            reference_channel=protocol.reference_channel,
            channel=protocol.channel,
            wavelength=protocol.physical_wavelength,
            reference_error=reference_error[training_positions],
            channel_error=channel_error[training_positions],
            sigma_clip=protocol.sigma_clip,
            max_iter=protocol.maximum_fit_iterations,
            min_pairs=protocol.minimum_fit_pairs,
        )

        prediction = (
            calibration.offset
            + calibration.scale * channel_flux[holdout_positions]
        )
        residual = reference_flux[holdout_positions] - prediction

        return InstrumentChannelCalibrationValidationFoldResult(
            fold_index=fold_index,
            n_training_pairs=int(training_positions.size),
            n_holdout_pairs=int(holdout_positions.size),
            successful=True,
            holdout_reference_flux_q05_q95_amplitude=amplitude,
            holdout_median_reference_flux_error=median_reference_error,
            holdout_normalized_rmse=float(
                np.sqrt(np.mean(residual**2)) / amplitude
            ),
            holdout_median_bias_normalized=float(
                np.median(residual) / amplitude
            ),
            maximum_absolute_time_separation=float(
                np.max(time_separation[holdout_positions])
            ),
            fitted_offset=float(calibration.offset),
            fitted_scale=float(calibration.scale),
        )
    except Exception as exc:
        return InstrumentChannelCalibrationValidationFoldResult(
            fold_index=fold_index,
            n_training_pairs=int(training_positions.size),
            n_holdout_pairs=int(holdout_positions.size),
            successful=False,
            holdout_reference_flux_q05_q95_amplitude=None,
            holdout_median_reference_flux_error=None,
            holdout_normalized_rmse=None,
            holdout_median_bias_normalized=None,
            maximum_absolute_time_separation=None,
            fitted_offset=None,
            fitted_scale=None,
            failure_reasons=(f"{type(exc).__name__}: {exc}",),
        )


def _validate_execution_request(
    *,
    repository_root: str | Path,
    package_version: str,
    package_commit: str,
    executed_at_utc: str,
) -> tuple[Path, str]:
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise ValueError("repository_root must identify an existing directory.")
    if not isinstance(package_version, str) or not package_version.strip():
        raise ValueError("package_version must be a non-empty string.")
    if not isinstance(package_commit, str) or not _COMMIT_PATTERN.fullmatch(
        package_commit
    ):
        raise ValueError(
            "package_commit must be a lowercase 40-character Git commit."
        )
    if not isinstance(executed_at_utc, str) or not _UTC_PATTERN.fullmatch(
        executed_at_utc
    ):
        raise ValueError(
            "executed_at_utc must use YYYY-MM-DDTHH:MM:SSZ."
        )
    return root, package_version.strip()


def _execute_validation_dataset(
    *,
    repository_root: Path,
    protocol: InstrumentChannelCalibrationValidationProtocol,
    dataset: InstrumentChannelCalibrationValidationDataset,
) -> InstrumentChannelCalibrationValidationSourceResult:
    reference_rows, channel_rows = _load_dataset_channel_rows(
        repository_root,
        protocol,
        dataset,
    )

    reference_positions, channel_positions = (
        _construct_partitioned_pair_positions(
            reference_rows[:, 0],
            channel_rows[:, 0],
            protocol=protocol,
        )
    )

    reference_time = reference_rows[reference_positions, 0]
    reference_flux = reference_rows[reference_positions, 1]
    reference_error = reference_rows[reference_positions, 2]
    channel_time = channel_rows[channel_positions, 0]
    channel_flux = channel_rows[channel_positions, 1]
    channel_error = channel_rows[channel_positions, 2]
    time_separation = np.abs(reference_time - channel_time)

    n_pairs = int(reference_flux.size)
    fold_positions = tuple(
        np.asarray(item, dtype=int)
        for item in np.array_split(
            np.arange(n_pairs, dtype=int),
            protocol.acceptance_criteria.temporal_fold_count,
        )
    )

    fold_results = tuple(
        _execute_fold(
            fold_index=fold_index,
            holdout_positions=holdout,
            reference_flux=reference_flux,
            channel_flux=channel_flux,
            reference_error=reference_error,
            channel_error=channel_error,
            time_separation=time_separation,
            protocol=protocol,
        )
        for fold_index, holdout in enumerate(fold_positions)
    )

    source_failure_reasons: tuple[str, ...] = ()
    if any(not fold.successful for fold in fold_results):
        source_failure_reasons = (
            "one_or_more_temporal_folds_unsuccessful",
        )

    return InstrumentChannelCalibrationValidationSourceResult(
        dataset=dataset,
        n_matched_pairs=n_pairs,
        fold_results=fold_results,
        failure_reasons=source_failure_reasons,
    )


def _build_validation_result(
    *,
    protocol: InstrumentChannelCalibrationValidationProtocol,
    source_results: tuple[
        InstrumentChannelCalibrationValidationSourceResult,
        ...,
    ],
    result_id: str,
    result_version: str,
    execution_reference: str,
    package_version: str,
    package_commit: str,
    executed_at_utc: str,
) -> tuple[
    InstrumentChannelCalibrationValidationResult,
    InstrumentChannelCalibrationValidationReport,
]:
    result = InstrumentChannelCalibrationValidationResult(
        result_id=result_id,
        result_version=result_version,
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_sha256=protocol.canonical_sha256,
        rule_id=protocol.rule_id,
        execution_reference=execution_reference,
        package_version=package_version,
        package_commit=package_commit,
        executed_at_utc=executed_at_utc,
        source_results=source_results,
        execution_completed=True,
    )

    result_payload = result.to_dict()
    InstrumentChannelCalibrationValidationResult.from_dict(result_payload)
    report = assess_instrument_channel_calibration_validation_result(
        protocol,
        result,
    )
    return result, report


def execute_instrument_channel_calibration_validation_protocol(
    *,
    repository_root: str | Path,
    protocol_reference: str = (
        DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_PROTOCOL_REFERENCE
    ),
    result_id: str = (
        DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_ID
    ),
    result_version: str = "1.0",
    execution_reference: str = (
        DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_REFERENCE
    ),
    package_version: str,
    package_commit: str,
    executed_at_utc: str,
) -> tuple[
    InstrumentChannelCalibrationValidationResult,
    InstrumentChannelCalibrationValidationReport,
]:
    """Execute one frozen protocol against its repository-contained anchor."""

    root, normalized_version = _validate_execution_request(
        repository_root=repository_root,
        package_version=package_version,
        package_commit=package_commit,
        executed_at_utc=executed_at_utc,
    )
    protocol = _load_protocol(root, protocol_reference)
    source_result = _execute_validation_dataset(
        repository_root=root,
        protocol=protocol,
        dataset=protocol.anchor_dataset,
    )
    return _build_validation_result(
        protocol=protocol,
        source_results=(source_result,),
        result_id=result_id,
        result_version=result_version,
        execution_reference=execution_reference,
        package_version=normalized_version,
        package_commit=package_commit,
        executed_at_utc=executed_at_utc,
    )


def execute_instrument_channel_calibration_validation_dataset_manifest(
    *,
    repository_root: str | Path,
    dataset_manifest_reference: str,
    protocol_reference: str = (
        DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_PROTOCOL_REFERENCE
    ),
    result_id: str = (
        DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_ID
    ),
    result_version: str = "1.0",
    execution_reference: str = (
        DEFAULT_INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_REFERENCE
    ),
    package_version: str,
    package_commit: str,
    executed_at_utc: str,
) -> tuple[
    InstrumentChannelCalibrationValidationResult,
    InstrumentChannelCalibrationValidationReport,
]:
    """Execute the anchor plus strict additional independent datasets."""

    root, normalized_version = _validate_execution_request(
        repository_root=repository_root,
        package_version=package_version,
        package_commit=package_commit,
        executed_at_utc=executed_at_utc,
    )
    protocol = _load_protocol(root, protocol_reference)
    manifest = _load_dataset_manifest(
        root,
        dataset_manifest_reference,
    )

    if (
        manifest.protocol_id != protocol.protocol_id
        or manifest.protocol_version != protocol.protocol_version
    ):
        raise ValueError(
            "Dataset manifest protocol identity does not match the "
            "frozen protocol."
        )
    if manifest.protocol_sha256 != protocol.canonical_sha256:
        raise ValueError(
            "Dataset manifest protocol digest does not match the "
            "frozen protocol."
        )

    anchor = protocol.anchor_dataset
    for dataset in manifest.datasets:
        if dataset.dataset_id == anchor.dataset_id:
            raise ValueError(
                "Additional dataset_id must differ from the anchor dataset."
            )
        if (
            dataset.astrophysical_source_id
            == anchor.astrophysical_source_id
        ):
            raise ValueError(
                "Additional datasets must represent astrophysical sources "
                "independent of the anchor source."
            )

    source_results = (
        _execute_validation_dataset(
            repository_root=root,
            protocol=protocol,
            dataset=anchor,
        ),
        *(
            _execute_validation_dataset(
                repository_root=root,
                protocol=protocol,
                dataset=dataset,
            )
            for dataset in manifest.datasets
        ),
    )

    return _build_validation_result(
        protocol=protocol,
        source_results=source_results,
        result_id=result_id,
        result_version=result_version,
        execution_reference=execution_reference,
        package_version=normalized_version,
        package_commit=package_commit,
        executed_at_utc=executed_at_utc,
    )


def write_instrument_channel_calibration_validation_artifacts(
    *,
    result: InstrumentChannelCalibrationValidationResult,
    report: InstrumentChannelCalibrationValidationReport,
    result_path: str | Path,
    report_path: str | Path,
) -> None:
    """Write strict result and assessment JSON artifacts."""

    if result.result_id != report.result_id:
        raise ValueError("Result and report identifiers must match.")

    for path, payload in (
        (Path(result_path), result.to_dict()),
        (Path(report_path), report.to_dict()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
