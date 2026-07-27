"""Execute source-balanced multi-source calibration validation.

The public engine in this module consumes already prepared per-source arrays.
It performs no file discovery and has no Parquet dependency.  A maintainer-only
runner may load private data and pass those arrays here.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    construct_instrument_channel_pairing,
    fit_instrument_channel_calibration,
)
from pgmuvi.instrument_channel_calibration_multisource_validation import (
    InstrumentChannelCalibrationMultiSourceValidationDisposition,
    InstrumentChannelCalibrationMultiSourceValidationProtocol,
    InstrumentChannelCalibrationMultiSourceValidationSummary,
)
from pgmuvi.instrument_channel_calibration_validation import (
    InstrumentChannelCalibrationValidationProtocol,
)

__all__ = [
    "InstrumentChannelCalibrationMultiSourceChannelData",
    "InstrumentChannelCalibrationMultiSourceEligibilityResult",
    "InstrumentChannelCalibrationMultiSourceExecutionResult",
    "InstrumentChannelCalibrationMultiSourceFoldResult",
    "InstrumentChannelCalibrationMultiSourceHoldoutResult",
    "InstrumentChannelCalibrationMultiSourceSourceData",
    "InstrumentChannelCalibrationMultiSourceTrainingSample",
    "execute_instrument_channel_calibration_multisource_validation",
]


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _float_tuple(value: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional numeric sequence.")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a one-dimensional numeric sequence."
        ) from exc


def _source_hash(source_id: str) -> str:
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()


def _mad(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


@dataclass(frozen=True)
class InstrumentChannelCalibrationMultiSourceChannelData:
    """Rows for one observational channel of one astrophysical source."""

    channel: str
    time: tuple[float, ...]
    flux: tuple[float, ...]
    flux_error: tuple[float, ...]
    wavelength: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", _text(self.channel, name="channel"))
        for name in ("time", "flux", "flux_error", "wavelength"):
            object.__setattr__(
                self,
                name,
                _float_tuple(getattr(self, name), name=name),
            )
        lengths = {
            len(self.time),
            len(self.flux),
            len(self.flux_error),
            len(self.wavelength),
        }
        if len(lengths) != 1:
            raise ValueError("Channel data arrays must have equal length.")


@dataclass(frozen=True)
class InstrumentChannelCalibrationMultiSourceSourceData:
    """Data-agnostic input for one independent astrophysical source."""

    astrophysical_source_id: str
    reference: InstrumentChannelCalibrationMultiSourceChannelData
    channel: InstrumentChannelCalibrationMultiSourceChannelData
    is_primary_source: bool = True
    derivation_parent_source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "astrophysical_source_id",
            _text(
                self.astrophysical_source_id,
                name="astrophysical_source_id",
            ),
        )
        if not isinstance(
            self.reference,
            InstrumentChannelCalibrationMultiSourceChannelData,
        ):
            raise TypeError("reference must be channel data.")
        if not isinstance(
            self.channel,
            InstrumentChannelCalibrationMultiSourceChannelData,
        ):
            raise TypeError("channel must be channel data.")
        if not isinstance(self.is_primary_source, bool):
            raise TypeError("is_primary_source must be boolean.")
        parent = self.derivation_parent_source_id
        if parent is not None:
            parent = _text(parent, name="derivation_parent_source_id")
        object.__setattr__(self, "derivation_parent_source_id", parent)


@dataclass(frozen=True)
class InstrumentChannelCalibrationMultiSourceEligibilityResult:
    """Pre-fit structural and informativeness assessment for one source."""

    astrophysical_source_id: str
    source_id_sha256: str
    eligible: bool
    reasons: tuple[str, ...]
    n_reference_input_rows: int
    n_channel_input_rows: int
    n_reference_usable_rows: int
    n_channel_usable_rows: int
    n_matched_pairs: int
    temporal_fold_pair_counts: tuple[int, ...]
    temporal_fold_amplitude_to_median_reference_error: tuple[float, ...]

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "astrophysical_source_id": self.astrophysical_source_id,
            "source_id_sha256": self.source_id_sha256,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "n_reference_input_rows": self.n_reference_input_rows,
            "n_channel_input_rows": self.n_channel_input_rows,
            "n_reference_usable_rows": self.n_reference_usable_rows,
            "n_channel_usable_rows": self.n_channel_usable_rows,
            "n_matched_pairs": self.n_matched_pairs,
            "temporal_fold_pair_counts": list(
                self.temporal_fold_pair_counts
            ),
            "temporal_fold_amplitude_to_median_reference_error": list(
                self.temporal_fold_amplitude_to_median_reference_error
            ),
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationMultiSourceFoldResult:
    """Held-out temporal-fold evidence without refitting."""

    fold_index: int
    n_holdout_pairs: int
    successful: bool
    holdout_reference_flux_q05_q95_amplitude: float | None
    holdout_median_reference_flux_error: float | None
    holdout_amplitude_to_median_reference_error: float | None
    holdout_normalized_rmse: float | None
    holdout_median_bias_normalized: float | None
    maximum_absolute_time_separation: float | None
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "n_holdout_pairs": self.n_holdout_pairs,
            "successful": self.successful,
            "holdout_reference_flux_q05_q95_amplitude": (
                self.holdout_reference_flux_q05_q95_amplitude
            ),
            "holdout_median_reference_flux_error": (
                self.holdout_median_reference_flux_error
            ),
            "holdout_amplitude_to_median_reference_error": (
                self.holdout_amplitude_to_median_reference_error
            ),
            "holdout_normalized_rmse": self.holdout_normalized_rmse,
            "holdout_median_bias_normalized": (
                self.holdout_median_bias_normalized
            ),
            "maximum_absolute_time_separation": (
                self.maximum_absolute_time_separation
            ),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationMultiSourceTrainingSample:
    """Auditable source-balanced subsample used in one held-out fit."""

    astrophysical_source_id: str
    source_id_sha256: str
    derived_seed: int
    n_available_pairs: int
    selected_pair_positions: tuple[int, ...]

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "astrophysical_source_id": self.astrophysical_source_id,
            "source_id_sha256": self.source_id_sha256,
            "derived_seed": self.derived_seed,
            "n_available_pairs": self.n_available_pairs,
            "selected_pair_positions": list(self.selected_pair_positions),
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationMultiSourceHoldoutResult:
    """One leave-one-source-out transfer test."""

    heldout_astrophysical_source_id: str
    heldout_source_id_sha256: str
    training_source_id_sha256s: tuple[str, ...]
    training_samples: tuple[
        InstrumentChannelCalibrationMultiSourceTrainingSample,
        ...,
    ]
    matched_pairs_per_training_source: int
    n_training_pairs: int
    fitted_offset: float | None
    fitted_scale: float | None
    fold_results: tuple[
        InstrumentChannelCalibrationMultiSourceFoldResult,
        ...,
    ]
    source_holdout_normalized_rmse: float | None
    source_holdout_bias_normalized: float | None
    successful: bool
    reasons: tuple[str, ...]

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "heldout_astrophysical_source_id": (
                self.heldout_astrophysical_source_id
            ),
            "heldout_source_id_sha256": self.heldout_source_id_sha256,
            "training_source_id_sha256s": list(
                self.training_source_id_sha256s
            ),
            "training_samples": [
                sample.to_private_dict() for sample in self.training_samples
            ],
            "matched_pairs_per_training_source": (
                self.matched_pairs_per_training_source
            ),
            "n_training_pairs": self.n_training_pairs,
            "fitted_offset": self.fitted_offset,
            "fitted_scale": self.fitted_scale,
            "fold_results": [
                result.to_dict() for result in self.fold_results
            ],
            "source_holdout_normalized_rmse": (
                self.source_holdout_normalized_rmse
            ),
            "source_holdout_bias_normalized": (
                self.source_holdout_bias_normalized
            ),
            "successful": self.successful,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationMultiSourceExecutionResult:
    """Private detailed evidence plus its redacted public summary."""

    selection_seed: int
    requested_source_count: int | str
    eligible_source_ids: tuple[str, ...]
    selected_source_ids: tuple[str, ...]
    eligibility_results: tuple[
        InstrumentChannelCalibrationMultiSourceEligibilityResult,
        ...,
    ]
    holdout_results: tuple[
        InstrumentChannelCalibrationMultiSourceHoldoutResult,
        ...,
    ]
    disposition: InstrumentChannelCalibrationMultiSourceValidationDisposition
    reasons: tuple[str, ...]
    summary: InstrumentChannelCalibrationMultiSourceValidationSummary

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "selection_seed": self.selection_seed,
            "requested_source_count": self.requested_source_count,
            "eligible_source_ids": list(self.eligible_source_ids),
            "selected_source_ids": list(self.selected_source_ids),
            "eligibility_results": [
                result.to_private_dict()
                for result in self.eligibility_results
            ],
            "holdout_results": [
                result.to_private_dict()
                for result in self.holdout_results
            ],
            "disposition": self.disposition.value,
            "reasons": list(self.reasons),
            "redacted_summary": self.summary.to_dict(),
            "catalogue_population_performed": False,
        }


@dataclass(frozen=True)
class _PreparedSource:
    source_id: str
    source_hash: str
    reference_flux: np.ndarray
    channel_flux: np.ndarray
    reference_error: np.ndarray
    channel_error: np.ndarray
    time_separation: np.ndarray
    fold_positions: tuple[np.ndarray, ...]


def _verify_protocol_binding(
    protocol: InstrumentChannelCalibrationMultiSourceValidationProtocol,
    candidate_protocol: InstrumentChannelCalibrationValidationProtocol,
) -> None:
    comparisons = {
        "candidate protocol id": (
            protocol.candidate_protocol_id,
            candidate_protocol.protocol_id,
        ),
        "candidate protocol version": (
            protocol.candidate_protocol_version,
            candidate_protocol.protocol_version,
        ),
        "candidate protocol digest": (
            protocol.candidate_protocol_sha256,
            candidate_protocol.canonical_sha256,
        ),
        "rule id": (protocol.rule_id, candidate_protocol.rule_id),
        "reference channel": (
            protocol.reference_channel,
            candidate_protocol.reference_channel,
        ),
        "target channel": (protocol.channel, candidate_protocol.channel),
        "physical wavelength": (
            protocol.physical_wavelength,
            candidate_protocol.physical_wavelength,
        ),
    }
    mismatched = [
        name for name, (left, right) in comparisons.items() if left != right
    ]
    if mismatched:
        raise ValueError(
            "Multi-source and candidate protocols do not match: "
            + ", ".join(mismatched)
            + "."
        )


def _usable_rows(
    data: InstrumentChannelCalibrationMultiSourceChannelData,
    *,
    expected_channel: str,
    physical_wavelength: float,
    wavelength_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data.channel != expected_channel:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    time = np.asarray(data.time, dtype=float)
    flux = np.asarray(data.flux, dtype=float)
    error = np.asarray(data.flux_error, dtype=float)
    wavelength = np.asarray(data.wavelength, dtype=float)

    usable = (
        np.isfinite(time)
        & np.isfinite(flux)
        & np.isfinite(error)
        & np.isfinite(wavelength)
        & (flux > 0.0)
        & (error > 0.0)
        & np.isclose(
            wavelength,
            physical_wavelength,
            rtol=0.0,
            atol=wavelength_tolerance,
        )
    )
    return time[usable], flux[usable], error[usable]


def _construct_partitioned_pairs(
    reference_times: np.ndarray,
    channel_times: np.ndarray,
    *,
    protocol: InstrumentChannelCalibrationMultiSourceValidationProtocol,
    candidate_protocol: InstrumentChannelCalibrationValidationProtocol,
) -> tuple[np.ndarray, np.ndarray]:
    tolerance = float(candidate_protocol.maximum_time_separation or 0.0)
    combined = [
        (float(value), 0, index)
        for index, value in enumerate(reference_times)
    ]
    combined.extend(
        (float(value), 1, index)
        for index, value in enumerate(channel_times)
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
            time_unit=candidate_protocol.time_unit,
            method=candidate_protocol.pairing_method,
            maximum_time_separation=(
                candidate_protocol.maximum_time_separation
            ),
            reference_row_indices=local_reference,
            channel_row_indices=local_channel,
        )
        reference_positions.extend(pairing.reference_row_indices)
        channel_positions.extend(pairing.channel_row_indices)

    if not reference_positions:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    reference_array = np.asarray(reference_positions, dtype=int)
    channel_array = np.asarray(channel_positions, dtype=int)
    midpoint = 0.5 * (
        reference_times[reference_array] + channel_times[channel_array]
    )
    ordering = np.lexsort((channel_array, reference_array, midpoint))
    return reference_array[ordering], channel_array[ordering]


def _prepare_source(
    source: InstrumentChannelCalibrationMultiSourceSourceData,
    *,
    protocol: InstrumentChannelCalibrationMultiSourceValidationProtocol,
    candidate_protocol: InstrumentChannelCalibrationValidationProtocol,
) -> tuple[
    InstrumentChannelCalibrationMultiSourceEligibilityResult,
    _PreparedSource | None,
]:
    reasons: list[str] = []

    if not source.is_primary_source:
        reasons.append("source_not_asserted_primary")
    if source.derivation_parent_source_id is not None:
        reasons.append("derived_source_excluded")

    reference_time, reference_flux, reference_error = _usable_rows(
        source.reference,
        expected_channel=protocol.reference_channel,
        physical_wavelength=protocol.physical_wavelength,
        wavelength_tolerance=(
            protocol.physical_wavelength_absolute_tolerance
        ),
    )
    channel_time, channel_flux, channel_error = _usable_rows(
        source.channel,
        expected_channel=protocol.channel,
        physical_wavelength=protocol.physical_wavelength,
        wavelength_tolerance=(
            protocol.physical_wavelength_absolute_tolerance
        ),
    )

    if source.reference.channel != protocol.reference_channel:
        reasons.append("missing_exact_reference_observational_channel")
    elif reference_time.size == 0:
        reasons.append("no_usable_reference_channel_rows")

    if source.channel.channel != protocol.channel:
        reasons.append("missing_exact_target_observational_channel")
    elif channel_time.size == 0:
        reasons.append("no_usable_target_channel_rows")

    reference_positions = np.empty(0, dtype=int)
    channel_positions = np.empty(0, dtype=int)
    if reference_time.size and channel_time.size:
        reference_positions, channel_positions = _construct_partitioned_pairs(
            reference_time,
            channel_time,
            protocol=protocol,
            candidate_protocol=candidate_protocol,
        )

    n_pairs = int(reference_positions.size)
    if n_pairs < protocol.minimum_matched_pairs_per_source:
        reasons.append("insufficient_matched_pairs")

    fold_counts: tuple[int, ...] = ()
    fold_ratios: tuple[float, ...] = ()
    fold_positions: tuple[np.ndarray, ...] = ()
    if n_pairs:
        paired_reference_flux = reference_flux[reference_positions]
        paired_reference_error = reference_error[reference_positions]
        fold_positions = tuple(
            np.asarray(item, dtype=int)
            for item in np.array_split(
                np.arange(n_pairs, dtype=int),
                protocol.temporal_fold_count,
            )
        )
        counts: list[int] = []
        ratios: list[float] = []
        for index, positions in enumerate(fold_positions):
            count = int(positions.size)
            counts.append(count)
            if count:
                amplitude = float(
                    np.quantile(
                        paired_reference_flux[positions],
                        0.95,
                        method="linear",
                    )
                    - np.quantile(
                        paired_reference_flux[positions],
                        0.05,
                        method="linear",
                    )
                )
                median_error = float(
                    np.median(paired_reference_error[positions])
                )
                ratio = (
                    amplitude / median_error
                    if median_error > 0.0
                    else math.nan
                )
            else:
                ratio = math.nan
            ratios.append(float(ratio))
            if count < protocol.minimum_holdout_pairs_per_fold:
                reasons.append(f"fold_{index}_insufficient_holdout_pairs")
            if (
                not math.isfinite(ratio)
                or ratio
                < protocol.minimum_holdout_amplitude_to_median_reference_error
            ):
                reasons.append(f"fold_{index}_insufficient_dynamic_range")
        fold_counts = tuple(counts)
        fold_ratios = tuple(ratios)

    eligible = not reasons
    result = InstrumentChannelCalibrationMultiSourceEligibilityResult(
        astrophysical_source_id=source.astrophysical_source_id,
        source_id_sha256=_source_hash(source.astrophysical_source_id),
        eligible=eligible,
        reasons=tuple(reasons),
        n_reference_input_rows=len(source.reference.time),
        n_channel_input_rows=len(source.channel.time),
        n_reference_usable_rows=int(reference_time.size),
        n_channel_usable_rows=int(channel_time.size),
        n_matched_pairs=n_pairs,
        temporal_fold_pair_counts=fold_counts,
        temporal_fold_amplitude_to_median_reference_error=fold_ratios,
    )
    if not eligible:
        return result, None

    prepared = _PreparedSource(
        source_id=source.astrophysical_source_id,
        source_hash=result.source_id_sha256,
        reference_flux=reference_flux[reference_positions],
        channel_flux=channel_flux[channel_positions],
        reference_error=reference_error[reference_positions],
        channel_error=channel_error[channel_positions],
        time_separation=np.abs(
            reference_time[reference_positions]
            - channel_time[channel_positions]
        ),
        fold_positions=fold_positions,
    )
    return result, prepared


def _subsample_positions(
    *,
    n_pairs: int,
    sample_size: int,
    selection_seed: int,
    heldout_source_hash: str,
    training_source_hash: str,
) -> tuple[np.ndarray, int]:
    material = (
        f"{selection_seed}|{heldout_source_hash}|{training_source_hash}"
    ).encode()
    derived_seed = int.from_bytes(
        hashlib.sha256(material).digest()[:8],
        byteorder="big",
        signed=False,
    )
    generator = np.random.default_rng(derived_seed)
    positions = generator.choice(
        n_pairs,
        size=sample_size,
        replace=False,
    )
    return np.sort(np.asarray(positions, dtype=int)), derived_seed


def _evaluate_fold(
    *,
    fold_index: int,
    positions: np.ndarray,
    heldout: _PreparedSource,
    offset: float,
    scale: float,
    protocol: InstrumentChannelCalibrationMultiSourceValidationProtocol,
) -> InstrumentChannelCalibrationMultiSourceFoldResult:
    amplitude = float(
        np.quantile(
            heldout.reference_flux[positions],
            0.95,
            method="linear",
        )
        - np.quantile(
            heldout.reference_flux[positions],
            0.05,
            method="linear",
        )
    )
    median_error = float(np.median(heldout.reference_error[positions]))
    ratio = amplitude / median_error
    reasons: list[str] = []
    if positions.size < protocol.minimum_holdout_pairs_per_fold:
        reasons.append("insufficient_holdout_pairs")
    if (
        not math.isfinite(ratio)
        or ratio
        < protocol.minimum_holdout_amplitude_to_median_reference_error
    ):
        reasons.append("insufficient_dynamic_range")

    prediction = offset + scale * heldout.channel_flux[positions]
    residual = heldout.reference_flux[positions] - prediction
    if (
        amplitude <= 0.0
        or np.any(~np.isfinite(prediction))
        or np.any(~np.isfinite(residual))
    ):
        reasons.append("nonfinite_or_zero_amplitude_holdout_metrics")

    successful = not reasons
    return InstrumentChannelCalibrationMultiSourceFoldResult(
        fold_index=fold_index,
        n_holdout_pairs=int(positions.size),
        successful=successful,
        holdout_reference_flux_q05_q95_amplitude=amplitude,
        holdout_median_reference_flux_error=median_error,
        holdout_amplitude_to_median_reference_error=ratio,
        holdout_normalized_rmse=(
            float(np.sqrt(np.mean(residual**2)) / amplitude)
            if successful
            else None
        ),
        holdout_median_bias_normalized=(
            float(np.median(residual) / amplitude)
            if successful
            else None
        ),
        maximum_absolute_time_separation=(
            float(np.max(heldout.time_separation[positions]))
            if positions.size
            else None
        ),
        reasons=tuple(reasons),
    )


def _execute_holdout(
    *,
    heldout: _PreparedSource,
    training: tuple[_PreparedSource, ...],
    selection_seed: int,
    protocol: InstrumentChannelCalibrationMultiSourceValidationProtocol,
    candidate_protocol: InstrumentChannelCalibrationValidationProtocol,
) -> InstrumentChannelCalibrationMultiSourceHoldoutResult:
    pairs_per_source = min(
        int(source.reference_flux.size) for source in training
    )
    reference_blocks: list[np.ndarray] = []
    channel_blocks: list[np.ndarray] = []
    reference_error_blocks: list[np.ndarray] = []
    channel_error_blocks: list[np.ndarray] = []
    training_samples: list[
        InstrumentChannelCalibrationMultiSourceTrainingSample
    ] = []

    for source in training:
        positions, derived_seed = _subsample_positions(
            n_pairs=int(source.reference_flux.size),
            sample_size=pairs_per_source,
            selection_seed=selection_seed,
            heldout_source_hash=heldout.source_hash,
            training_source_hash=source.source_hash,
        )
        reference_blocks.append(source.reference_flux[positions])
        channel_blocks.append(source.channel_flux[positions])
        reference_error_blocks.append(source.reference_error[positions])
        channel_error_blocks.append(source.channel_error[positions])
        training_samples.append(
            InstrumentChannelCalibrationMultiSourceTrainingSample(
                astrophysical_source_id=source.source_id,
                source_id_sha256=source.source_hash,
                derived_seed=derived_seed,
                n_available_pairs=int(source.reference_flux.size),
                selected_pair_positions=tuple(
                    int(position) for position in positions
                ),
            )
        )

    try:
        calibration = fit_instrument_channel_calibration(
            np.concatenate(reference_blocks),
            np.concatenate(channel_blocks),
            reference_channel=protocol.reference_channel,
            channel=protocol.channel,
            wavelength=protocol.physical_wavelength,
            reference_error=np.concatenate(reference_error_blocks),
            channel_error=np.concatenate(channel_error_blocks),
            sigma_clip=candidate_protocol.sigma_clip,
            max_iter=candidate_protocol.maximum_fit_iterations,
            min_pairs=candidate_protocol.minimum_fit_pairs,
        )
    except Exception as exc:
        return InstrumentChannelCalibrationMultiSourceHoldoutResult(
            heldout_astrophysical_source_id=heldout.source_id,
            heldout_source_id_sha256=heldout.source_hash,
            training_source_id_sha256s=tuple(
                source.source_hash for source in training
            ),
            training_samples=tuple(training_samples),
            matched_pairs_per_training_source=pairs_per_source,
            n_training_pairs=pairs_per_source * len(training),
            fitted_offset=None,
            fitted_scale=None,
            fold_results=(),
            source_holdout_normalized_rmse=None,
            source_holdout_bias_normalized=None,
            successful=False,
            reasons=(f"calibration_fit_failed:{type(exc).__name__}:{exc}",),
        )

    folds = tuple(
        _evaluate_fold(
            fold_index=index,
            positions=positions,
            heldout=heldout,
            offset=float(calibration.offset),
            scale=float(calibration.scale),
            protocol=protocol,
        )
        for index, positions in enumerate(heldout.fold_positions)
    )
    successful = all(fold.successful for fold in folds)
    nrmse_values = [
        fold.holdout_normalized_rmse
        for fold in folds
        if fold.holdout_normalized_rmse is not None
    ]
    bias_values = [
        fold.holdout_median_bias_normalized
        for fold in folds
        if fold.holdout_median_bias_normalized is not None
    ]
    reasons = (
        ()
        if successful
        else ("one_or_more_heldout_temporal_folds_unsuccessful",)
    )
    return InstrumentChannelCalibrationMultiSourceHoldoutResult(
        heldout_astrophysical_source_id=heldout.source_id,
        heldout_source_id_sha256=heldout.source_hash,
        training_source_id_sha256s=tuple(
            source.source_hash for source in training
        ),
        training_samples=tuple(training_samples),
        matched_pairs_per_training_source=pairs_per_source,
        n_training_pairs=pairs_per_source * len(training),
        fitted_offset=float(calibration.offset),
        fitted_scale=float(calibration.scale),
        fold_results=folds,
        source_holdout_normalized_rmse=(
            float(np.median(nrmse_values)) if successful else None
        ),
        source_holdout_bias_normalized=(
            float(np.median(bias_values)) if successful else None
        ),
        successful=successful,
        reasons=reasons,
    )


def execute_instrument_channel_calibration_multisource_validation(
    sources: Sequence[InstrumentChannelCalibrationMultiSourceSourceData],
    *,
    protocol: InstrumentChannelCalibrationMultiSourceValidationProtocol,
    candidate_protocol: InstrumentChannelCalibrationValidationProtocol,
    private_input_sha256: str,
    selection_seed: int,
    n_sources: int | str = 5,
    summary_id: str = "maintainer-private-kelt-r3-multisource-run-v1",
    summary_version: str = "1.0",
    package_version: str,
    package_commit: str,
    executed_at_utc: str,
) -> InstrumentChannelCalibrationMultiSourceExecutionResult:
    """Run the frozen source-balanced leave-one-source-out validation."""

    if not isinstance(
        protocol,
        InstrumentChannelCalibrationMultiSourceValidationProtocol,
    ):
        raise TypeError("protocol must be a multi-source protocol.")
    if not isinstance(
        candidate_protocol,
        InstrumentChannelCalibrationValidationProtocol,
    ):
        raise TypeError("candidate_protocol must be a validation protocol.")
    _verify_protocol_binding(protocol, candidate_protocol)

    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise TypeError("selection_seed must be an integer.")
    if selection_seed < 0:
        raise ValueError("selection_seed must be non-negative.")

    if n_sources != "all":
        if isinstance(n_sources, bool) or not isinstance(n_sources, int):
            raise TypeError("n_sources must be an integer or 'all'.")
        if n_sources < protocol.required_source_count:
            raise ValueError(
                "n_sources cannot be smaller than required_source_count."
            )

    if not isinstance(sources, Sequence):
        raise TypeError("sources must be a sequence.")
    normalized_sources = tuple(sources)
    if any(
        not isinstance(
            source,
            InstrumentChannelCalibrationMultiSourceSourceData,
        )
        for source in normalized_sources
    ):
        raise TypeError("Every source must be multi-source source data.")

    ids = [source.astrophysical_source_id for source in normalized_sources]
    if len(ids) != len(set(ids)):
        raise ValueError("astrophysical_source_id values must be unique.")

    eligibility: list[
        InstrumentChannelCalibrationMultiSourceEligibilityResult
    ] = []
    prepared_by_id: dict[str, _PreparedSource] = {}
    for source in normalized_sources:
        result, prepared = _prepare_source(
            source,
            protocol=protocol,
            candidate_protocol=candidate_protocol,
        )
        eligibility.append(result)
        if prepared is not None:
            prepared_by_id[source.astrophysical_source_id] = prepared

    eligible_ids = sorted(prepared_by_id)
    permutation = list(eligible_ids)
    random.Random(selection_seed).shuffle(permutation)
    requested_count = (
        len(permutation) if n_sources == "all" else int(n_sources)
    )
    selected_ids = tuple(permutation[:requested_count])

    holdouts: tuple[
        InstrumentChannelCalibrationMultiSourceHoldoutResult,
        ...,
    ] = ()
    reasons: list[str] = []
    disposition = (
        InstrumentChannelCalibrationMultiSourceValidationDisposition.INCONCLUSIVE
    )

    if len(eligible_ids) < protocol.required_source_count:
        reasons.append("insufficient_private_validation_sample")
    elif n_sources != "all" and len(eligible_ids) < requested_count:
        reasons.append("insufficient_requested_source_count")
    elif len(selected_ids) < protocol.required_source_count:
        reasons.append("insufficient_selected_source_count")
    else:
        selected = tuple(prepared_by_id[item] for item in selected_ids)
        holdouts = tuple(
            _execute_holdout(
                heldout=heldout,
                training=tuple(
                    source for source in selected if source is not heldout
                ),
                selection_seed=selection_seed,
                protocol=protocol,
                candidate_protocol=candidate_protocol,
            )
            for heldout in selected
        )
        if any(not result.successful for result in holdouts):
            reasons.append("one_or_more_source_holdouts_unsuccessful")
        else:
            source_nrmse = np.asarray(
                [
                    result.source_holdout_normalized_rmse
                    for result in holdouts
                ],
                dtype=float,
            )
            source_bias = np.asarray(
                [
                    result.source_holdout_bias_normalized
                    for result in holdouts
                ],
                dtype=float,
            )
            median_nrmse = float(np.median(source_nrmse))
            worst_nrmse = float(np.max(source_nrmse))
            median_bias = float(np.median(source_bias))
            if (
                median_nrmse
                > protocol.maximum_median_source_holdout_normalized_rmse
            ):
                reasons.append(
                    "median_source_holdout_normalized_rmse_exceeds_gate"
                )
            if (
                worst_nrmse
                > protocol.maximum_worst_source_holdout_normalized_rmse
            ):
                reasons.append(
                    "worst_source_holdout_normalized_rmse_exceeds_gate"
                )
            if (
                abs(median_bias)
                > protocol.maximum_absolute_median_source_holdout_bias_normalized
            ):
                reasons.append(
                    "absolute_median_source_holdout_bias_exceeds_gate"
                )
            disposition = (
                InstrumentChannelCalibrationMultiSourceValidationDisposition.PASSED
                if not reasons
                else InstrumentChannelCalibrationMultiSourceValidationDisposition.FAILED
            )

    successful_holdouts = tuple(
        result for result in holdouts if result.successful
    )
    source_nrmse_values = [
        result.source_holdout_normalized_rmse
        for result in successful_holdouts
        if result.source_holdout_normalized_rmse is not None
    ]
    source_bias_values = [
        result.source_holdout_bias_normalized
        for result in successful_holdouts
        if result.source_holdout_bias_normalized is not None
    ]
    scales = [
        result.fitted_scale
        for result in successful_holdouts
        if result.fitted_scale is not None
    ]
    offsets = [
        result.fitted_offset
        for result in successful_holdouts
        if result.fitted_offset is not None
    ]

    summary = InstrumentChannelCalibrationMultiSourceValidationSummary(
        summary_id=summary_id,
        summary_version=summary_version,
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.protocol_version,
        protocol_sha256=protocol.canonical_sha256,
        candidate_protocol_id=protocol.candidate_protocol_id,
        candidate_protocol_version=protocol.candidate_protocol_version,
        candidate_protocol_sha256=protocol.candidate_protocol_sha256,
        rule_id=protocol.rule_id,
        private_input_sha256=private_input_sha256,
        selection_seed=selection_seed,
        eligible_source_count=len(eligible_ids),
        selected_source_id_sha256s=tuple(
            _source_hash(item) for item in selected_ids
        ),
        successful_source_holdout_count=len(successful_holdouts),
        median_source_holdout_normalized_rmse=(
            float(np.median(source_nrmse_values))
            if source_nrmse_values
            else None
        ),
        worst_source_holdout_normalized_rmse=(
            float(np.max(source_nrmse_values))
            if source_nrmse_values
            else None
        ),
        median_source_holdout_bias_normalized=(
            float(np.median(source_bias_values))
            if source_bias_values
            else None
        ),
        fitted_scale_median=(
            float(np.median(scales)) if scales else None
        ),
        fitted_scale_mad=_mad(scales),
        fitted_offset_median=(
            float(np.median(offsets)) if offsets else None
        ),
        fitted_offset_mad=_mad(offsets),
        package_version=package_version,
        package_commit=package_commit,
        executed_at_utc=executed_at_utc,
        disposition=disposition,
        reasons=tuple(reasons),
        all_acceptance_criteria_passed=(
            disposition
            is InstrumentChannelCalibrationMultiSourceValidationDisposition.PASSED
        ),
    )
    return InstrumentChannelCalibrationMultiSourceExecutionResult(
        selection_seed=selection_seed,
        requested_source_count=n_sources,
        eligible_source_ids=tuple(eligible_ids),
        selected_source_ids=selected_ids,
        eligibility_results=tuple(eligibility),
        holdout_results=holdouts,
        disposition=disposition,
        reasons=tuple(reasons),
        summary=summary,
    )
