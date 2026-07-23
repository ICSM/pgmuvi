"""Explicit observational-channel calibration primitives.

An observational channel identifies an instrument, detector, filter, or data
stream.  It is distinct from the numeric physical wavelength coordinate used by
the GP, and multiple observational channels may share one physical wavelength.

This module provides immutable, JSON-safe requirement, pairing, and
dataset-level orchestration records and execution results, explicit
deterministic time-pair construction and plan-execution callables, and low-level
fitting and application primitives for an affine mapping.  It does not choose
a reference channel, pairing method, time tolerance, or calibration family;
merge channels; alter wavelengths; or integrate calibration automatically into
a light-curve fit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any

import numpy as np
from scipy import optimize


INSTRUMENT_CHANNEL_CALIBRATION_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER = (
    "TBD[instrument-channel-calibration]"
)
INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-pairing-v2"
)
INSTRUMENT_CHANNEL_CALIBRATION_PLAN_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-plan-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_EXECUTION_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-execution-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-uncertainty-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_FIT_PROVENANCE_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-fit-provenance-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_SCALE_DEPENDENT_UNCERTAINTY_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-scale-dependent-uncertainty-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_UNCERTAINTY_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-predictive-uncertainty-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_ORCHESTRATION_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-predictive-orchestration-v1"
)


__all__ = [
    "INSTRUMENT_CHANNEL_CALIBRATION_EXECUTION_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_FIT_PROVENANCE_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_PLAN_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_ORCHESTRATION_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_UNCERTAINTY_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_SCALE_DEPENDENT_UNCERTAINTY_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER",
    "INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION",
    "InstrumentChannelCalibration",
    "InstrumentChannelCalibrationAssessment",
    "InstrumentChannelCalibrationChannelPlan",
    "InstrumentChannelCalibrationCoefficientUncertainty",
    "InstrumentChannelCalibrationDisposition",
    "InstrumentChannelCalibrationExecution",
    "InstrumentChannelCalibrationFitProvenance",
    "InstrumentChannelCalibrationGroupPlan",
    "InstrumentChannelCalibrationPlan",
    "InstrumentChannelCalibrationPredictiveCovarianceMode",
    "InstrumentChannelCalibrationPredictiveUncertainty",
    "InstrumentChannelCalibrationPredictiveUncertaintyChannelResult",
    "InstrumentChannelCalibrationPredictiveUncertaintyDisposition",
    "InstrumentChannelCalibrationPredictiveUncertaintyGroupResult",
    "InstrumentChannelCalibrationPredictiveUncertaintyOrchestration",
    "InstrumentChannelCalibrationPredictiveUncertaintyRequest",
    "InstrumentChannelCalibrationPredictiveUncertaintyStatus",
    "InstrumentChannelCalibrationScaleDependentUncertaintyEstimate",
    "InstrumentChannelCalibrationStatus",
    "InstrumentChannelCalibrationUncertaintyEstimator",
    "InstrumentChannelCalibrationUncertaintyIntegrationStatus",
    "InstrumentChannelCalibrationUncertaintyStatus",
    "InstrumentChannelPairing",
    "InstrumentChannelPairingMethod",
    "SharedWavelengthChannelGroup",
    "apply_instrument_channel_calibration",
    "apply_instrument_channel_calibration_with_predictive_uncertainty",
    "assess_instrument_channel_calibration_requirement",
    "construct_instrument_channel_pairing",
    "define_instrument_channel_calibration_plan",
    "estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty",
    "execute_instrument_channel_calibration_plan",
    "fit_instrument_channel_calibration",
    "select_instrument_channel_calibration_uncertainty_estimator",
]


class _StringEnum(str, Enum):
    """Enum whose members serialize as stable public strings."""

    def __str__(self) -> str:
        return self.value


class InstrumentChannelCalibrationStatus(_StringEnum):
    """Implementation state for an observational-channel calibration need."""

    NOT_REQUIRED = "not_required"
    REQUIRED_NOT_IMPLEMENTED = "required_not_implemented"


class InstrumentChannelPairingMethod(_StringEnum):
    """Caller-selected deterministic time-pair construction method."""

    EXACT_TIMESTAMP = "exact_timestamp"
    NEAREST_WITHIN_TOLERANCE = "nearest_within_tolerance"


class InstrumentChannelCalibrationDisposition(_StringEnum):
    """Caller-selected disposition for one non-reference channel."""

    PLANNED = "planned"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


class InstrumentChannelCalibrationUncertaintyStatus(_StringEnum):
    """Availability of affine coefficient-uncertainty provenance."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class InstrumentChannelCalibrationUncertaintyEstimator(_StringEnum):
    """Selected affine coefficient-uncertainty estimator family."""

    FIXED_WEIGHT_NORMAL_MATRIX = "fixed_weight_normal_matrix"
    SCALE_DEPENDENT_FULL_OBJECTIVE = "scale_dependent_full_objective"


class InstrumentChannelCalibrationUncertaintyIntegrationStatus(_StringEnum):
    """Integration state for the selected uncertainty estimator."""

    ACTIVE = "active"
    DEFINED_NOT_ACTIVATED = "defined_not_activated"
    ATTEMPTED_UNAVAILABLE_FALLBACK = (
        "attempted_unavailable_fallback"
    )


class InstrumentChannelCalibrationPredictiveUncertaintyStatus(
    _StringEnum
):
    """Availability of one requested predictive-uncertainty result."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class InstrumentChannelCalibrationPredictiveUncertaintyDisposition(
    _StringEnum
):
    """Dataset-orchestration disposition for predictive propagation."""

    NOT_REQUESTED = "not_requested"
    AVAILABLE = "available"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


class InstrumentChannelCalibrationPredictiveCovarianceMode(_StringEnum):
    """Requested predictive-covariance representation."""

    MARGINAL_VARIANCE = "marginal_variance"
    FULL_COVARIANCE = "full_covariance"


@dataclass(frozen=True)
class SharedWavelengthChannelGroup:
    """Observational channels attached to one physical wavelength."""

    physical_wavelength: float
    observational_channels: tuple[str, ...]

    def __post_init__(self) -> None:
        wavelength = float(self.physical_wavelength)
        if not math.isfinite(wavelength):
            raise ValueError("physical_wavelength must be finite.")

        channels = tuple(
            sorted(
                {
                    str(channel).strip()
                    for channel in self.observational_channels
                    if str(channel).strip()
                }
            )
        )
        if len(channels) < 2:
            raise ValueError(
                "A shared-wavelength group requires at least two distinct "
                "observational channels."
            )

        object.__setattr__(self, "physical_wavelength", wavelength)
        object.__setattr__(self, "observational_channels", channels)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe representation."""

        return {
            "physical_wavelength": self.physical_wavelength,
            "observational_channels": list(self.observational_channels),
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationAssessment:
    """Whether channel calibration is required for the supplied structure."""

    status: InstrumentChannelCalibrationStatus
    required: bool
    shared_wavelength_groups: tuple[SharedWavelengthChannelGroup, ...] = ()
    schema_version: str = INSTRUMENT_CHANNEL_CALIBRATION_SCHEMA_VERSION
    implemented: bool = False
    calibration_applied: bool = False
    silent_calibration_permitted: bool = False
    policy: str = "not_applicable"
    marker: str = INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER

    def __post_init__(self) -> None:
        status = self.status
        if not isinstance(status, InstrumentChannelCalibrationStatus):
            try:
                status = InstrumentChannelCalibrationStatus(str(status))
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported calibration status {self.status!r}."
                ) from exc

        groups = tuple(self.shared_wavelength_groups)
        required = bool(self.required)
        expected_required = bool(groups)
        if required != expected_required:
            raise ValueError(
                "required must agree with the presence of shared-wavelength "
                "channel groups."
            )

        expected_status = (
            InstrumentChannelCalibrationStatus.REQUIRED_NOT_IMPLEMENTED
            if required
            else InstrumentChannelCalibrationStatus.NOT_REQUIRED
        )
        if status is not expected_status:
            raise ValueError(
                "status must agree with the shared-wavelength requirement."
            )

        if self.implemented or self.calibration_applied:
            raise ValueError(
                "Automatic instrument-channel calibration has not been "
                "applied by this requirement assessment."
            )
        if self.silent_calibration_permitted:
            raise ValueError("Silent calibration is never permitted.")

        expected_policy = (
            "preserve_channels_without_calibration"
            if required
            else "not_applicable"
        )
        if self.policy != expected_policy:
            raise ValueError(
                "policy must describe the maintained uncalibrated behavior."
            )

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "shared_wavelength_groups", groups)

    @property
    def shared_wavelength_channels(
        self,
    ) -> dict[float, tuple[str, ...]]:
        """Return channels keyed by their shared physical wavelength."""

        return {
            group.physical_wavelength: group.observational_channels
            for group in self.shared_wavelength_groups
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe representation."""

        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "required": self.required,
            "implemented": self.implemented,
            "calibration_applied": self.calibration_applied,
            "silent_calibration_permitted": (
                self.silent_calibration_permitted
            ),
            "policy": self.policy,
            "marker": self.marker,
            "n_shared_physical_wavelengths": len(
                self.shared_wavelength_groups
            ),
            "shared_wavelength_groups": [
                group.to_dict()
                for group in self.shared_wavelength_groups
            ],
        }


def assess_instrument_channel_calibration_requirement(
    physical_wavelengths: Any,
    observational_channel_labels: Any,
) -> InstrumentChannelCalibrationAssessment:
    """Describe whether shared wavelengths require channel calibration.

    The function only inspects channel-to-wavelength structure.  It does not
    estimate, infer, fit, or apply a calibration correction.
    """

    wavelengths = np.asarray(physical_wavelengths, dtype=float)
    channels = np.asarray(observational_channel_labels, dtype=str)

    if wavelengths.ndim != 1:
        raise ValueError("physical_wavelengths must be one-dimensional.")
    if channels.ndim != 1:
        raise ValueError(
            "observational_channel_labels must be one-dimensional."
        )
    if wavelengths.size != channels.size:
        raise ValueError(
            "physical_wavelengths and observational_channel_labels must "
            "have the same length."
        )

    channel_wavelengths: dict[str, float] = {}
    for wavelength_value, channel_value in zip(
        wavelengths,
        channels,
        strict=True,
    ):
        wavelength = float(wavelength_value)
        channel = str(channel_value).strip()

        if not math.isfinite(wavelength):
            raise ValueError("Physical wavelength coordinates must be finite.")
        if not channel:
            raise ValueError(
                "Observational-channel labels must be non-empty."
            )

        previous = channel_wavelengths.get(channel)
        if previous is not None and previous != wavelength:
            raise ValueError(
                "Each observational channel must map to exactly one physical "
                f"wavelength; {channel!r} maps to both {previous!r} and "
                f"{wavelength!r}."
            )
        channel_wavelengths[channel] = wavelength

    channels_by_wavelength: dict[float, list[str]] = {}
    for channel, wavelength in channel_wavelengths.items():
        channels_by_wavelength.setdefault(wavelength, []).append(channel)

    groups = tuple(
        SharedWavelengthChannelGroup(
            physical_wavelength=wavelength,
            observational_channels=tuple(attached_channels),
        )
        for wavelength, attached_channels in sorted(
            channels_by_wavelength.items()
        )
        if len(attached_channels) > 1
    )
    required = bool(groups)

    return InstrumentChannelCalibrationAssessment(
        status=(
            InstrumentChannelCalibrationStatus.REQUIRED_NOT_IMPLEMENTED
            if required
            else InstrumentChannelCalibrationStatus.NOT_REQUIRED
        ),
        required=required,
        shared_wavelength_groups=groups,
        policy=(
            "preserve_channels_without_calibration"
            if required
            else "not_applicable"
        ),
    )



@dataclass(frozen=True)
class InstrumentChannelPairing:
    """Pairing provenance for two observational channels.

    The record can describe pairs supplied directly by a caller or pairs
    produced by :func:`construct_instrument_channel_pairing`. It preserves
    source-row and time provenance without claiming that the caller-selected
    method or tolerance is scientifically appropriate.

    Pair construction never interpolates measurements, reuses observations,
    chooses a reference channel, or selects a method or tolerance.
    """

    schema_version: str
    reference_channel: str
    channel: str
    wavelength: float
    reference_row_indices: tuple[int, ...]
    channel_row_indices: tuple[int, ...]
    reference_times: tuple[float, ...]
    channel_times: tuple[float, ...]
    time_unit: str
    method: str = "caller_supplied_explicit_pairs"
    maximum_time_separation: float | None = None
    pairing_source: str = "caller_supplied_explicit_pairs"
    n_reference_observations: int | None = None
    n_channel_observations: int | None = None
    allow_reference_reuse: bool = False
    allow_channel_reuse: bool = False
    interpolation_used: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported instrument-channel pairing schema version: "
                f"{self.schema_version!r}."
            )

        reference_channel = _normalize_calibration_channel(
            self.reference_channel,
            name="reference_channel",
        )
        channel = _normalize_calibration_channel(
            self.channel,
            name="channel",
        )
        if reference_channel == channel:
            raise ValueError(
                "reference_channel and channel must identify different "
                "observational channels."
            )

        wavelength = float(self.wavelength)
        if not math.isfinite(wavelength):
            raise ValueError("wavelength must be finite.")

        reference_indices = self._normalize_indices(
            self.reference_row_indices,
            name="reference_row_indices",
        )
        channel_indices = self._normalize_indices(
            self.channel_row_indices,
            name="channel_row_indices",
        )
        reference_times = self._normalize_times(
            self.reference_times,
            name="reference_times",
        )
        channel_times = self._normalize_times(
            self.channel_times,
            name="channel_times",
        )

        lengths = {
            len(reference_indices),
            len(channel_indices),
            len(reference_times),
            len(channel_times),
        }
        if len(lengths) != 1:
            raise ValueError(
                "Pairing index and time sequences must have the same length."
            )
        if not reference_indices:
            raise ValueError(
                "Instrument-channel pairing requires at least one pair."
            )

        time_unit = self._normalize_text(
            self.time_unit,
            name="time_unit",
        )
        method = self._normalize_text(
            self.method,
            name="method",
        )
        pairing_source = self._normalize_text(
            self.pairing_source,
            name="pairing_source",
        )

        maximum_time_separation = self.maximum_time_separation
        if maximum_time_separation is not None:
            if isinstance(
                maximum_time_separation,
                (bool, np.bool_),
            ):
                raise TypeError(
                    "maximum_time_separation must be numeric, not boolean."
                )
            maximum_time_separation = float(maximum_time_separation)
            if (
                not math.isfinite(maximum_time_separation)
                or maximum_time_separation < 0.0
            ):
                raise ValueError(
                    "maximum_time_separation must be finite and "
                    "non-negative when supplied."
                )

        n_reference_observations = self._normalize_optional_count(
            self.n_reference_observations,
            name="n_reference_observations",
        )
        n_channel_observations = self._normalize_optional_count(
            self.n_channel_observations,
            name="n_channel_observations",
        )

        for name in (
            "allow_reference_reuse",
            "allow_channel_reuse",
            "interpolation_used",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")

        if (
            not self.allow_reference_reuse
            and len(set(reference_indices)) != len(reference_indices)
        ):
            raise ValueError(
                "reference_row_indices contain reused rows, but "
                "allow_reference_reuse is False."
            )
        if (
            not self.allow_channel_reuse
            and len(set(channel_indices)) != len(channel_indices)
        ):
            raise ValueError(
                "channel_row_indices contain reused rows, but "
                "allow_channel_reuse is False."
            )

        if (
            n_reference_observations is not None
            and n_reference_observations < len(set(reference_indices))
        ):
            raise ValueError(
                "n_reference_observations cannot be smaller than the "
                "number of distinct matched reference rows."
            )
        if (
            n_channel_observations is not None
            and n_channel_observations < len(set(channel_indices))
        ):
            raise ValueError(
                "n_channel_observations cannot be smaller than the "
                "number of distinct matched channel rows."
            )

        absolute_time_differences = tuple(
            abs(reference_time - channel_time)
            for reference_time, channel_time in zip(
                reference_times,
                channel_times,
                strict=True,
            )
        )

        if method == InstrumentChannelPairingMethod.EXACT_TIMESTAMP.value:
            if (
                maximum_time_separation is not None
                and maximum_time_separation != 0.0
            ):
                raise ValueError(
                    "exact_timestamp pairing permits no non-zero "
                    "maximum_time_separation."
                )
            if any(value != 0.0 for value in absolute_time_differences):
                raise ValueError(
                    "exact_timestamp pairing requires identical paired "
                    "time coordinates."
                )
        elif (
            method
            == InstrumentChannelPairingMethod.NEAREST_WITHIN_TOLERANCE.value
        ):
            if (
                maximum_time_separation is None
                or maximum_time_separation <= 0.0
            ):
                raise ValueError(
                    "nearest_within_tolerance pairing requires a finite "
                    "positive maximum_time_separation."
                )
            if any(
                value > maximum_time_separation
                for value in absolute_time_differences
            ):
                raise ValueError(
                    "A paired time separation exceeds "
                    "maximum_time_separation."
                )

        if pairing_source == "pgmuvi_deterministic_time_matching":
            if (
                self.allow_reference_reuse
                or self.allow_channel_reuse
                or self.interpolation_used
            ):
                raise ValueError(
                    "PGMUVI deterministic pair construction prohibits "
                    "row reuse and interpolation."
                )

        object.__setattr__(
            self,
            "reference_channel",
            reference_channel,
        )
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(
            self,
            "reference_row_indices",
            reference_indices,
        )
        object.__setattr__(
            self,
            "channel_row_indices",
            channel_indices,
        )
        object.__setattr__(
            self,
            "reference_times",
            reference_times,
        )
        object.__setattr__(self, "channel_times", channel_times)
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self,
            "maximum_time_separation",
            maximum_time_separation,
        )
        object.__setattr__(self, "pairing_source", pairing_source)
        object.__setattr__(
            self,
            "n_reference_observations",
            n_reference_observations,
        )
        object.__setattr__(
            self,
            "n_channel_observations",
            n_channel_observations,
        )

    @staticmethod
    def _normalize_text(
        value: Any,
        *,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")

        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{name} must be non-empty.")

        return normalized

    @staticmethod
    def _normalize_optional_count(
        value: Any,
        *,
        name: str,
    ) -> int | None:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(f"{name} must be an integer when supplied.")

        normalized = int(value)
        if normalized < 1:
            raise ValueError(f"{name} must be at least 1 when supplied.")

        return normalized

    @staticmethod
    def _normalize_indices(
        values: Any,
        *,
        name: str,
    ) -> tuple[int, ...]:
        try:
            sequence = tuple(values)
        except TypeError as exc:
            raise TypeError(f"{name} must be an iterable of integers.") from exc

        normalized: list[int] = []
        for value in sequence:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(
                    f"{name} must contain only integer row indices."
                )

            index = int(value)
            if index < 0:
                raise ValueError(
                    f"{name} must contain only non-negative row indices."
                )
            normalized.append(index)

        return tuple(normalized)

    @staticmethod
    def _normalize_times(
        values: Any,
        *,
        name: str,
    ) -> tuple[float, ...]:
        try:
            sequence = tuple(values)
        except TypeError as exc:
            raise TypeError(f"{name} must be an iterable of times.") from exc

        normalized: list[float] = []
        for value in sequence:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"{name} must contain numeric times, not boolean values."
                )

            normalized_value = float(value)
            if not math.isfinite(normalized_value):
                raise ValueError(
                    f"{name} must contain only finite values."
                )

            normalized.append(normalized_value)

        return tuple(normalized)

    @property
    def n_pairs(self) -> int:
        """Number of recorded pairs."""

        return len(self.reference_row_indices)

    @property
    def time_differences(self) -> tuple[float, ...]:
        """Reference time minus channel time for every pair."""

        return tuple(
            reference_time - channel_time
            for reference_time, channel_time in zip(
                self.reference_times,
                self.channel_times,
                strict=True,
            )
        )

    @property
    def absolute_time_differences(self) -> tuple[float, ...]:
        """Absolute time separation for every pair."""

        return tuple(
            abs(value)
            for value in self.time_differences
        )

    @property
    def usable_for_affine_calibration(self) -> bool:
        """Whether the record meets the affine fitter's minimum pair count."""

        return self.n_pairs >= 3

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe provenance representation."""

        absolute_differences = np.asarray(
            self.absolute_time_differences,
            dtype=float,
        )

        return {
            "schema_version": self.schema_version,
            "reference_channel": self.reference_channel,
            "channel": self.channel,
            "wavelength": self.wavelength,
            "reference_row_indices": list(
                self.reference_row_indices
            ),
            "channel_row_indices": list(self.channel_row_indices),
            "reference_times": list(self.reference_times),
            "channel_times": list(self.channel_times),
            "time_differences": list(self.time_differences),
            "time_unit": self.time_unit,
            "method": self.method,
            "maximum_time_separation": self.maximum_time_separation,
            "pairing_source": self.pairing_source,
            "n_reference_observations": self.n_reference_observations,
            "n_channel_observations": self.n_channel_observations,
            "n_unmatched_reference_observations": (
                None
                if self.n_reference_observations is None
                else self.n_reference_observations
                - len(set(self.reference_row_indices))
            ),
            "n_unmatched_channel_observations": (
                None
                if self.n_channel_observations is None
                else self.n_channel_observations
                - len(set(self.channel_row_indices))
            ),
            "n_pairs": self.n_pairs,
            "maximum_absolute_time_difference": float(
                np.max(absolute_differences)
            ),
            "median_absolute_time_difference": float(
                np.median(absolute_differences)
            ),
            "n_exact_time_matches": int(
                np.count_nonzero(absolute_differences == 0.0)
            ),
            "allow_reference_reuse": self.allow_reference_reuse,
            "allow_channel_reuse": self.allow_channel_reuse,
            "interpolation_used": self.interpolation_used,
            "usable_for_affine_calibration": (
                self.usable_for_affine_calibration
            ),
            "caller_supplied_pairing": (
                self.pairing_source == "caller_supplied_explicit_pairs"
            ),
            "automatic_pair_construction": (
                self.pairing_source
                == "pgmuvi_deterministic_time_matching"
            ),
            "automatic_reference_channel_selection": False,
            "automatic_pairing_method_selection": False,
            "automatic_time_tolerance_selection": False,
            "scientific_pairing_validation_performed": False,
        }


def _normalize_pairing_method(
    method: Any,
) -> InstrumentChannelPairingMethod:
    if isinstance(method, InstrumentChannelPairingMethod):
        return method
    if not isinstance(method, str):
        raise TypeError(
            "method must be an InstrumentChannelPairingMethod or string."
        )

    normalized = method.strip()
    try:
        return InstrumentChannelPairingMethod(normalized)
    except ValueError as exc:
        allowed = ", ".join(
            member.value for member in InstrumentChannelPairingMethod
        )
        raise ValueError(
            f"Unknown instrument-channel pairing method {method!r}; "
            f"expected one of: {allowed}."
        ) from exc


def _normalize_pairing_source_indices(
    values: Any | None,
    *,
    size: int,
    name: str,
) -> tuple[int, ...]:
    if values is None:
        return tuple(range(size))

    normalized = InstrumentChannelPairing._normalize_indices(
        values,
        name=name,
    )
    if len(normalized) != size:
        raise ValueError(
            f"{name} must contain one source-row index per input time."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{name} must not identify the same source row more than once."
        )

    return normalized


def _construct_monotonic_time_pairs(
    reference_records: tuple[tuple[float, int, int], ...],
    channel_records: tuple[tuple[float, int, int], ...],
    *,
    maximum_time_separation: float,
) -> tuple[tuple[int, int], ...]:
    """Maximize pair count, then minimize summed absolute separation."""

    n_reference = len(reference_records)
    n_channel = len(channel_records)

    pair_counts = np.zeros(
        (n_reference + 1, n_channel + 1),
        dtype=np.int64,
    )
    total_separations = np.zeros(
        (n_reference + 1, n_channel + 1),
        dtype=float,
    )
    actions = np.zeros(
        (n_reference, n_channel),
        dtype=np.int8,
    )

    match_action = 1
    skip_reference_action = 2
    skip_channel_action = 3

    for reference_position in range(n_reference - 1, -1, -1):
        for channel_position in range(n_channel - 1, -1, -1):
            candidates = [
                (
                    int(pair_counts[reference_position + 1, channel_position]),
                    float(
                        total_separations[
                            reference_position + 1,
                            channel_position,
                        ]
                    ),
                    1,
                    skip_reference_action,
                ),
                (
                    int(pair_counts[reference_position, channel_position + 1]),
                    float(
                        total_separations[
                            reference_position,
                            channel_position + 1,
                        ]
                    ),
                    2,
                    skip_channel_action,
                ),
            ]

            separation = abs(
                reference_records[reference_position][0]
                - channel_records[channel_position][0]
            )
            if separation <= maximum_time_separation:
                candidates.append(
                    (
                        1
                        + int(
                            pair_counts[
                                reference_position + 1,
                                channel_position + 1,
                            ]
                        ),
                        separation
                        + float(
                            total_separations[
                                reference_position + 1,
                                channel_position + 1,
                            ]
                        ),
                        0,
                        match_action,
                    )
                )

            best = max(
                candidates,
                key=lambda candidate: (
                    candidate[0],
                    -candidate[1],
                    -candidate[2],
                ),
            )
            pair_counts[reference_position, channel_position] = best[0]
            total_separations[
                reference_position,
                channel_position,
            ] = best[1]
            actions[reference_position, channel_position] = best[3]

    pairs: list[tuple[int, int]] = []
    reference_position = 0
    channel_position = 0

    while (
        reference_position < n_reference
        and channel_position < n_channel
    ):
        action = int(actions[reference_position, channel_position])
        if action == match_action:
            pairs.append((reference_position, channel_position))
            reference_position += 1
            channel_position += 1
        elif action == skip_reference_action:
            reference_position += 1
        elif action == skip_channel_action:
            channel_position += 1
        else:
            raise RuntimeError(
                "Internal instrument-channel pairing reconstruction failed."
            )

    return tuple(pairs)


def construct_instrument_channel_pairing(
    reference_times: Any,
    channel_times: Any,
    *,
    reference_channel: str,
    channel: str,
    wavelength: float,
    time_unit: str,
    method: InstrumentChannelPairingMethod | str,
    maximum_time_separation: float | None = None,
    reference_row_indices: Any | None = None,
    channel_row_indices: Any | None = None,
) -> InstrumentChannelPairing:
    """Construct deterministic one-to-one time pairs.

    The caller must explicitly choose the reference channel, pairing method,
    and—when using nearest-within-tolerance matching—the positive maximum time
    separation. Inputs must already use the same time coordinate and unit.

    Exact matching accepts only numerically identical timestamps. Nearest-
    within-tolerance matching maximizes the number of chronological one-to-one
    pairs and, among maximum-cardinality solutions, minimizes the summed
    absolute time separation. Deterministic tie-breaking favours earlier sorted
    observations.

    Pair construction uses a dynamic-programming grid with
    ``O(n_reference * n_channel)`` time and memory cost. Callers with large
    channels should restrict inputs to the time range relevant to calibration
    before invoking this callable.

    This callable does not interpolate, reuse observations, select a reference
    channel, choose a method or tolerance, fit a calibration, merge channels,
    or claim that the selected tolerance is scientifically appropriate.
    """

    normalized_method = _normalize_pairing_method(method)

    normalized_reference_times = InstrumentChannelPairing._normalize_times(
        reference_times,
        name="reference_times",
    )
    normalized_channel_times = InstrumentChannelPairing._normalize_times(
        channel_times,
        name="channel_times",
    )

    if not normalized_reference_times:
        raise ValueError("reference_times must contain at least one value.")
    if not normalized_channel_times:
        raise ValueError("channel_times must contain at least one value.")

    normalized_reference_indices = _normalize_pairing_source_indices(
        reference_row_indices,
        size=len(normalized_reference_times),
        name="reference_row_indices",
    )
    normalized_channel_indices = _normalize_pairing_source_indices(
        channel_row_indices,
        size=len(normalized_channel_times),
        name="channel_row_indices",
    )

    if normalized_method is InstrumentChannelPairingMethod.EXACT_TIMESTAMP:
        if maximum_time_separation is None:
            normalized_maximum_separation = 0.0
            recorded_maximum_separation = None
        else:
            if isinstance(
                maximum_time_separation,
                (bool, np.bool_),
            ):
                raise TypeError(
                    "maximum_time_separation must be numeric, not boolean."
                )
            normalized_maximum_separation = float(
                maximum_time_separation
            )
            if normalized_maximum_separation != 0.0:
                raise ValueError(
                    "exact_timestamp pairing requires "
                    "maximum_time_separation to be None or zero."
                )
            recorded_maximum_separation = 0.0
    else:
        if maximum_time_separation is None:
            raise ValueError(
                "nearest_within_tolerance pairing requires "
                "maximum_time_separation."
            )
        if isinstance(maximum_time_separation, (bool, np.bool_)):
            raise TypeError(
                "maximum_time_separation must be numeric, not boolean."
            )

        normalized_maximum_separation = float(maximum_time_separation)
        if (
            not math.isfinite(normalized_maximum_separation)
            or normalized_maximum_separation <= 0.0
        ):
            raise ValueError(
                "nearest_within_tolerance pairing requires a finite "
                "positive maximum_time_separation."
            )
        recorded_maximum_separation = normalized_maximum_separation

    reference_records = tuple(
        sorted(
            (
                (time_value, row_index, input_position)
                for input_position, (time_value, row_index) in enumerate(
                    zip(
                        normalized_reference_times,
                        normalized_reference_indices,
                        strict=True,
                    )
                )
            ),
            key=lambda record: (record[0], record[1], record[2]),
        )
    )
    channel_records = tuple(
        sorted(
            (
                (time_value, row_index, input_position)
                for input_position, (time_value, row_index) in enumerate(
                    zip(
                        normalized_channel_times,
                        normalized_channel_indices,
                        strict=True,
                    )
                )
            ),
            key=lambda record: (record[0], record[1], record[2]),
        )
    )

    matched_positions = _construct_monotonic_time_pairs(
        reference_records,
        channel_records,
        maximum_time_separation=normalized_maximum_separation,
    )
    if not matched_positions:
        raise ValueError(
            "No eligible one-to-one instrument-channel time pairs were "
            "found for the caller-selected method and tolerance."
        )

    matched_reference_records = tuple(
        reference_records[reference_position]
        for reference_position, _ in matched_positions
    )
    matched_channel_records = tuple(
        channel_records[channel_position]
        for _, channel_position in matched_positions
    )

    return InstrumentChannelPairing(
        schema_version=INSTRUMENT_CHANNEL_PAIRING_SCHEMA_VERSION,
        reference_channel=reference_channel,
        channel=channel,
        wavelength=wavelength,
        reference_row_indices=tuple(
            record[1] for record in matched_reference_records
        ),
        channel_row_indices=tuple(
            record[1] for record in matched_channel_records
        ),
        reference_times=tuple(
            record[0] for record in matched_reference_records
        ),
        channel_times=tuple(
            record[0] for record in matched_channel_records
        ),
        time_unit=time_unit,
        method=normalized_method.value,
        maximum_time_separation=recorded_maximum_separation,
        pairing_source="pgmuvi_deterministic_time_matching",
        n_reference_observations=len(normalized_reference_times),
        n_channel_observations=len(normalized_channel_times),
        allow_reference_reuse=False,
        allow_channel_reuse=False,
        interpolation_used=False,
    )


def select_instrument_channel_calibration_uncertainty_estimator(
    *,
    reference_error_supplied: bool,
    channel_error_supplied: bool,
) -> InstrumentChannelCalibrationUncertaintyEstimator:
    """Select the deterministic coefficient-uncertainty estimator family.

    Reference-axis errors alone retain the existing fixed-weight covariance
    path. Any observational-channel-axis error selects the scale-dependent
    full-objective path, whether or not reference-axis errors are also present.
    The affine fitter applies this routing and activates the selected
    estimator after finite filtering and final-inlier selection.
    """

    for name, value in (
        ("reference_error_supplied", reference_error_supplied),
        ("channel_error_supplied", channel_error_supplied),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean.")

    if bool(channel_error_supplied):
        return (
            InstrumentChannelCalibrationUncertaintyEstimator
            .SCALE_DEPENDENT_FULL_OBJECTIVE
        )

    return (
        InstrumentChannelCalibrationUncertaintyEstimator
        .FIXED_WEIGHT_NORMAL_MATRIX
    )


@dataclass(frozen=True)
class InstrumentChannelCalibrationFitProvenance:
    """Point-estimate and uncertainty-integration provenance.

    Pair positions refer to the caller-supplied paired arrays before finite
    filtering. The selected estimator is deterministic from error-axis
    participation. A scale-dependent available covariance must share the same
    full-objective optimum and final-inlier set as the reported coefficients.
    If that estimator fails after activation, a fallback point estimate must
    be represented explicitly rather than paired with its covariance.
    """

    schema_version: str
    selected_uncertainty_estimator: (
        InstrumentChannelCalibrationUncertaintyEstimator | str
    )
    integration_status: (
        InstrumentChannelCalibrationUncertaintyIntegrationStatus | str
    )
    reference_error_supplied: bool
    channel_error_supplied: bool
    n_input_pairs: int
    finite_pair_indices: tuple[int, ...]
    final_inlier_indices: tuple[int, ...]
    point_estimate_source: str
    point_estimate_objective: str
    point_estimate_matches_uncertainty_objective: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_FIT_PROVENANCE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported instrument-channel calibration fit provenance "
                f"schema version: {self.schema_version!r}."
            )

        estimator = self.selected_uncertainty_estimator
        if not isinstance(
            estimator,
            InstrumentChannelCalibrationUncertaintyEstimator,
        ):
            try:
                estimator = (
                    InstrumentChannelCalibrationUncertaintyEstimator(
                        str(estimator)
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "Unsupported instrument-channel calibration uncertainty "
                    f"estimator: {self.selected_uncertainty_estimator!r}."
                ) from exc

        integration_status = self.integration_status
        if not isinstance(
            integration_status,
            InstrumentChannelCalibrationUncertaintyIntegrationStatus,
        ):
            try:
                integration_status = (
                    InstrumentChannelCalibrationUncertaintyIntegrationStatus(
                        str(integration_status)
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "Unsupported instrument-channel calibration uncertainty "
                    f"integration status: {self.integration_status!r}."
                ) from exc

        normalized_booleans = {}
        for name in (
            "reference_error_supplied",
            "channel_error_supplied",
            "point_estimate_matches_uncertainty_objective",
        ):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean.")
            normalized_booleans[name] = bool(value)

        if isinstance(self.n_input_pairs, (bool, np.bool_)) or not isinstance(
            self.n_input_pairs,
            (int, np.integer),
        ):
            raise TypeError("n_input_pairs must be an integer.")
        n_input_pairs = int(self.n_input_pairs)
        if n_input_pairs < 3:
            raise ValueError("n_input_pairs must be at least 3.")

        finite_pair_indices = InstrumentChannelPairing._normalize_indices(
            self.finite_pair_indices,
            name="finite_pair_indices",
        )
        final_inlier_indices = InstrumentChannelPairing._normalize_indices(
            self.final_inlier_indices,
            name="final_inlier_indices",
        )

        for name, indices in (
            ("finite_pair_indices", finite_pair_indices),
            ("final_inlier_indices", final_inlier_indices),
        ):
            if len(indices) < 3:
                raise ValueError(f"{name} must contain at least 3 positions.")
            if tuple(sorted(set(indices))) != indices:
                raise ValueError(
                    f"{name} must contain unique positions in increasing "
                    "order."
                )
            if indices[-1] >= n_input_pairs:
                raise ValueError(
                    f"{name} positions must be smaller than n_input_pairs."
                )

        if not set(final_inlier_indices).issubset(finite_pair_indices):
            raise ValueError(
                "final_inlier_indices must be a subset of "
                "finite_pair_indices."
            )

        point_estimate_source = (
            InstrumentChannelCalibrationCoefficientUncertainty
            ._normalize_required_text(
                self.point_estimate_source,
                name="point_estimate_source",
            )
        )
        point_estimate_objective = (
            InstrumentChannelCalibrationCoefficientUncertainty
            ._normalize_required_text(
                self.point_estimate_objective,
                name="point_estimate_objective",
            )
        )
        reason = (
            InstrumentChannelCalibrationCoefficientUncertainty
            ._normalize_optional_text(
                self.reason,
                name="reason",
            )
        )

        selected = select_instrument_channel_calibration_uncertainty_estimator(
            reference_error_supplied=normalized_booleans[
                "reference_error_supplied"
            ],
            channel_error_supplied=normalized_booleans[
                "channel_error_supplied"
            ],
        )
        if estimator is not selected:
            raise ValueError(
                "selected_uncertainty_estimator is inconsistent with the "
                "supplied error axes."
            )

        matches_objective = normalized_booleans[
            "point_estimate_matches_uncertainty_objective"
        ]
        if integration_status is (
            InstrumentChannelCalibrationUncertaintyIntegrationStatus.ACTIVE
        ):
            if reason is not None:
                raise ValueError(
                    "Active uncertainty integration must not carry a reason."
                )
            if not matches_objective:
                raise ValueError(
                    "Active uncertainty integration requires the point "
                    "estimate to match the uncertainty objective."
                )
        else:
            if reason is None:
                raise ValueError(
                    "Inactive or fallback uncertainty integration requires "
                    "an explicit reason."
                )
            if matches_objective:
                raise ValueError(
                    "Inactive or fallback uncertainty integration cannot "
                    "claim a shared point-estimate objective."
                )

        fixed_estimator = (
            InstrumentChannelCalibrationUncertaintyEstimator
            .FIXED_WEIGHT_NORMAL_MATRIX
        )
        if estimator is fixed_estimator:
            if integration_status is not (
                InstrumentChannelCalibrationUncertaintyIntegrationStatus
                .ACTIVE
            ):
                raise ValueError(
                    "The implemented fixed-weight uncertainty path must be "
                    "active."
                )
            expected_source = "pgmuvi_affine_fit_final_inliers"
            expected_objective = "fixed_weight_least_squares"
        elif integration_status is (
            InstrumentChannelCalibrationUncertaintyIntegrationStatus.ACTIVE
        ):
            expected_source = (
                "pgmuvi_scale_dependent_full_objective_final_inliers"
            )
            expected_objective = (
                "gaussian_negative_log_likelihood_"
                "scale_dependent_effective_variance"
            )
        else:
            expected_source = (
                "pgmuvi_iterative_mad_clipped_affine_fallback_"
                "final_inliers"
            )
            expected_objective = (
                "iterative_scale_frozen_weighted_least_squares"
            )

        if point_estimate_source != expected_source:
            raise ValueError(
                "point_estimate_source is inconsistent with the selected "
                "uncertainty integration path."
            )
        if point_estimate_objective != expected_objective:
            raise ValueError(
                "point_estimate_objective is inconsistent with the selected "
                "uncertainty integration path."
            )

        object.__setattr__(
            self,
            "selected_uncertainty_estimator",
            estimator,
        )
        object.__setattr__(
            self,
            "integration_status",
            integration_status,
        )
        for name, value in normalized_booleans.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "n_input_pairs", n_input_pairs)
        object.__setattr__(
            self,
            "finite_pair_indices",
            finite_pair_indices,
        )
        object.__setattr__(
            self,
            "final_inlier_indices",
            final_inlier_indices,
        )
        object.__setattr__(
            self,
            "point_estimate_source",
            point_estimate_source,
        )
        object.__setattr__(
            self,
            "point_estimate_objective",
            point_estimate_objective,
        )
        object.__setattr__(self, "reason", reason)

    @property
    def n_finite_pairs(self) -> int:
        """Return the number of pairs retained by finite-value filtering."""

        return len(self.finite_pair_indices)

    @property
    def n_final_inliers(self) -> int:
        """Return the caller-selected final-inlier count."""

        return len(self.final_inlier_indices)

    def to_dict(self) -> dict[str, Any]:
        """Return strict JSON-safe fit and integration provenance."""

        return {
            "schema_version": self.schema_version,
            "selected_uncertainty_estimator": (
                self.selected_uncertainty_estimator.value
            ),
            "integration_status": self.integration_status.value,
            "reference_error_supplied": self.reference_error_supplied,
            "channel_error_supplied": self.channel_error_supplied,
            "error_axes": [
                name
                for name, supplied in (
                    ("reference", self.reference_error_supplied),
                    ("observational_channel", self.channel_error_supplied),
                )
                if supplied
            ],
            "coefficient_order": ["offset", "scale"],
            "n_input_pairs": self.n_input_pairs,
            "n_finite_pairs": self.n_finite_pairs,
            "n_final_inliers": self.n_final_inliers,
            "finite_pair_indices": list(self.finite_pair_indices),
            "final_inlier_indices": list(self.final_inlier_indices),
            "point_estimate_source": self.point_estimate_source,
            "point_estimate_objective": self.point_estimate_objective,
            "point_estimate_matches_uncertainty_objective": (
                self.point_estimate_matches_uncertainty_objective
            ),
            "uncertainty_conditioned_on_final_inlier_set": True,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationCoefficientUncertainty:
    """Uncertainty provenance for affine offset and scale coefficients.

    Available uncertainty uses the fixed coefficient order ``offset, scale``
    and stores the complete symmetric 2-by-2 covariance matrix. Standard
    errors are derived from its diagonal rather than stored independently.
    Unavailable uncertainty instead requires an explicit reason and cannot
    carry placeholder numerical values.
    """

    schema_version: str
    status: InstrumentChannelCalibrationUncertaintyStatus | str
    uncertainty_source: str
    coefficient_covariance: (
        tuple[tuple[float, float], tuple[float, float]] | None
    )
    estimation_method: str | None = None
    degrees_of_freedom: int | None = None
    residual_variance: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported instrument-channel calibration uncertainty "
                f"schema version: {self.schema_version!r}."
            )

        status = self.status
        if not isinstance(
            status,
            InstrumentChannelCalibrationUncertaintyStatus,
        ):
            try:
                status = InstrumentChannelCalibrationUncertaintyStatus(
                    str(status)
                )
            except ValueError as exc:
                raise ValueError(
                    "Unsupported instrument-channel calibration uncertainty "
                    f"status: {self.status!r}."
                ) from exc

        uncertainty_source = self._normalize_required_text(
            self.uncertainty_source,
            name="uncertainty_source",
        )
        estimation_method = self._normalize_optional_text(
            self.estimation_method,
            name="estimation_method",
        )
        reason = self._normalize_optional_text(
            self.reason,
            name="reason",
        )

        covariance = self._normalize_covariance(
            self.coefficient_covariance
        )
        degrees_of_freedom = self._normalize_degrees_of_freedom(
            self.degrees_of_freedom
        )
        residual_variance = self._normalize_residual_variance(
            self.residual_variance
        )

        if (
            status
            is InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE
        ):
            if covariance is None:
                raise ValueError(
                    "Available coefficient uncertainty requires a complete "
                    "coefficient_covariance matrix."
                )
            if estimation_method is None:
                raise ValueError(
                    "Available coefficient uncertainty requires an explicit "
                    "estimation_method."
                )
            if reason is not None:
                raise ValueError(
                    "Available coefficient uncertainty must not carry an "
                    "unavailable reason."
                )
        else:
            if reason is None:
                raise ValueError(
                    "Unavailable coefficient uncertainty requires a reason."
                )
            if any(
                value is not None
                for value in (
                    covariance,
                    estimation_method,
                    degrees_of_freedom,
                    residual_variance,
                )
            ):
                raise ValueError(
                    "Unavailable coefficient uncertainty cannot contain "
                    "covariance or estimation metadata."
                )

        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "uncertainty_source",
            uncertainty_source,
        )
        object.__setattr__(
            self,
            "coefficient_covariance",
            covariance,
        )
        object.__setattr__(
            self,
            "estimation_method",
            estimation_method,
        )
        object.__setattr__(
            self,
            "degrees_of_freedom",
            degrees_of_freedom,
        )
        object.__setattr__(
            self,
            "residual_variance",
            residual_variance,
        )
        object.__setattr__(self, "reason", reason)

    @staticmethod
    def _normalize_required_text(value: Any, *, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{name} must be non-empty.")
        return normalized

    @classmethod
    def _normalize_optional_text(
        cls,
        value: Any,
        *,
        name: str,
    ) -> str | None:
        if value is None:
            return None
        return cls._normalize_required_text(value, name=name)

    @staticmethod
    def _normalize_covariance(
        value: Any,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if value is None:
            return None

        def contains_boolean(item: Any) -> bool:
            if isinstance(item, (bool, np.bool_)):
                return True
            if isinstance(item, np.ndarray):
                return any(
                    contains_boolean(element)
                    for element in item.flat
                )
            if isinstance(item, (list, tuple)):
                return any(
                    contains_boolean(element)
                    for element in item
                )
            return False

        if contains_boolean(value):
            raise TypeError(
                "coefficient_covariance must contain numeric values, "
                "not booleans."
            )

        covariance = np.asarray(value, dtype=float)
        if covariance.shape != (2, 2):
            raise ValueError(
                "coefficient_covariance must have shape (2, 2)."
            )
        if np.any(~np.isfinite(covariance)):
            raise ValueError(
                "coefficient_covariance must contain finite values."
            )
        if np.any(np.diag(covariance) < 0.0):
            raise ValueError(
                "coefficient_covariance diagonal variances must be "
                "non-negative."
            )

        scale = max(1.0, float(np.max(np.abs(covariance))))
        tolerance = 64.0 * np.finfo(float).eps * scale
        if not np.allclose(
            covariance,
            covariance.T,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "coefficient_covariance must be symmetric."
            )

        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalue_scale = max(
            1.0,
            float(np.max(np.abs(covariance))),
            float(np.max(np.abs(eigenvalues))),
        )
        eigenvalue_tolerance = (
            64.0 * np.finfo(float).eps * eigenvalue_scale
        )
        if float(np.min(eigenvalues)) < -eigenvalue_tolerance:
            raise ValueError(
                "coefficient_covariance must be positive semidefinite."
            )

        return (
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        )

    @staticmethod
    def _normalize_degrees_of_freedom(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(
                "degrees_of_freedom must be an integer when supplied."
            )
        normalized = int(value)
        if normalized < 1:
            raise ValueError(
                "degrees_of_freedom must be at least 1 when supplied."
            )
        return normalized

    @staticmethod
    def _normalize_residual_variance(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(
                "residual_variance must be numeric, not boolean."
            )
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError(
                "residual_variance must be finite and non-negative when "
                "supplied."
            )
        return normalized

    @property
    def offset_standard_error(self) -> float | None:
        """Return the derived offset standard error when available."""

        if self.coefficient_covariance is None:
            return None
        return math.sqrt(self.coefficient_covariance[0][0])

    @property
    def scale_standard_error(self) -> float | None:
        """Return the derived scale standard error when available."""

        if self.coefficient_covariance is None:
            return None
        return math.sqrt(self.coefficient_covariance[1][1])

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe uncertainty representation."""

        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "uncertainty_source": self.uncertainty_source,
            "coefficient_order": ["offset", "scale"],
            "coefficient_covariance": (
                None
                if self.coefficient_covariance is None
                else [
                    list(row)
                    for row in self.coefficient_covariance
                ]
            ),
            "offset_standard_error": self.offset_standard_error,
            "scale_standard_error": self.scale_standard_error,
            "estimation_method": self.estimation_method,
            "degrees_of_freedom": self.degrees_of_freedom,
            "residual_variance": self.residual_variance,
            "reason": self.reason,
            "predictive_uncertainty_propagated": False,
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationScaleDependentUncertaintyEstimate:
    """Result of full-objective channel-axis error estimation.

    The estimator is conditioned on a caller-selected final inlier set and
    minimizes the Gaussian negative log likelihood

    ``0.5 * sum(log(v_i) + residual_i**2 / v_i)``,

    where ``v_i = reference_error_i**2 + scale**2 * channel_error_i**2``.
    Coefficient order is fixed as ``offset, scale``. Available covariance is
    the inverse observed Hessian of that full objective at a converged optimum
    in the strictly positive-scale domain.
    """

    schema_version: str
    status: InstrumentChannelCalibrationUncertaintyStatus | str
    uncertainty_source: str
    n_inliers: int
    coefficient_covariance: (
        tuple[tuple[float, float], tuple[float, float]] | None
    ) = None
    offset: float | None = None
    scale: float | None = None
    objective_value: float | None = None
    optimizer: str | None = None
    optimizer_converged: bool | None = None
    gradient_method: str | None = None
    gradient_norm: float | None = None
    hessian_method: str | None = None
    hessian_eigenvalues: tuple[float, float] | None = None
    reason: str | None = None
    conditioned_on_final_inlier_set: bool = True
    positive_scale_domain_enforced: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_SCALE_DEPENDENT_UNCERTAINTY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported scale-dependent calibration uncertainty schema "
                f"version: {self.schema_version!r}."
            )

        status = self.status
        if not isinstance(
            status,
            InstrumentChannelCalibrationUncertaintyStatus,
        ):
            try:
                status = InstrumentChannelCalibrationUncertaintyStatus(
                    str(status)
                )
            except ValueError as exc:
                raise ValueError(
                    "Unsupported scale-dependent calibration uncertainty "
                    f"status: {self.status!r}."
                ) from exc

        uncertainty_source = (
            InstrumentChannelCalibrationCoefficientUncertainty
            ._normalize_required_text(
                self.uncertainty_source,
                name="uncertainty_source",
            )
        )
        optimizer = (
            InstrumentChannelCalibrationCoefficientUncertainty
            ._normalize_optional_text(
                self.optimizer,
                name="optimizer",
            )
        )
        gradient_method = (
            InstrumentChannelCalibrationCoefficientUncertainty
            ._normalize_optional_text(
                self.gradient_method,
                name="gradient_method",
            )
        )
        hessian_method = (
            InstrumentChannelCalibrationCoefficientUncertainty
            ._normalize_optional_text(
                self.hessian_method,
                name="hessian_method",
            )
        )
        reason = (
            InstrumentChannelCalibrationCoefficientUncertainty
            ._normalize_optional_text(
                self.reason,
                name="reason",
            )
        )

        if isinstance(self.n_inliers, (bool, np.bool_)) or not isinstance(
            self.n_inliers,
            (int, np.integer),
        ):
            raise TypeError("n_inliers must be an integer.")
        n_inliers = int(self.n_inliers)
        if n_inliers < 3:
            raise ValueError("n_inliers must be at least 3.")

        for name in (
            "conditioned_on_final_inlier_set",
            "positive_scale_domain_enforced",
        ):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean.")
            if not bool(value):
                raise ValueError(f"{name} must be true for this contract.")

        optimizer_converged = self.optimizer_converged
        if optimizer_converged is not None and not isinstance(
            optimizer_converged,
            (bool, np.bool_),
        ):
            raise TypeError("optimizer_converged must be boolean when supplied.")
        if optimizer_converged is not None:
            optimizer_converged = bool(optimizer_converged)

        covariance = (
            InstrumentChannelCalibrationCoefficientUncertainty
            ._normalize_covariance(self.coefficient_covariance)
        )

        def finite_optional(
            value: Any,
            *,
            name: str,
            non_negative: bool = False,
            strictly_positive: bool = False,
        ) -> float | None:
            if value is None:
                return None
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be numeric, not boolean.")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite when supplied.")
            if non_negative and normalized < 0.0:
                raise ValueError(
                    f"{name} must be non-negative when supplied."
                )
            if strictly_positive and normalized <= 0.0:
                raise ValueError(
                    f"{name} must be strictly positive when supplied."
                )
            return normalized

        offset = finite_optional(self.offset, name="offset")
        scale = finite_optional(
            self.scale,
            name="scale",
            strictly_positive=True,
        )
        objective_value = finite_optional(
            self.objective_value,
            name="objective_value",
        )
        gradient_norm = finite_optional(
            self.gradient_norm,
            name="gradient_norm",
            non_negative=True,
        )

        hessian_eigenvalues = self._normalize_hessian_eigenvalues(
            self.hessian_eigenvalues
        )

        if (
            status
            is InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE
        ):
            required = {
                "coefficient_covariance": covariance,
                "offset": offset,
                "scale": scale,
                "objective_value": objective_value,
                "optimizer": optimizer,
                "optimizer_converged": optimizer_converged,
                "gradient_method": gradient_method,
                "gradient_norm": gradient_norm,
                "hessian_method": hessian_method,
                "hessian_eigenvalues": hessian_eigenvalues,
            }
            missing = [
                name for name, value in required.items() if value is None
            ]
            if missing:
                raise ValueError(
                    "Available scale-dependent uncertainty requires complete "
                    "estimation provenance; missing "
                    + ", ".join(missing)
                    + "."
                )
            if optimizer_converged is not True:
                raise ValueError(
                    "Available scale-dependent uncertainty requires a "
                    "converged optimizer."
                )
            if reason is not None:
                raise ValueError(
                    "Available scale-dependent uncertainty must not carry "
                    "an unavailable reason."
                )
        else:
            if reason is None:
                raise ValueError(
                    "Unavailable scale-dependent uncertainty requires a "
                    "reason."
                )
            if covariance is not None:
                raise ValueError(
                    "Unavailable scale-dependent uncertainty cannot carry "
                    "coefficient covariance."
                )
            if any(
                value is not None
                for value in (
                    offset,
                    scale,
                    objective_value,
                    gradient_norm,
                    hessian_eigenvalues,
                )
            ):
                raise ValueError(
                    "Unavailable scale-dependent uncertainty cannot carry "
                    "numerical estimation results."
                )
            if optimizer_converged is True:
                raise ValueError(
                    "Unavailable scale-dependent uncertainty cannot report "
                    "a converged optimizer."
                )

        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "uncertainty_source",
            uncertainty_source,
        )
        object.__setattr__(self, "n_inliers", n_inliers)
        object.__setattr__(
            self,
            "coefficient_covariance",
            covariance,
        )
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(
            self,
            "objective_value",
            objective_value,
        )
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(
            self,
            "optimizer_converged",
            optimizer_converged,
        )
        object.__setattr__(
            self,
            "gradient_method",
            gradient_method,
        )
        object.__setattr__(
            self,
            "gradient_norm",
            gradient_norm,
        )
        object.__setattr__(
            self,
            "hessian_method",
            hessian_method,
        )
        object.__setattr__(
            self,
            "hessian_eigenvalues",
            hessian_eigenvalues,
        )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "conditioned_on_final_inlier_set",
            True,
        )
        object.__setattr__(
            self,
            "positive_scale_domain_enforced",
            True,
        )

    @staticmethod
    def _normalize_hessian_eigenvalues(
        value: Any,
    ) -> tuple[float, float] | None:
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            raw = value.tolist()
        else:
            raw = value

        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(
                "hessian_eigenvalues must contain exactly two values."
            )

        normalized = []
        for item in raw:
            if isinstance(item, (bool, np.bool_)):
                raise TypeError(
                    "hessian_eigenvalues must be numeric, not boolean."
                )
            numeric = float(item)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(
                    "hessian_eigenvalues must be finite and strictly "
                    "positive."
                )
            normalized.append(numeric)

        return (normalized[0], normalized[1])

    @property
    def offset_standard_error(self) -> float | None:
        """Return the derived offset standard error when available."""

        if self.coefficient_covariance is None:
            return None
        return math.sqrt(self.coefficient_covariance[0][0])

    @property
    def scale_standard_error(self) -> float | None:
        """Return the derived scale standard error when available."""

        if self.coefficient_covariance is None:
            return None
        return math.sqrt(self.coefficient_covariance[1][1])

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-safe estimator contract."""

        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "uncertainty_source": self.uncertainty_source,
            "coefficient_order": ["offset", "scale"],
            "objective": (
                "gaussian_negative_log_likelihood_"
                "scale_dependent_effective_variance"
            ),
            "objective_equation": (
                "0.5 * sum(log(v_i) + residual_i**2 / v_i)"
            ),
            "effective_variance_equation": (
                "reference_error_i**2 + "
                "scale**2 * channel_error_i**2"
            ),
            "covariance_estimator": (
                "inverse_observed_hessian_at_converged_optimum"
            ),
            "conditioned_on_final_inlier_set": (
                self.conditioned_on_final_inlier_set
            ),
            "positive_scale_domain_enforced": (
                self.positive_scale_domain_enforced
            ),
            "n_inliers": self.n_inliers,
            "offset": self.offset,
            "scale": self.scale,
            "coefficient_covariance": (
                None
                if self.coefficient_covariance is None
                else [
                    list(row)
                    for row in self.coefficient_covariance
                ]
            ),
            "offset_standard_error": self.offset_standard_error,
            "scale_standard_error": self.scale_standard_error,
            "objective_value": self.objective_value,
            "optimizer": self.optimizer,
            "optimizer_converged": self.optimizer_converged,
            "gradient_method": self.gradient_method,
            "gradient_norm": self.gradient_norm,
            "hessian_method": self.hessian_method,
            "hessian_eigenvalues": (
                None
                if self.hessian_eigenvalues is None
                else list(self.hessian_eigenvalues)
            ),
            "reason": self.reason,
            "implemented": True,
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationPredictiveUncertainty:
    """Immutable contract for affine predictive-uncertainty propagation.

    Component vectors use flattened row-major order. ``input_shape`` preserves
    the calibrated flux shape. The fixed coefficient Jacobian is ``[1, x]``
    in coefficient order ``offset, scale``. This record validates supplied
    results but does not calculate them.
    """

    schema_version: str
    status: InstrumentChannelCalibrationPredictiveUncertaintyStatus | str
    covariance_mode: InstrumentChannelCalibrationPredictiveCovarianceMode | str
    input_shape: tuple[int, ...]
    input_measurement_uncertainty_supplied: bool
    coefficient_uncertainty_status: (
        InstrumentChannelCalibrationUncertaintyStatus | str
    )
    coefficient_uncertainty_source: str
    measurement_variance: tuple[float, ...] | None
    offset_variance: tuple[float, ...] | None
    scale_variance: tuple[float, ...] | None
    offset_scale_covariance_term: tuple[float, ...] | None
    predictive_covariance: tuple[tuple[float, ...], ...] | None
    input_coefficient_independence_assumed: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_UNCERTAINTY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported instrument-channel calibration predictive "
                f"uncertainty schema version: {self.schema_version!r}."
            )

        status = self._coerce_enum(
            self.status,
            InstrumentChannelCalibrationPredictiveUncertaintyStatus,
            "predictive uncertainty status",
        )
        covariance_mode = self._coerce_enum(
            self.covariance_mode,
            InstrumentChannelCalibrationPredictiveCovarianceMode,
            "predictive covariance mode",
        )
        coefficient_status = self._coerce_enum(
            self.coefficient_uncertainty_status,
            InstrumentChannelCalibrationUncertaintyStatus,
            "coefficient uncertainty status",
        )
        input_shape = self._normalize_shape(self.input_shape)
        size = math.prod(input_shape) if input_shape else 1
        measurement_supplied = self._normalize_boolean(
            self.input_measurement_uncertainty_supplied,
            name="input_measurement_uncertainty_supplied",
        )
        independence_assumed = self._normalize_boolean(
            self.input_coefficient_independence_assumed,
            name="input_coefficient_independence_assumed",
        )
        if not independence_assumed:
            raise ValueError(
                "Input measurement errors must be independent of fitted "
                "calibration coefficients."
            )

        source = self._normalize_text(
            self.coefficient_uncertainty_source,
            name="coefficient_uncertainty_source",
        )
        reason = (
            None
            if self.reason is None
            else self._normalize_text(self.reason, name="reason")
        )
        component_names = (
            "measurement_variance",
            "offset_variance",
            "scale_variance",
            "offset_scale_covariance_term",
        )

        if status is (
            InstrumentChannelCalibrationPredictiveUncertaintyStatus.AVAILABLE
        ):
            if coefficient_status is not (
                InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE
            ):
                raise ValueError(
                    "Available predictive uncertainty requires available "
                    "coefficient uncertainty."
                )
            if reason is not None:
                raise ValueError(
                    "Available predictive uncertainty must not carry a reason."
                )
            components = {
                name: self._normalize_vector(
                    getattr(self, name),
                    name=name,
                    size=size,
                    non_negative=(
                        name != "offset_scale_covariance_term"
                    ),
                )
                for name in component_names
            }
            if not measurement_supplied and any(
                components["measurement_variance"]
            ):
                raise ValueError(
                    "measurement_variance must be zero when input "
                    "measurement uncertainty was not supplied."
                )
            predictive_variance = tuple(
                measurement + offset + scale + cross
                for measurement, offset, scale, cross in zip(
                    components["measurement_variance"],
                    components["offset_variance"],
                    components["scale_variance"],
                    components["offset_scale_covariance_term"],
                    strict=True,
                )
            )
            if any(value < 0.0 for value in predictive_variance):
                raise ValueError(
                    "Derived predictive variance must be non-negative."
                )
            predictive_covariance = self._normalize_covariance(
                self.predictive_covariance,
                size=size,
            )
            if covariance_mode is (
                InstrumentChannelCalibrationPredictiveCovarianceMode.FULL_COVARIANCE
            ):
                if predictive_covariance is None:
                    raise ValueError(
                        "full_covariance mode requires predictive_covariance."
                    )
                if not np.allclose(
                    np.diag(predictive_covariance),
                    predictive_variance,
                    rtol=0.0,
                    atol=self._tolerance(predictive_variance),
                ):
                    raise ValueError(
                        "predictive_covariance diagonal must equal the "
                        "derived predictive variance."
                    )
            elif predictive_covariance is not None:
                raise ValueError(
                    "marginal_variance mode must not carry "
                    "predictive_covariance."
                )
        else:
            if coefficient_status is not (
                InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE
            ):
                raise ValueError(
                    "Unavailable predictive uncertainty requires unavailable "
                    "coefficient uncertainty."
                )
            if reason is None:
                raise ValueError(
                    "Unavailable predictive uncertainty requires a reason."
                )
            if any(getattr(self, name) is not None for name in component_names):
                raise ValueError(
                    "Unavailable predictive uncertainty cannot carry "
                    "variance components."
                )
            if self.predictive_covariance is not None:
                raise ValueError(
                    "Unavailable predictive uncertainty cannot carry a "
                    "predictive covariance."
                )
            components = {name: None for name in component_names}
            predictive_variance = None
            predictive_covariance = None

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "covariance_mode", covariance_mode)
        object.__setattr__(
            self,
            "coefficient_uncertainty_status",
            coefficient_status,
        )
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(
            self,
            "input_measurement_uncertainty_supplied",
            measurement_supplied,
        )
        object.__setattr__(
            self,
            "input_coefficient_independence_assumed",
            independence_assumed,
        )
        object.__setattr__(self, "coefficient_uncertainty_source", source)
        object.__setattr__(self, "reason", reason)
        for name, value in components.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "predictive_covariance",
            predictive_covariance,
        )
    @staticmethod
    def _coerce_enum(value: Any, enum_type: type[_StringEnum], name: str):
        if isinstance(value, enum_type):
            return value
        try:
            return enum_type(str(value))
        except ValueError as exc:
            raise ValueError(f"Unsupported {name}: {value!r}.") from exc

    @staticmethod
    def _normalize_shape(value: Any) -> tuple[int, ...]:
        if not isinstance(value, (tuple, list)):
            raise TypeError("input_shape must be a tuple or list of integers.")
        shape = []
        for dimension in value:
            if isinstance(dimension, (bool, np.bool_)) or not isinstance(
                dimension,
                (int, np.integer),
            ):
                raise TypeError(
                    "input_shape dimensions must be integers, not booleans."
                )
            if int(dimension) < 1:
                raise ValueError(
                    "input_shape dimensions must be strictly positive."
                )
            shape.append(int(dimension))
        return tuple(shape)

    @staticmethod
    def _normalize_boolean(value: Any, *, name: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean.")
        return bool(value)

    @staticmethod
    def _normalize_text(value: Any, *, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{name} must be non-empty.")
        return normalized

    @staticmethod
    def _contains_boolean(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return True
        if isinstance(value, np.ndarray):
            return any(
                InstrumentChannelCalibrationPredictiveUncertainty._contains_boolean(
                    item
                )
                for item in value.flat
            )
        if isinstance(value, (list, tuple)):
            return any(
                InstrumentChannelCalibrationPredictiveUncertainty._contains_boolean(
                    item
                )
                for item in value
            )
        return False

    @classmethod
    def _normalize_vector(
        cls,
        value: Any,
        *,
        name: str,
        size: int,
        non_negative: bool,
    ) -> tuple[float, ...]:
        if value is None:
            raise ValueError(
                f"Available predictive uncertainty requires {name}."
            )
        if cls._contains_boolean(value):
            raise TypeError(
                f"{name} must contain numeric values, not booleans."
            )
        array = np.asarray(value, dtype=float)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},).")
        if np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values.")
        if non_negative and np.any(array < 0.0):
            raise ValueError(f"{name} must be non-negative.")
        return tuple(float(item) for item in array)

    @classmethod
    def _normalize_covariance(
        cls,
        value: Any,
        *,
        size: int,
    ) -> tuple[tuple[float, ...], ...] | None:
        if value is None:
            return None
        if cls._contains_boolean(value):
            raise TypeError(
                "predictive_covariance must contain numeric values, not "
                "booleans."
            )
        covariance = np.asarray(value, dtype=float)
        if covariance.shape != (size, size):
            raise ValueError(
                f"predictive_covariance must have shape ({size}, {size})."
            )
        if np.any(~np.isfinite(covariance)):
            raise ValueError(
                "predictive_covariance must contain finite values."
            )
        if not np.allclose(
            covariance,
            covariance.T,
            rtol=0.0,
            atol=cls._tolerance(covariance),
        ):
            raise ValueError("predictive_covariance must be symmetric.")
        covariance = 0.5 * (covariance + covariance.T)
        if float(np.min(np.linalg.eigvalsh(covariance))) < -cls._tolerance(
            covariance
        ):
            raise ValueError(
                "predictive_covariance must be positive semidefinite."
            )
        return tuple(
            tuple(float(item) for item in row)
            for row in covariance
        )

    @staticmethod
    def _tolerance(value: Any) -> float:
        array = np.asarray(value, dtype=float)
        scale = max(1.0, float(np.max(np.abs(array))))
        return 256.0 * np.finfo(float).eps * scale

    @property
    def coefficient_variance(self) -> tuple[float, ...] | None:
        """Return marginal variance from shared fitted coefficients."""

        if self.offset_variance is None:
            return None
        return tuple(
            offset + scale + cross
            for offset, scale, cross in zip(
                self.offset_variance,
                self.scale_variance,
                self.offset_scale_covariance_term,
                strict=True,
            )
        )

    @property
    def predictive_variance(self) -> tuple[float, ...] | None:
        """Return total marginal predictive variance when available."""

        if self.measurement_variance is None:
            return None
        return tuple(
            measurement + coefficient
            for measurement, coefficient in zip(
                self.measurement_variance,
                self.coefficient_variance,
                strict=True,
            )
        )

    @property
    def predictive_standard_deviation(self) -> tuple[float, ...] | None:
        """Return total marginal predictive standard deviation."""

        if self.predictive_variance is None:
            return None
        return tuple(math.sqrt(value) for value in self.predictive_variance)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe representation."""

        def vector(value: tuple[float, ...] | None) -> list[float] | None:
            return None if value is None else list(value)

        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "covariance_mode": self.covariance_mode.value,
            "input_shape": list(self.input_shape),
            "input_measurement_uncertainty_supplied": (
                self.input_measurement_uncertainty_supplied
            ),
            "coefficient_uncertainty_status": (
                self.coefficient_uncertainty_status.value
            ),
            "coefficient_uncertainty_source": (
                self.coefficient_uncertainty_source
            ),
            "coefficient_order": ["offset", "scale"],
            "coefficient_jacobian": "[1, x]",
            "measurement_variance_equation": (
                "scale**2 * flux_error**2"
            ),
            "coefficient_covariance_equation": (
                "[1, x_i] C [1, x_j]^T"
            ),
            "measurement_variance": vector(self.measurement_variance),
            "offset_variance": vector(self.offset_variance),
            "scale_variance": vector(self.scale_variance),
            "offset_scale_covariance_term": vector(
                self.offset_scale_covariance_term
            ),
            "coefficient_variance": vector(self.coefficient_variance),
            "predictive_variance": vector(self.predictive_variance),
            "predictive_standard_deviation": vector(
                self.predictive_standard_deviation
            ),
            "predictive_covariance": (
                None
                if self.predictive_covariance is None
                else [list(row) for row in self.predictive_covariance]
            ),
            "input_coefficient_independence_assumed": (
                self.input_coefficient_independence_assumed
            ),
            "shared_coefficient_correlation_represented": (
                self.status
                is InstrumentChannelCalibrationPredictiveUncertaintyStatus.AVAILABLE
                and self.covariance_mode
                is InstrumentChannelCalibrationPredictiveCovarianceMode.FULL_COVARIANCE
            ),
            "predictive_uncertainty_propagated": (
                self.status
                is InstrumentChannelCalibrationPredictiveUncertaintyStatus.AVAILABLE
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationPredictiveUncertaintyRequest:
    """Explicit opt-in request for dataset predictive propagation.

    Absence of this object means propagation was not requested.  Supplying a
    request selects either marginal variance or per-observational-channel full
    covariance.  Full covariance is never represented as one global dense
    dataset matrix.
    """

    covariance_mode: InstrumentChannelCalibrationPredictiveCovarianceMode | str
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_ORCHESTRATION_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_ORCHESTRATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported instrument-channel calibration predictive "
                f"orchestration schema version: {self.schema_version!r}."
            )
        if isinstance(self.covariance_mode, (bool, np.bool_)):
            raise TypeError(
                "covariance_mode must be a predictive covariance mode, "
                "not boolean."
            )
        mode = self.covariance_mode
        if not isinstance(
            mode,
            InstrumentChannelCalibrationPredictiveCovarianceMode,
        ):
            try:
                mode = InstrumentChannelCalibrationPredictiveCovarianceMode(
                    str(mode)
                )
            except ValueError as exc:
                raise ValueError(
                    "Unsupported predictive covariance mode: "
                    f"{self.covariance_mode!r}."
                ) from exc
        object.__setattr__(self, "covariance_mode", mode)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe request representation."""

        return {
            "schema_version": self.schema_version,
            "requested": True,
            "covariance_mode": self.covariance_mode.value,
            "full_covariance_scope": (
                "per_observational_channel"
                if self.covariance_mode
                is InstrumentChannelCalibrationPredictiveCovarianceMode
                .FULL_COVARIANCE
                else None
            ),
            "global_dense_covariance_requested": False,
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationPredictiveUncertaintyChannelResult:
    """Predictive-propagation disposition for one non-reference channel."""

    physical_wavelength: float
    reference_channel: str
    channel: str
    disposition: (
        InstrumentChannelCalibrationPredictiveUncertaintyDisposition | str
    )
    source_row_indices: tuple[int, ...]
    predictive_uncertainty: (
        InstrumentChannelCalibrationPredictiveUncertainty | None
    ) = None
    reason: str | None = None

    def __post_init__(self) -> None:
        wavelength = float(self.physical_wavelength)
        if not math.isfinite(wavelength):
            raise ValueError("physical_wavelength must be finite.")
        reference_channel = _normalize_calibration_channel(
            self.reference_channel,
            name="reference_channel",
        )
        channel = _normalize_calibration_channel(
            self.channel,
            name="channel",
        )
        if channel == reference_channel:
            raise ValueError(
                "channel and reference_channel must identify different "
                "observational channels."
            )
        if isinstance(self.disposition, (bool, np.bool_)):
            raise TypeError(
                "disposition must be a predictive uncertainty disposition, "
                "not boolean."
            )
        disposition = self.disposition
        if not isinstance(
            disposition,
            InstrumentChannelCalibrationPredictiveUncertaintyDisposition,
        ):
            try:
                disposition = (
                    InstrumentChannelCalibrationPredictiveUncertaintyDisposition(
                        str(disposition)
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "Unsupported predictive uncertainty disposition: "
                    f"{self.disposition!r}."
                ) from exc
        indices = InstrumentChannelPairing._normalize_indices(
            self.source_row_indices,
            name="source_row_indices",
        )
        if len(set(indices)) != len(indices):
            raise ValueError(
                "source_row_indices must not identify the same source row "
                "more than once."
            )
        if not indices:
            raise ValueError(
                "A channel predictive result requires at least one source row."
            )
        reason = self.reason
        if reason is not None:
            if not isinstance(reason, str):
                raise TypeError("reason must be a string when supplied.")
            reason = reason.strip()
            if not reason:
                raise ValueError("reason must be non-empty when supplied.")
        predictive = self.predictive_uncertainty

        if disposition is (
            InstrumentChannelCalibrationPredictiveUncertaintyDisposition.AVAILABLE
        ):
            if not isinstance(
                predictive,
                InstrumentChannelCalibrationPredictiveUncertainty,
            ):
                raise ValueError(
                    "An available channel result requires predictive_uncertainty."
                )
            if predictive.status is not (
                InstrumentChannelCalibrationPredictiveUncertaintyStatus.AVAILABLE
            ):
                raise ValueError(
                    "An available channel result requires available predictive "
                    "uncertainty."
                )
            if predictive.input_shape != (len(indices),):
                raise ValueError(
                    "predictive_uncertainty input_shape must match the channel "
                    "source-row count."
                )
            if reason is not None:
                raise ValueError(
                    "An available channel result must not carry a reason."
                )
        elif disposition is (
            InstrumentChannelCalibrationPredictiveUncertaintyDisposition.UNAVAILABLE
        ):
            if not isinstance(
                predictive,
                InstrumentChannelCalibrationPredictiveUncertainty,
            ):
                raise ValueError(
                    "An unavailable channel result requires an explicit "
                    "predictive_uncertainty record."
                )
            if predictive.status is not (
                InstrumentChannelCalibrationPredictiveUncertaintyStatus.UNAVAILABLE
            ):
                raise ValueError(
                    "An unavailable channel result requires unavailable "
                    "predictive uncertainty."
                )
            if predictive.input_shape != (len(indices),):
                raise ValueError(
                    "predictive_uncertainty input_shape must match the channel "
                    "source-row count."
                )
            if reason is None or predictive.reason != reason:
                raise ValueError(
                    "Unavailable channel reason must match the nested predictive "
                    "uncertainty reason."
                )
        else:
            if predictive is not None:
                raise ValueError(
                    "not_requested and skipped channel results must not carry "
                    "predictive_uncertainty."
                )
            if disposition is (
                InstrumentChannelCalibrationPredictiveUncertaintyDisposition.SKIPPED
            ):
                if reason is None:
                    raise ValueError(
                        "A skipped channel result requires a reason."
                    )
            elif reason is not None:
                raise ValueError(
                    "A not_requested channel result must not carry a reason."
                )

        object.__setattr__(self, "physical_wavelength", wavelength)
        object.__setattr__(self, "reference_channel", reference_channel)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "source_row_indices", indices)
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe per-channel representation."""

        return {
            "physical_wavelength": self.physical_wavelength,
            "reference_channel": self.reference_channel,
            "channel": self.channel,
            "disposition": self.disposition.value,
            "source_row_indices": list(self.source_row_indices),
            "predictive_uncertainty": (
                None
                if self.predictive_uncertainty is None
                else self.predictive_uncertainty.to_dict()
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationPredictiveUncertaintyGroupResult:
    """Predictive results for distinct channels at one physical wavelength."""

    physical_wavelength: float
    reference_channel: str
    reference_source_row_indices: tuple[int, ...]
    channel_results: tuple[
        InstrumentChannelCalibrationPredictiveUncertaintyChannelResult,
        ...,
    ]

    def __post_init__(self) -> None:
        wavelength = float(self.physical_wavelength)
        if not math.isfinite(wavelength):
            raise ValueError("physical_wavelength must be finite.")
        reference_channel = _normalize_calibration_channel(
            self.reference_channel,
            name="reference_channel",
        )
        reference_indices = InstrumentChannelPairing._normalize_indices(
            self.reference_source_row_indices,
            name="reference_source_row_indices",
        )
        if not reference_indices:
            raise ValueError(
                "A predictive group requires at least one reference source row."
            )
        if len(set(reference_indices)) != len(reference_indices):
            raise ValueError(
                "reference_source_row_indices must not contain duplicates."
            )
        channel_results = tuple(self.channel_results)
        if not channel_results:
            raise ValueError(
                "A predictive group requires at least one non-reference "
                "channel result."
            )
        if any(
            not isinstance(
                result,
                InstrumentChannelCalibrationPredictiveUncertaintyChannelResult,
            )
            for result in channel_results
        ):
            raise TypeError(
                "channel_results must contain only predictive channel results."
            )
        channels = tuple(result.channel for result in channel_results)
        if len(set(channels)) != len(channels):
            raise ValueError(
                "Each observational channel may appear only once in a "
                "predictive group."
            )
        for result in channel_results:
            if result.physical_wavelength != wavelength:
                raise ValueError(
                    "Channel-result physical wavelength must match the group."
                )
            if result.reference_channel != reference_channel:
                raise ValueError(
                    "Channel-result reference channel must match the group."
                )
        all_indices = [*reference_indices]
        for result in channel_results:
            all_indices.extend(result.source_row_indices)
        if len(set(all_indices)) != len(all_indices):
            raise ValueError(
                "Reference and non-reference channel results must describe "
                "disjoint source rows."
            )

        object.__setattr__(self, "physical_wavelength", wavelength)
        object.__setattr__(self, "reference_channel", reference_channel)
        object.__setattr__(
            self,
            "reference_source_row_indices",
            reference_indices,
        )
        object.__setattr__(
            self,
            "channel_results",
            tuple(sorted(channel_results, key=lambda item: item.channel)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe group representation."""

        return {
            "physical_wavelength": self.physical_wavelength,
            "reference_channel": self.reference_channel,
            "reference_source_row_indices": list(
                self.reference_source_row_indices
            ),
            "reference_channel_fitted_coefficient_uncertainty_disposition": (
                "not_applicable"
            ),
            "channel_results": [
                result.to_dict() for result in self.channel_results
            ],
            "channel_merging_performed": False,
            "physical_wavelength_used_as_covariance_identity": False,
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationPredictiveUncertaintyOrchestration:
    """Dataset-level contract for predictive propagation results.

    Dataset rows are partitioned among reference rows, non-reference channel
    results, and unaffected rows.  Numerical predictive uncertainty remains
    channel-local; missing values are represented structurally rather than by
    NaN-filled dataset arrays.
    """

    schema_version: str
    covariance_mode: (
        InstrumentChannelCalibrationPredictiveCovarianceMode | str | None
    )
    source_row_indices: tuple[int, ...]
    group_results: tuple[
        InstrumentChannelCalibrationPredictiveUncertaintyGroupResult,
        ...,
    ]
    unaffected_source_row_indices: tuple[int, ...]
    ordinary_calibrated_flux_error_available: bool

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_ORCHESTRATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported instrument-channel calibration predictive "
                f"orchestration schema version: {self.schema_version!r}."
            )
        mode = self.covariance_mode
        if isinstance(mode, (bool, np.bool_)):
            raise TypeError(
                "covariance_mode must be a predictive covariance mode or None, "
                "not boolean."
            )
        if mode is not None and not isinstance(
            mode,
            InstrumentChannelCalibrationPredictiveCovarianceMode,
        ):
            try:
                mode = InstrumentChannelCalibrationPredictiveCovarianceMode(
                    str(mode)
                )
            except ValueError as exc:
                raise ValueError(
                    "Unsupported predictive covariance mode: "
                    f"{self.covariance_mode!r}."
                ) from exc
        raw_source_indices = tuple(self.source_row_indices)
        source_indices = _normalize_pairing_source_indices(
            raw_source_indices,
            size=len(raw_source_indices),
            name="source_row_indices",
        )
        unaffected_indices = InstrumentChannelPairing._normalize_indices(
            self.unaffected_source_row_indices,
            name="unaffected_source_row_indices",
        )
        if len(set(unaffected_indices)) != len(unaffected_indices):
            raise ValueError(
                "unaffected_source_row_indices must not contain duplicates."
            )
        if not isinstance(
            self.ordinary_calibrated_flux_error_available,
            (bool, np.bool_),
        ):
            raise TypeError(
                "ordinary_calibrated_flux_error_available must be boolean."
            )
        group_results = tuple(self.group_results)
        if any(
            not isinstance(
                group,
                InstrumentChannelCalibrationPredictiveUncertaintyGroupResult,
            )
            for group in group_results
        ):
            raise TypeError(
                "group_results must contain only predictive group results."
            )
        wavelengths = tuple(
            group.physical_wavelength for group in group_results
        )
        if len(set(wavelengths)) != len(wavelengths):
            raise ValueError(
                "Each physical wavelength may appear only once in group_results."
            )
        all_group_channels = tuple(
            channel
            for group in group_results
            for channel in (
                group.reference_channel,
                *(result.channel for result in group.channel_results),
            )
        )
        if len(set(all_group_channels)) != len(all_group_channels):
            raise ValueError(
                "Each observational channel may appear in only one predictive "
                "orchestration group."
            )
        channel_results = tuple(
            result
            for group in group_results
            for result in group.channel_results
        )
        if mode is None:
            invalid = tuple(
                result.disposition
                for result in channel_results
                if result.disposition
                not in (
                    InstrumentChannelCalibrationPredictiveUncertaintyDisposition
                    .NOT_REQUESTED,
                    InstrumentChannelCalibrationPredictiveUncertaintyDisposition
                    .SKIPPED,
                )
            )
            if invalid:
                raise ValueError(
                    "A not-requested orchestration result may contain only "
                    "not_requested or skipped channel dispositions."
                )
        else:
            if any(
                result.disposition is (
                    InstrumentChannelCalibrationPredictiveUncertaintyDisposition
                    .NOT_REQUESTED
                )
                for result in channel_results
            ):
                raise ValueError(
                    "A requested orchestration result cannot contain a "
                    "not_requested channel disposition."
                )
            for result in channel_results:
                predictive = result.predictive_uncertainty
                if predictive is not None and predictive.covariance_mode is not mode:
                    raise ValueError(
                        "Per-channel predictive covariance mode must match the "
                        "dataset request."
                    )
        represented_indices = [*unaffected_indices]
        for group in group_results:
            represented_indices.extend(group.reference_source_row_indices)
            for result in group.channel_results:
                represented_indices.extend(result.source_row_indices)
        if len(set(represented_indices)) != len(represented_indices):
            raise ValueError(
                "Predictive orchestration row partitions must be disjoint."
            )
        if set(represented_indices) != set(source_indices):
            raise ValueError(
                "Reference, channel, and unaffected source rows must partition "
                "source_row_indices exactly."
            )
        source_position = {
            source_index: position
            for position, source_index in enumerate(source_indices)
        }
        ordered_partitions = [
            ("unaffected_source_row_indices", unaffected_indices),
        ]
        for group in group_results:
            ordered_partitions.append(
                (
                    "reference_source_row_indices",
                    group.reference_source_row_indices,
                )
            )
            ordered_partitions.extend(
                ("channel source_row_indices", result.source_row_indices)
                for result in group.channel_results
            )
        for name, partition in ordered_partitions:
            positions = tuple(source_position[index] for index in partition)
            if positions != tuple(sorted(positions)):
                raise ValueError(
                    f"{name} must preserve original dataset row order."
                )

        object.__setattr__(self, "covariance_mode", mode)
        object.__setattr__(self, "source_row_indices", source_indices)
        object.__setattr__(
            self,
            "group_results",
            tuple(
                sorted(
                    group_results,
                    key=lambda item: item.physical_wavelength,
                )
            ),
        )
        object.__setattr__(
            self,
            "unaffected_source_row_indices",
            unaffected_indices,
        )
        object.__setattr__(
            self,
            "ordinary_calibrated_flux_error_available",
            bool(self.ordinary_calibrated_flux_error_available),
        )

    @property
    def requested(self) -> bool:
        """Return whether fitted-coefficient propagation was requested."""

        return self.covariance_mode is not None

    @property
    def channel_results(
        self,
    ) -> tuple[
        InstrumentChannelCalibrationPredictiveUncertaintyChannelResult,
        ...,
    ]:
        """Return all non-reference channel results."""

        return tuple(
            result
            for group in self.group_results
            for result in group.channel_results
        )

    def _count(
        self,
        disposition: InstrumentChannelCalibrationPredictiveUncertaintyDisposition,
    ) -> int:
        return sum(
            result.disposition is disposition
            for result in self.channel_results
        )

    @property
    def n_available_channels(self) -> int:
        return self._count(
            InstrumentChannelCalibrationPredictiveUncertaintyDisposition.AVAILABLE
        )

    @property
    def n_unavailable_channels(self) -> int:
        return self._count(
            InstrumentChannelCalibrationPredictiveUncertaintyDisposition.UNAVAILABLE
        )

    @property
    def n_skipped_channels(self) -> int:
        return self._count(
            InstrumentChannelCalibrationPredictiveUncertaintyDisposition.SKIPPED
        )

    @property
    def n_not_requested_channels(self) -> int:
        return self._count(
            InstrumentChannelCalibrationPredictiveUncertaintyDisposition
            .NOT_REQUESTED
        )

    @property
    def all_eligible_channels_available(self) -> bool:
        eligible = self.n_available_channels + self.n_unavailable_channels
        return self.requested and eligible > 0 and self.n_unavailable_channels == 0

    @property
    def fitted_coefficient_uncertainty_propagated(self) -> bool:
        """Return true when at least one channel has an available result."""

        return self.n_available_channels > 0

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dataset-level representation."""

        return {
            "schema_version": self.schema_version,
            "requested": self.requested,
            "covariance_mode": (
                None
                if self.covariance_mode is None
                else self.covariance_mode.value
            ),
            "source_row_indices": list(self.source_row_indices),
            "group_results": [
                group.to_dict() for group in self.group_results
            ],
            "unaffected_source_row_indices": list(
                self.unaffected_source_row_indices
            ),
            "n_available_channels": self.n_available_channels,
            "n_unavailable_channels": self.n_unavailable_channels,
            "n_skipped_channels": self.n_skipped_channels,
            "n_not_requested_channels": self.n_not_requested_channels,
            "any_predictive_uncertainty_available": (
                self.n_available_channels > 0
            ),
            "all_eligible_channels_available": (
                self.all_eligible_channels_available
            ),
            "ordinary_calibrated_flux_error_available": (
                self.ordinary_calibrated_flux_error_available
            ),
            "predictive_standard_deviation_blocks_available": (
                self.n_available_channels > 0
            ),
            "predictive_covariance_blocks_available": (
                self.covariance_mode
                is InstrumentChannelCalibrationPredictiveCovarianceMode
                .FULL_COVARIANCE
                and self.n_available_channels > 0
            ),
            "global_predictive_standard_deviation_emitted": False,
            "global_dense_predictive_covariance_emitted": False,
            "cross_observational_channel_covariance_emitted": False,
            "cross_observational_channel_covariance_assumed_zero": False,
            "fitted_coefficient_uncertainty_propagated": (
                self.fitted_coefficient_uncertainty_propagated
            ),
            "measurement_coefficient_independence_assumed": True,
            "coefficient_order": ["offset", "scale"],
            "row_alignment": "original_dataset_source_row_indices",
            "missing_values_represented_by_nan": False,
        }


INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class InstrumentChannelCalibration:
    """Affine mapping from one observational channel to a reference channel.

    The stored coefficients follow the explicit convention

    ``reference_flux = offset + scale * channel_flux``.

    The model is estimated only from caller-supplied paired measurements.
    PGMUVI does not create pairs from asynchronous light curves.
    """

    schema_version: str
    reference_channel: str
    channel: str
    wavelength: float
    offset: float
    scale: float
    n_pairs: int
    n_inliers: int
    residual_mad_sigma: float | None
    coefficient_uncertainty: (
        InstrumentChannelCalibrationCoefficientUncertainty
    )
    fit_method: str = "iterative_mad_clipped_affine"
    fit_provenance: InstrumentChannelCalibrationFitProvenance | None = None

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported instrument-channel calibration model "
                f"schema version: {self.schema_version!r}."
            )

        reference_channel = str(self.reference_channel).strip()
        channel = str(self.channel).strip()

        if not reference_channel:
            raise ValueError("reference_channel must be non-empty.")
        if not channel:
            raise ValueError("channel must be non-empty.")
        if channel == reference_channel:
            raise ValueError(
                "channel and reference_channel must identify different "
                "observational channels."
            )

        object.__setattr__(
            self,
            "reference_channel",
            reference_channel,
        )
        object.__setattr__(self, "channel", channel)

        if not isinstance(self.fit_method, str):
            raise TypeError("fit_method must be a string.")

        fit_method = self.fit_method.strip()
        if not fit_method:
            raise ValueError("fit_method must be non-empty.")

        object.__setattr__(self, "fit_method", fit_method)

        finite_values = {
            "wavelength": self.wavelength,
            "offset": self.offset,
            "scale": self.scale,
        }
        for name, value in finite_values.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if self.scale <= 0.0:
            raise ValueError("scale must be strictly positive.")
        if self.n_pairs < 3:
            raise ValueError("n_pairs must be at least 3.")
        if not 3 <= self.n_inliers <= self.n_pairs:
            raise ValueError(
                "n_inliers must be between 3 and n_pairs, inclusive."
            )

        if self.residual_mad_sigma is not None:
            residual_scale = float(self.residual_mad_sigma)
            if not np.isfinite(residual_scale) or residual_scale < 0.0:
                raise ValueError(
                    "residual_mad_sigma must be finite and non-negative "
                    "when supplied."
                )

        if not isinstance(
            self.coefficient_uncertainty,
            InstrumentChannelCalibrationCoefficientUncertainty,
        ):
            raise TypeError(
                "coefficient_uncertainty must be an "
                "InstrumentChannelCalibrationCoefficientUncertainty "
                "instance."
            )

        fit_provenance = self.fit_provenance
        if fit_provenance is not None:
            if not isinstance(
                fit_provenance,
                InstrumentChannelCalibrationFitProvenance,
            ):
                raise TypeError(
                    "fit_provenance must be an "
                    "InstrumentChannelCalibrationFitProvenance instance "
                    "when supplied."
                )
            if fit_provenance.n_finite_pairs != self.n_pairs:
                raise ValueError(
                    "fit_provenance finite-pair count must equal n_pairs."
                )
            if fit_provenance.n_final_inliers != self.n_inliers:
                raise ValueError(
                    "fit_provenance final-inlier count must equal "
                    "n_inliers."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe representation."""

        return {
            "schema_version": self.schema_version,
            "reference_channel": self.reference_channel,
            "channel": self.channel,
            "wavelength": float(self.wavelength),
            "offset": float(self.offset),
            "scale": float(self.scale),
            "n_pairs": int(self.n_pairs),
            "n_inliers": int(self.n_inliers),
            "residual_mad_sigma": (
                None
                if self.residual_mad_sigma is None
                else float(self.residual_mad_sigma)
            ),
            "fit_method": self.fit_method,
            "fit_provenance": (
                None
                if self.fit_provenance is None
                else self.fit_provenance.to_dict()
            ),
            "coefficient_uncertainty": (
                self.coefficient_uncertainty.to_dict()
            ),
            "application_equation": (
                "reference_flux = offset + scale * channel_flux"
            ),
            "automatic_time_matching": False,
            "automatic_model_selection": False,
        }



@dataclass(frozen=True)
class InstrumentChannelCalibrationChannelPlan:
    """Explicit plan for one non-reference observational channel.

    A planned entry records caller-selected pairing and calibration
    configuration. Skipped and unavailable entries require a reason and
    contain no executable pairing or calibration configuration.

    Optional pairing and fitted-calibration records preserve provenance.
    This record never constructs pairs, fits a calibration, or applies one.
    """

    channel: str
    disposition: InstrumentChannelCalibrationDisposition | str
    pairing_method: str | None = None
    maximum_time_separation: float | None = None
    time_unit: str | None = None
    calibration_family: str | None = None
    reason: str | None = None
    pairing: InstrumentChannelPairing | None = None
    calibration: InstrumentChannelCalibration | None = None

    def __post_init__(self) -> None:
        channel = str(self.channel).strip()
        if not channel:
            raise ValueError("channel must be non-empty.")

        disposition = self.disposition
        if not isinstance(
            disposition,
            InstrumentChannelCalibrationDisposition,
        ):
            try:
                disposition = InstrumentChannelCalibrationDisposition(
                    str(disposition)
                )
            except ValueError as exc:
                raise ValueError(
                    "Unsupported instrument-channel calibration "
                    f"disposition: {self.disposition!r}."
                ) from exc

        pairing_method = self._normalize_optional_text(
            self.pairing_method,
            name="pairing_method",
        )
        if pairing_method is not None:
            try:
                pairing_method = InstrumentChannelPairingMethod(
                    pairing_method
                ).value
            except ValueError as exc:
                raise ValueError(
                    "Unsupported instrument-channel pairing method: "
                    f"{pairing_method!r}."
                ) from exc

        time_unit = self._normalize_optional_text(
            self.time_unit,
            name="time_unit",
        )
        calibration_family = self._normalize_optional_text(
            self.calibration_family,
            name="calibration_family",
        )
        reason = self._normalize_optional_text(
            self.reason,
            name="reason",
        )

        maximum_time_separation = self.maximum_time_separation
        if maximum_time_separation is not None:
            if isinstance(
                maximum_time_separation,
                (bool, np.bool_),
            ):
                raise TypeError(
                    "maximum_time_separation must be numeric, "
                    "not boolean."
                )
            maximum_time_separation = float(
                maximum_time_separation
            )
            if (
                not math.isfinite(maximum_time_separation)
                or maximum_time_separation < 0.0
            ):
                raise ValueError(
                    "maximum_time_separation must be finite and "
                    "non-negative when supplied."
                )

        if (
            disposition
            is InstrumentChannelCalibrationDisposition.PLANNED
        ):
            if pairing_method is None:
                raise ValueError(
                    "A planned channel requires caller-selected "
                    "pairing_method."
                )
            if time_unit is None:
                raise ValueError(
                    "A planned channel requires an explicit time_unit."
                )
            if calibration_family is None:
                raise ValueError(
                    "A planned channel requires caller-selected "
                    "calibration_family."
                )
            if reason is not None:
                raise ValueError(
                    "A planned channel must not carry a skip or "
                    "unavailable reason."
                )

            if (
                pairing_method
                == InstrumentChannelPairingMethod.EXACT_TIMESTAMP.value
            ):
                if (
                    maximum_time_separation is not None
                    and maximum_time_separation != 0.0
                ):
                    raise ValueError(
                        "exact_timestamp planning permits no non-zero "
                        "maximum_time_separation."
                    )
            elif (
                pairing_method
                == InstrumentChannelPairingMethod
                .NEAREST_WITHIN_TOLERANCE.value
            ):
                if (
                    maximum_time_separation is None
                    or maximum_time_separation <= 0.0
                ):
                    raise ValueError(
                        "nearest_within_tolerance planning requires a "
                        "finite positive maximum_time_separation."
                    )
        else:
            if reason is None:
                raise ValueError(
                    "Skipped and unavailable channels require a reason."
                )
            if any(
                value is not None
                for value in (
                    pairing_method,
                    maximum_time_separation,
                    time_unit,
                    calibration_family,
                    self.pairing,
                    self.calibration,
                )
            ):
                raise ValueError(
                    "Skipped and unavailable channels cannot contain "
                    "pairing or calibration configuration."
                )

        if self.pairing is not None:
            if not isinstance(self.pairing, InstrumentChannelPairing):
                raise TypeError(
                    "pairing must be an InstrumentChannelPairing "
                    "instance when supplied."
                )
            if self.pairing.channel != channel:
                raise ValueError(
                    "pairing.channel must match the planned channel."
                )
            if self.pairing.method != pairing_method:
                raise ValueError(
                    "pairing.method must match pairing_method."
                )
            if self.pairing.time_unit != time_unit:
                raise ValueError(
                    "pairing.time_unit must match the planned time_unit."
                )

            planned_separation = maximum_time_separation
            pairing_separation = (
                self.pairing.maximum_time_separation
            )
            if (
                pairing_method
                == InstrumentChannelPairingMethod.EXACT_TIMESTAMP.value
            ):
                planned_separation = (
                    0.0
                    if planned_separation is None
                    else planned_separation
                )
                pairing_separation = (
                    0.0
                    if pairing_separation is None
                    else pairing_separation
                )

            if pairing_separation != planned_separation:
                raise ValueError(
                    "Pairing tolerance provenance must match the "
                    "planned maximum_time_separation."
                )

        if self.calibration is not None:
            if not isinstance(
                self.calibration,
                InstrumentChannelCalibration,
            ):
                raise TypeError(
                    "calibration must be an "
                    "InstrumentChannelCalibration instance when supplied."
                )
            if self.pairing is None:
                raise ValueError(
                    "A fitted calibration requires pairing provenance."
                )
            if self.calibration.channel != channel:
                raise ValueError(
                    "calibration.channel must match the planned channel."
                )
            if calibration_family != "affine":
                raise ValueError(
                    "InstrumentChannelCalibration provenance requires "
                    "calibration_family='affine'."
                )
            if self.calibration.n_pairs > self.pairing.n_pairs:
                raise ValueError(
                    "calibration.n_pairs cannot exceed pairing.n_pairs."
                )

        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "pairing_method",
            pairing_method,
        )
        object.__setattr__(
            self,
            "maximum_time_separation",
            maximum_time_separation,
        )
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(
            self,
            "calibration_family",
            calibration_family,
        )
        object.__setattr__(self, "reason", reason)

    @staticmethod
    def _normalize_optional_text(
        value: Any,
        *,
        name: str,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string when supplied.")

        normalized = value.strip()
        if not normalized:
            raise ValueError(
                f"{name} must be non-empty when supplied."
            )
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe representation."""

        return {
            "channel": self.channel,
            "disposition": self.disposition.value,
            "pairing_method": self.pairing_method,
            "maximum_time_separation": (
                self.maximum_time_separation
            ),
            "time_unit": self.time_unit,
            "calibration_family": self.calibration_family,
            "reason": self.reason,
            "pairing": (
                None
                if self.pairing is None
                else self.pairing.to_dict()
            ),
            "calibration": (
                None
                if self.calibration is None
                else self.calibration.to_dict()
            ),
            "calibration_applied": False,
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationGroupPlan:
    """Explicit calibration plan for one shared-wavelength group."""

    physical_wavelength: float
    reference_channel: str
    channel_plans: tuple[
        InstrumentChannelCalibrationChannelPlan,
        ...,
    ]

    def __post_init__(self) -> None:
        wavelength = float(self.physical_wavelength)
        if not math.isfinite(wavelength):
            raise ValueError(
                "physical_wavelength must be finite."
            )

        reference_channel = _normalize_calibration_channel(
            self.reference_channel,
            name="reference_channel",
        )
        channel_plans = tuple(self.channel_plans)
        if not channel_plans:
            raise ValueError(
                "A calibration group plan requires at least one "
                "non-reference channel plan."
            )
        if any(
            not isinstance(
                channel_plan,
                InstrumentChannelCalibrationChannelPlan,
            )
            for channel_plan in channel_plans
        ):
            raise TypeError(
                "channel_plans must contain only "
                "InstrumentChannelCalibrationChannelPlan instances."
            )

        planned_channels = [
            channel_plan.channel
            for channel_plan in channel_plans
        ]
        if len(set(planned_channels)) != len(planned_channels):
            raise ValueError(
                "Each non-reference channel may appear only once in a "
                "calibration group plan."
            )
        if reference_channel in planned_channels:
            raise ValueError(
                "reference_channel cannot also be a non-reference "
                "channel plan."
            )

        for channel_plan in channel_plans:
            pairing = channel_plan.pairing
            if pairing is not None:
                if pairing.reference_channel != reference_channel:
                    raise ValueError(
                        "pairing.reference_channel must match the group "
                        "reference_channel."
                    )
                if pairing.wavelength != wavelength:
                    raise ValueError(
                        "pairing.wavelength must match the group "
                        "physical_wavelength."
                    )

            calibration = channel_plan.calibration
            if calibration is not None:
                if (
                    calibration.reference_channel
                    != reference_channel
                ):
                    raise ValueError(
                        "calibration.reference_channel must match the "
                        "group reference_channel."
                    )
                if calibration.wavelength != wavelength:
                    raise ValueError(
                        "calibration.wavelength must match the group "
                        "physical_wavelength."
                    )

        object.__setattr__(
            self,
            "physical_wavelength",
            wavelength,
        )
        object.__setattr__(
            self,
            "reference_channel",
            reference_channel,
        )
        object.__setattr__(
            self,
            "channel_plans",
            tuple(
                sorted(
                    channel_plans,
                    key=lambda item: item.channel,
                )
            ),
        )

    @property
    def observational_channels(self) -> tuple[str, ...]:
        """Return all channels represented by this group plan."""

        return tuple(
            sorted(
                (
                    self.reference_channel,
                    *(
                        item.channel
                        for item in self.channel_plans
                    ),
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe representation."""

        return {
            "physical_wavelength": self.physical_wavelength,
            "reference_channel": self.reference_channel,
            "observational_channels": list(
                self.observational_channels
            ),
            "channel_plans": [
                channel_plan.to_dict()
                for channel_plan in self.channel_plans
            ],
        }


@dataclass(frozen=True)
class InstrumentChannelCalibrationPlan:
    """Immutable dataset-level calibration orchestration contract."""

    schema_version: str
    assessment: InstrumentChannelCalibrationAssessment
    group_plans: tuple[
        InstrumentChannelCalibrationGroupPlan,
        ...,
    ]

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_PLAN_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported instrument-channel calibration plan "
                f"schema version: {self.schema_version!r}."
            )
        if not isinstance(
            self.assessment,
            InstrumentChannelCalibrationAssessment,
        ):
            raise TypeError(
                "assessment must be an "
                "InstrumentChannelCalibrationAssessment instance."
            )

        group_plans = tuple(self.group_plans)
        if any(
            not isinstance(
                group_plan,
                InstrumentChannelCalibrationGroupPlan,
            )
            for group_plan in group_plans
        ):
            raise TypeError(
                "group_plans must contain only "
                "InstrumentChannelCalibrationGroupPlan instances."
            )

        observed_wavelengths = [
            group_plan.physical_wavelength
            for group_plan in group_plans
        ]
        if (
            len(set(observed_wavelengths))
            != len(observed_wavelengths)
        ):
            raise ValueError(
                "Each shared physical wavelength may have only one "
                "calibration group plan."
            )

        expected_groups = {
            group.physical_wavelength: (
                group.observational_channels
            )
            for group in (
                self.assessment.shared_wavelength_groups
            )
        }
        observed_groups = {
            group_plan.physical_wavelength: (
                group_plan.observational_channels
            )
            for group_plan in group_plans
        }

        if observed_groups != expected_groups:
            expected_group_summary = tuple(
                sorted(expected_groups.items())
            )
            observed_group_summary = tuple(
                sorted(observed_groups.items())
            )
            raise ValueError(
                "group_plans must cover exactly the shared-wavelength "
                "groups and observational channels in assessment. "
                f"expected_groups={expected_group_summary!r}; "
                f"observed_groups={observed_group_summary!r}."
            )

        object.__setattr__(
            self,
            "group_plans",
            tuple(
                sorted(
                    group_plans,
                    key=lambda item: item.physical_wavelength,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe representation."""

        return {
            "schema_version": self.schema_version,
            "assessment": self.assessment.to_dict(),
            "group_plans": [
                group_plan.to_dict()
                for group_plan in self.group_plans
            ],
            "n_shared_wavelength_groups": len(
                self.group_plans
            ),
            "automatic_reference_channel_selection": False,
            "automatic_pairing_method_selection": False,
            "automatic_time_tolerance_selection": False,
            "automatic_calibration_family_selection": False,
            "automatic_pair_construction": False,
            "automatic_calibration_fit": False,
            "automatic_calibration_application": False,
            "channel_merging_performed": False,
            "wavelength_reassignment_performed": False,
            "lightcurve_mutation_performed": False,
            "marker": INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER,
        }


def define_instrument_channel_calibration_plan(
    assessment: InstrumentChannelCalibrationAssessment,
    group_plans: Any,
) -> InstrumentChannelCalibrationPlan:
    """Validate and record explicit dataset-level calibration choices.

    The caller supplies every reference channel, per-channel disposition,
    pairing method, tolerance, time unit, and calibration family. Optional
    pairing and fitted-calibration records are retained as provenance.

    This callable does not construct pairs, fit or apply calibrations,
    merge channels, alter wavelengths, mutate a light curve, or select any
    scientific configuration automatically.
    """

    if not isinstance(
        assessment,
        InstrumentChannelCalibrationAssessment,
    ):
        raise TypeError(
            "assessment must be an "
            "InstrumentChannelCalibrationAssessment instance."
        )

    return InstrumentChannelCalibrationPlan(
        schema_version=(
            INSTRUMENT_CHANNEL_CALIBRATION_PLAN_SCHEMA_VERSION
        ),
        assessment=assessment,
        group_plans=tuple(group_plans),
    )


@dataclass(frozen=True)
class InstrumentChannelCalibrationExecution:
    """Immutable result of executing an explicit calibration plan.

    The result contains copied calibrated arrays and a completed plan carrying
    the pairing and affine-calibration provenance used for every planned
    non-reference channel. Input arrays are never mutated.

    This record does not claim that the caller-selected reference channels,
    pairing methods, tolerances, or affine family are scientifically optimal.
    Fitted-coefficient uncertainty is not propagated.
    """

    schema_version: str
    plan: InstrumentChannelCalibrationPlan
    source_row_indices: tuple[int, ...]
    calibrated_flux: tuple[float, ...]
    calibrated_flux_error: tuple[float, ...] | None
    applied_channels: tuple[str, ...]
    applied_source_row_indices: tuple[int, ...]
    n_applied_observations: int

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_EXECUTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported instrument-channel calibration execution "
                f"schema version: {self.schema_version!r}."
            )

        if not isinstance(
            self.plan,
            InstrumentChannelCalibrationPlan,
        ):
            raise TypeError(
                "plan must be an InstrumentChannelCalibrationPlan "
                "instance."
            )

        calibrated_flux = tuple(
            float(value)
            for value in self.calibrated_flux
        )
        if any(
            not math.isfinite(value)
            for value in calibrated_flux
        ):
            raise ValueError("calibrated_flux must contain finite values.")

        source_row_indices = (
            _normalize_pairing_source_indices(
                self.source_row_indices,
                size=len(calibrated_flux),
                name="source_row_indices",
            )
        )

        calibrated_flux_error = self.calibrated_flux_error
        if calibrated_flux_error is not None:
            calibrated_flux_error = tuple(
                float(value)
                for value in calibrated_flux_error
            )
            if len(calibrated_flux_error) != len(calibrated_flux):
                raise ValueError(
                    "calibrated_flux_error must have the same length as "
                    "calibrated_flux."
                )
            if any(
                not math.isfinite(value) or value < 0.0
                for value in calibrated_flux_error
            ):
                raise ValueError(
                    "calibrated_flux_error must contain finite "
                    "non-negative values."
                )

        applied_channels = tuple(
            sorted(
                {
                    _normalize_calibration_channel(
                        channel,
                        name="applied channel",
                    )
                    for channel in self.applied_channels
                }
            )
        )

        applied_source_row_indices = (
            InstrumentChannelPairing._normalize_indices(
                self.applied_source_row_indices,
                name="applied_source_row_indices",
            )
        )
        if (
            len(set(applied_source_row_indices))
            != len(applied_source_row_indices)
        ):
            raise ValueError(
                "applied_source_row_indices must not contain "
                "duplicate source rows."
            )

        source_row_index_set = set(source_row_indices)
        missing_applied_indices = tuple(
            index
            for index in applied_source_row_indices
            if index not in source_row_index_set
        )
        if missing_applied_indices:
            raise ValueError(
                "applied_source_row_indices must be drawn from "
                "source_row_indices; missing="
                f"{missing_applied_indices!r}."
            )

        planned_channels = {
            channel_plan.channel
            for group_plan in self.plan.group_plans
            for channel_plan in group_plan.channel_plans
            if channel_plan.disposition
            is InstrumentChannelCalibrationDisposition.PLANNED
        }
        if set(applied_channels) != planned_channels:
            raise ValueError(
                "applied_channels must match exactly the planned "
                "channels in plan."
            )

        if (
            isinstance(self.n_applied_observations, (bool, np.bool_))
            or not isinstance(
                self.n_applied_observations,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "n_applied_observations must be an integer."
            )
        n_applied_observations = int(
            self.n_applied_observations
        )
        if n_applied_observations < 0:
            raise ValueError(
                "n_applied_observations must be non-negative."
            )
        if n_applied_observations != len(
            applied_source_row_indices
        ):
            raise ValueError(
                "n_applied_observations must equal the number of "
                "applied_source_row_indices."
            )

        object.__setattr__(
            self,
            "source_row_indices",
            source_row_indices,
        )
        object.__setattr__(
            self,
            "calibrated_flux",
            calibrated_flux,
        )
        object.__setattr__(
            self,
            "calibrated_flux_error",
            calibrated_flux_error,
        )
        object.__setattr__(
            self,
            "applied_channels",
            applied_channels,
        )
        object.__setattr__(
            self,
            "applied_source_row_indices",
            applied_source_row_indices,
        )
        object.__setattr__(
            self,
            "n_applied_observations",
            n_applied_observations,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe execution representation."""

        return {
            "schema_version": self.schema_version,
            "plan": self.plan.to_dict(),
            "source_row_indices": list(self.source_row_indices),
            "calibrated_flux": list(self.calibrated_flux),
            "calibrated_flux_error": (
                None
                if self.calibrated_flux_error is None
                else list(self.calibrated_flux_error)
            ),
            "applied_channels": list(self.applied_channels),
            "applied_source_row_indices": list(
                self.applied_source_row_indices
            ),
            "n_applied_channels": len(self.applied_channels),
            "n_applied_observations": (
                self.n_applied_observations
            ),
            "plan_execution_performed": True,
            "input_mutation_performed": False,
            "channel_merging_performed": False,
            "wavelength_reassignment_performed": False,
            "automatic_reference_channel_selection": False,
            "automatic_pairing_method_selection": False,
            "automatic_time_tolerance_selection": False,
            "automatic_calibration_family_selection": False,
            "fitted_coefficient_uncertainty_propagated": False,
            "scientific_pairing_validation_performed": False,
            "marker": INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER,
        }


def _as_finite_execution_vector(
    values: Any,
    *,
    name: str,
    non_negative: bool = False,
) -> np.ndarray:
    raw = np.asarray(values)
    contains_boolean = (
        raw.dtype.kind == "b"
        or (
            raw.dtype.kind == "O"
            and any(
                isinstance(value, (bool, np.bool_))
                for value in raw.flat
            )
        )
    )
    if contains_boolean:
        raise TypeError(
            f"{name} must contain numeric values, not booleans."
        )

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    if non_negative and np.any(array < 0.0):
        raise ValueError(
            f"{name} must contain non-negative values."
        )

    return array


def _execution_pair_positions(
    pairing: InstrumentChannelPairing,
    *,
    row_position_by_index: dict[int, int],
    observational_channels: tuple[str, ...],
    physical_wavelengths: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        reference_positions = np.asarray(
            [
                row_position_by_index[index]
                for index in pairing.reference_row_indices
            ],
            dtype=int,
        )
        channel_positions = np.asarray(
            [
                row_position_by_index[index]
                for index in pairing.channel_row_indices
            ],
            dtype=int,
        )
    except KeyError as exception:
        raise ValueError(
            "Pairing provenance references a source row index that is "
            "absent from the execution input: "
            f"{exception.args[0]!r}."
        ) from exception

    for position in reference_positions:
        if (
            observational_channels[position]
            != pairing.reference_channel
        ):
            raise ValueError(
                "Pairing reference_row_indices do not identify the "
                "declared reference_channel in the execution input."
            )
        if (
            physical_wavelengths[position]
            != pairing.wavelength
        ):
            raise ValueError(
                "Pairing reference_row_indices do not identify the "
                "declared wavelength in the execution input."
            )

    for position in channel_positions:
        if observational_channels[position] != pairing.channel:
            raise ValueError(
                "Pairing channel_row_indices do not identify the "
                "declared channel in the execution input."
            )
        if physical_wavelengths[position] != pairing.wavelength:
            raise ValueError(
                "Pairing channel_row_indices do not identify the "
                "declared wavelength in the execution input."
            )

    if not np.array_equal(
        times[reference_positions],
        np.asarray(pairing.reference_times, dtype=float),
    ):
        raise ValueError(
            "Pairing reference_times do not match the execution rows "
            "identified by reference_row_indices."
        )
    if not np.array_equal(
        times[channel_positions],
        np.asarray(pairing.channel_times, dtype=float),
    ):
        raise ValueError(
            "Pairing channel_times do not match the execution rows "
            "identified by channel_row_indices."
        )

    return reference_positions, channel_positions


def execute_instrument_channel_calibration_plan(
    plan: InstrumentChannelCalibrationPlan,
    times: Any,
    flux: Any,
    physical_wavelengths: Any,
    observational_channel_labels: Any,
    *,
    flux_error: Any | None = None,
    source_row_indices: Any | None = None,
    predictive_uncertainty_request: (
        InstrumentChannelCalibrationPredictiveUncertaintyRequest | None
    ) = None,
) -> InstrumentChannelCalibrationExecution:
    """Execute a caller-authored dataset-level calibration plan.

    Planned entries reuse attached pairing or calibration provenance when
    present. Otherwise the callable constructs pairs using the explicitly
    recorded method and tolerance, fits the explicitly recorded affine family,
    and applies the mapping to every row of the planned target channel at the
    shared physical wavelength.

    Skipped and unavailable entries are preserved without modification.
    Reference-channel rows are never transformed. Inputs are copied rather than
    mutated, observational channels remain distinct, and wavelengths are not
    reassigned.

    The callable performs no automatic reference-channel, pairing-method,
    tolerance, or calibration-family selection. Dataset predictive propagation
    is an explicit opt-in request. Its contract is defined, but execution is not
    activated here; a supplied request raises :class:`NotImplementedError`.
    Calibration is not integrated into :class:`pgmuvi.lightcurve.Lightcurve`.
    """

    if predictive_uncertainty_request is not None:
        if isinstance(predictive_uncertainty_request, (bool, np.bool_)):
            raise TypeError(
                "predictive_uncertainty_request must be an "
                "InstrumentChannelCalibrationPredictiveUncertaintyRequest, "
                "not boolean."
            )
        if not isinstance(
            predictive_uncertainty_request,
            InstrumentChannelCalibrationPredictiveUncertaintyRequest,
        ):
            raise TypeError(
                "predictive_uncertainty_request must be an "
                "InstrumentChannelCalibrationPredictiveUncertaintyRequest "
                "when supplied."
            )
        raise NotImplementedError(
            "Dataset calibration predictive-uncertainty propagation is "
            "defined but not yet activated."
        )

    if not isinstance(
        plan,
        InstrumentChannelCalibrationPlan,
    ):
        raise TypeError(
            "plan must be an InstrumentChannelCalibrationPlan "
            "instance."
        )

    normalized_times = np.asarray(
        InstrumentChannelPairing._normalize_times(
            times,
            name="times",
        ),
        dtype=float,
    )
    normalized_flux = _as_finite_execution_vector(
        flux,
        name="flux",
    )
    normalized_wavelengths = _as_finite_execution_vector(
        physical_wavelengths,
        name="physical_wavelengths",
    )

    labels_array = np.asarray(
        observational_channel_labels,
        dtype=object,
    )
    if labels_array.ndim != 1:
        raise ValueError(
            "observational_channel_labels must be one-dimensional."
        )
    normalized_labels = tuple(
        _normalize_calibration_channel(
            value,
            name="observational channel label",
        )
        for value in labels_array
    )

    size = normalized_flux.size
    for name, array_size in (
        ("times", normalized_times.size),
        ("physical_wavelengths", normalized_wavelengths.size),
        (
            "observational_channel_labels",
            len(normalized_labels),
        ),
    ):
        if array_size != size:
            raise ValueError(
                f"{name} must have the same length as flux."
            )

    normalized_error = None
    if flux_error is not None:
        normalized_error = _as_finite_execution_vector(
            flux_error,
            name="flux_error",
            non_negative=True,
        )
        if normalized_error.size != size:
            raise ValueError(
                "flux_error must have the same length as flux."
            )

    normalized_source_indices = (
        _normalize_pairing_source_indices(
            source_row_indices,
            size=size,
            name="source_row_indices",
        )
    )
    row_position_by_index = {
        index: position
        for position, index in enumerate(
            normalized_source_indices
        )
    }

    observed_assessment = (
        assess_instrument_channel_calibration_requirement(
            normalized_wavelengths,
            normalized_labels,
        )
    )
    if observed_assessment != plan.assessment:
        raise ValueError(
            "Execution inputs do not reproduce the calibration "
            "assessment stored in plan."
        )

    calibrated_flux = normalized_flux.copy()
    calibrated_error = (
        None
        if normalized_error is None
        else normalized_error.copy()
    )

    completed_group_plans = []
    applied_channels = []
    applied_observation_mask = np.zeros(size, dtype=bool)

    labels_for_mask = np.asarray(normalized_labels, dtype=object)
    source_indices_array = np.asarray(
        normalized_source_indices,
        dtype=int,
    )

    for group_plan in plan.group_plans:
        reference_mask = (
            (labels_for_mask == group_plan.reference_channel)
            & (
                normalized_wavelengths
                == group_plan.physical_wavelength
            )
        )
        if not np.any(reference_mask):
            raise ValueError(
                "Execution input contains no rows for reference channel "
                f"{group_plan.reference_channel!r} at wavelength "
                f"{group_plan.physical_wavelength!r}."
            )

        completed_channel_plans = []

        for channel_plan in group_plan.channel_plans:
            if channel_plan.disposition is not (
                InstrumentChannelCalibrationDisposition.PLANNED
            ):
                completed_channel_plans.append(channel_plan)
                continue

            channel_mask = (
                (labels_for_mask == channel_plan.channel)
                & (
                    normalized_wavelengths
                    == group_plan.physical_wavelength
                )
            )
            if not np.any(channel_mask):
                raise ValueError(
                    "Execution input contains no rows for planned channel "
                    f"{channel_plan.channel!r} at wavelength "
                    f"{group_plan.physical_wavelength!r}."
                )

            pairing = channel_plan.pairing
            if pairing is None:
                pairing = construct_instrument_channel_pairing(
                    normalized_times[reference_mask],
                    normalized_times[channel_mask],
                    reference_channel=(
                        group_plan.reference_channel
                    ),
                    channel=channel_plan.channel,
                    wavelength=group_plan.physical_wavelength,
                    time_unit=channel_plan.time_unit,
                    method=channel_plan.pairing_method,
                    maximum_time_separation=(
                        channel_plan.maximum_time_separation
                    ),
                    reference_row_indices=(
                        source_indices_array[reference_mask]
                    ),
                    channel_row_indices=(
                        source_indices_array[channel_mask]
                    ),
                )

            if not pairing.usable_for_affine_calibration:
                raise ValueError(
                    "Planned calibration requires at least three paired "
                    "observations for channel "
                    f"{channel_plan.channel!r}; pairing contains "
                    f"{pairing.n_pairs}."
                )

            (
                reference_pair_positions,
                channel_pair_positions,
            ) = _execution_pair_positions(
                pairing,
                row_position_by_index=row_position_by_index,
                observational_channels=normalized_labels,
                physical_wavelengths=normalized_wavelengths,
                times=normalized_times,
            )

            calibration = channel_plan.calibration
            if calibration is None:
                if channel_plan.calibration_family != "affine":
                    raise ValueError(
                        "Only the explicit 'affine' calibration family "
                        "can currently be executed."
                    )

                calibration = fit_instrument_channel_calibration(
                    normalized_flux[reference_pair_positions],
                    normalized_flux[channel_pair_positions],
                    reference_channel=(
                        group_plan.reference_channel
                    ),
                    channel=channel_plan.channel,
                    wavelength=group_plan.physical_wavelength,
                    reference_error=(
                        None
                        if normalized_error is None
                        else normalized_error[
                            reference_pair_positions
                        ]
                    ),
                    channel_error=(
                        None
                        if normalized_error is None
                        else normalized_error[
                            channel_pair_positions
                        ]
                    ),
                )

            if calibrated_error is None:
                calibrated_flux[channel_mask] = (
                    apply_instrument_channel_calibration(
                        normalized_flux[channel_mask],
                        calibration,
                    )
                )
            else:
                (
                    transformed_flux,
                    transformed_error,
                ) = apply_instrument_channel_calibration(
                    normalized_flux[channel_mask],
                    calibration,
                    flux_error=normalized_error[channel_mask],
                )
                calibrated_flux[channel_mask] = transformed_flux
                calibrated_error[channel_mask] = transformed_error

            completed_channel_plans.append(
                replace(
                    channel_plan,
                    pairing=pairing,
                    calibration=calibration,
                )
            )
            applied_channels.append(channel_plan.channel)
            applied_observation_mask[channel_mask] = True

        completed_group_plans.append(
            replace(
                group_plan,
                channel_plans=tuple(completed_channel_plans),
            )
        )

    completed_plan = define_instrument_channel_calibration_plan(
        plan.assessment,
        tuple(completed_group_plans),
    )

    return InstrumentChannelCalibrationExecution(
        schema_version=(
            INSTRUMENT_CHANNEL_CALIBRATION_EXECUTION_SCHEMA_VERSION
        ),
        plan=completed_plan,
        source_row_indices=normalized_source_indices,
        calibrated_flux=tuple(calibrated_flux),
        calibrated_flux_error=(
            None
            if calibrated_error is None
            else tuple(calibrated_error)
        ),
        applied_channels=tuple(applied_channels),
        applied_source_row_indices=tuple(
            source_indices_array[applied_observation_mask]
        ),
        n_applied_observations=int(
            np.count_nonzero(applied_observation_mask)
        ),
    )


def _as_calibration_vector(
    values: Any,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    return array


def _validate_optional_calibration_error(
    values: Any | None,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    if values is None:
        return None

    array = _as_calibration_vector(values, name=name)

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must have the same shape as the paired flux arrays."
        )

    return array


def _weighted_affine_solution(
    channel_flux: np.ndarray,
    reference_flux: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    design = np.column_stack(
        (
            np.ones(channel_flux.size, dtype=float),
            channel_flux,
        )
    )
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_reference = reference_flux * sqrt_weights

    offset, scale = np.linalg.lstsq(
        weighted_design,
        weighted_reference,
        rcond=None,
    )[0]

    return float(offset), float(scale)


def _estimate_affine_coefficient_uncertainty(
    channel_flux: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    *,
    measurement_errors_supplied: bool,
    channel_errors_supplied: bool,
) -> InstrumentChannelCalibrationCoefficientUncertainty:
    """Estimate covariance for the final fixed-weight affine solve.

    The covariance is conditional on the final clipped inlier set.
    Unweighted solves estimate residual variance, while reference-channel
    measurement variances are treated as known. Channel-axis measurement
    errors make the effective variances depend on the fitted scale, so this
    helper returns an explicit unavailable record rather than freezing those
    weights.
    """

    uncertainty_source = "pgmuvi_affine_fit_final_inliers"

    def unavailable(
        reason: str,
    ) -> InstrumentChannelCalibrationCoefficientUncertainty:
        return InstrumentChannelCalibrationCoefficientUncertainty(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION
            ),
            status=(
                InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE
            ),
            uncertainty_source=uncertainty_source,
            coefficient_covariance=None,
            reason=reason,
        )

    degrees_of_freedom = int(channel_flux.size - 2)
    if degrees_of_freedom < 1:
        return unavailable(
            "Final affine inlier set has zero residual degrees of freedom; "
            "coefficient covariance is unavailable."
        )

    if (
        residual.shape != channel_flux.shape
        or weights.shape != channel_flux.shape
    ):
        return unavailable(
            "Final affine covariance inputs have inconsistent shapes."
        )
    if np.any(~np.isfinite(weights) | (weights <= 0.0)):
        return unavailable(
            "Final affine weights are not finite and strictly positive; "
            "coefficient covariance is unavailable."
        )
    if np.any(~np.isfinite(channel_flux)) or np.any(~np.isfinite(residual)):
        return unavailable(
            "Final affine covariance inputs contain non-finite values."
        )

    if channel_errors_supplied:
        return unavailable(
            "Coefficient covariance is unavailable when channel-axis "
            "measurement errors contribute scale-dependent effective "
            "variances; the current affine fitter does not expose a "
            "covariance estimator for the full iterative weighting procedure."
        )

    design = np.column_stack(
        (
            np.ones(channel_flux.size, dtype=float),
            channel_flux,
        )
    )
    weighted_design = design * np.sqrt(weights)[:, None]

    try:
        _, singular_values, right_singular_vectors = np.linalg.svd(
            weighted_design,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return unavailable(
            "Final weighted affine design decomposition failed; "
            "coefficient covariance is unavailable."
        )

    if singular_values.shape != (2,) or np.any(
        ~np.isfinite(singular_values)
    ):
        return unavailable(
            "Final weighted affine design has invalid singular values; "
            "coefficient covariance is unavailable."
        )

    largest_singular_value = float(singular_values[0])
    smallest_singular_value = float(singular_values[-1])
    if largest_singular_value <= 0.0:
        return unavailable(
            "Final weighted affine design is rank-deficient; coefficient "
            "covariance is unavailable."
        )

    minimum_relative_singular_value = math.sqrt(np.finfo(float).eps)
    if (
        smallest_singular_value / largest_singular_value
        <= minimum_relative_singular_value
    ):
        return unavailable(
            "Final weighted affine design is numerically rank-deficient; "
            "coefficient covariance is unavailable."
        )

    inverse_squared_singular_values = 1.0 / singular_values**2
    covariance = (
        right_singular_vectors.T * inverse_squared_singular_values
    ) @ right_singular_vectors

    residual_variance = None
    if not measurement_errors_supplied:
        residual_variance = float(
            np.dot(residual, residual) / degrees_of_freedom
        )
        covariance = covariance * residual_variance
        estimation_method = (
            "ordinary_least_squares_residual_variance_scaled_"
            "normal_matrix_inverse"
        )
    else:
        estimation_method = (
            "known_variance_weighted_normal_matrix_inverse"
        )

    covariance = 0.5 * (covariance + covariance.T)
    if np.any(~np.isfinite(covariance)):
        return unavailable(
            "Final affine coefficient covariance contains non-finite values."
        )
    if residual_variance is not None and (
        not np.isfinite(residual_variance) or residual_variance < 0.0
    ):
        return unavailable(
            "Estimated affine residual variance is invalid; coefficient "
            "covariance is unavailable."
        )

    return InstrumentChannelCalibrationCoefficientUncertainty(
        schema_version=(
            INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION
        ),
        status=InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
        uncertainty_source=uncertainty_source,
        coefficient_covariance=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
        estimation_method=estimation_method,
        degrees_of_freedom=degrees_of_freedom,
        residual_variance=residual_variance,
    )


def _calibration_mad_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0

    centre = float(np.median(values))
    return float(
        1.4826 * np.median(np.abs(values - centre))
    )


def _normalize_calibration_channel(
    value: Any,
    *,
    name: str,
) -> str:
    channel_name = str(value).strip()

    if not channel_name:
        raise ValueError(f"{name} must be non-empty.")

    return channel_name


def estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty(
    reference_flux: Any,
    channel_flux: Any,
    *,
    reference_error: Any | None,
    channel_error: Any,
    initial_offset: float,
    initial_scale: float,
) -> InstrumentChannelCalibrationScaleDependentUncertaintyEstimate:
    """Estimate affine covariance with scale-dependent channel-axis errors.

    Inputs are the caller-selected final inlier set. The estimator minimizes

    ``0.5 * sum(log(v_i) + residual_i**2 / v_i)``

    with

    ``v_i = reference_error_i**2 + scale**2 * channel_error_i**2``.

    Optimization uses an analytic gradient and a log-scale parameterization
    that enforces a strictly positive scale. Available coefficient covariance
    is the inverse analytic observed Hessian of the full objective in fixed
    coefficient order ``offset, scale``.
    """

    uncertainty_source = (
        "pgmuvi_scale_dependent_full_objective_final_inliers"
    )
    optimizer_name = (
        "scipy.optimize.minimize:"
        "L-BFGS-B_log_scale_parameterization"
    )
    gradient_method = "analytic_full_objective_gradient"
    hessian_method = "analytic_observed_hessian"

    reference = _as_calibration_vector(
        reference_flux,
        name="reference_flux",
    )
    target = _as_calibration_vector(
        channel_flux,
        name="channel_flux",
    )

    if reference.shape != target.shape:
        raise ValueError(
            "reference_flux and channel_flux must have the same shape."
        )

    n_inliers = int(reference.size)
    if n_inliers < 3:
        raise ValueError(
            "Scale-dependent calibration uncertainty estimation requires "
            "at least 3 paired final inliers."
        )

    reference_sigma = _validate_optional_calibration_error(
        reference_error,
        name="reference_error",
        expected_shape=reference.shape,
    )
    channel_sigma = _validate_optional_calibration_error(
        channel_error,
        name="channel_error",
        expected_shape=reference.shape,
    )

    if channel_sigma is None:
        raise ValueError(
            "channel_error is required for scale-dependent calibration "
            "uncertainty estimation."
        )

    arrays = [reference, target, channel_sigma]
    if reference_sigma is not None:
        arrays.append(reference_sigma)

    if any(np.any(~np.isfinite(array)) for array in arrays):
        raise ValueError(
            "Scale-dependent calibration uncertainty inputs must contain "
            "only finite values."
        )

    if np.any(channel_sigma <= 0.0):
        raise ValueError(
            "channel_error must contain finite, strictly positive values."
        )
    if reference_sigma is not None and np.any(reference_sigma <= 0.0):
        raise ValueError(
            "reference_error must contain finite, strictly positive values "
            "when supplied."
        )

    if float(np.ptp(target)) <= 0.0:
        raise ValueError(
            "channel_flux must span more than one finite value."
        )

    for name, value in (
        ("initial_offset", initial_offset),
        ("initial_scale", initial_scale),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be numeric, not boolean.")

    normalized_initial_offset = float(initial_offset)
    normalized_initial_scale = float(initial_scale)

    if not math.isfinite(normalized_initial_offset):
        raise ValueError("initial_offset must be finite.")
    if (
        not math.isfinite(normalized_initial_scale)
        or normalized_initial_scale <= 0.0
    ):
        raise ValueError(
            "initial_scale must be finite and strictly positive."
        )

    reference_variance = (
        np.zeros(n_inliers, dtype=float)
        if reference_sigma is None
        else reference_sigma**2
    )
    channel_variance = channel_sigma**2

    minimum_channel_error = float(np.min(channel_sigma))
    maximum_channel_error = float(np.max(channel_sigma))
    maximum_target_magnitude = max(
        1.0,
        float(np.max(np.abs(target))),
    )

    log_scale_lower = (
        0.5 * math.log(np.finfo(float).tiny)
        - math.log(minimum_channel_error)
    )
    log_scale_upper = min(
        (
            0.5 * math.log(np.finfo(float).max / 4.0)
            - math.log(maximum_channel_error)
        ),
        (
            math.log(np.finfo(float).max / 4.0)
            - math.log(maximum_target_magnitude)
        ),
    )

    initial_log_scale = math.log(normalized_initial_scale)
    if not log_scale_lower < log_scale_upper:
        raise ValueError(
            "Input scales do not admit a numerically safe positive-scale "
            "optimization domain."
        )

    initial_log_scale = min(
        max(initial_log_scale, log_scale_lower),
        log_scale_upper,
    )

    def unavailable(
        reason: str,
    ) -> InstrumentChannelCalibrationScaleDependentUncertaintyEstimate:
        return InstrumentChannelCalibrationScaleDependentUncertaintyEstimate(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_SCALE_DEPENDENT_UNCERTAINTY_SCHEMA_VERSION
            ),
            status=(
                InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE
            ),
            uncertainty_source=uncertainty_source,
            n_inliers=n_inliers,
            optimizer=optimizer_name,
            optimizer_converged=False,
            gradient_method=gradient_method,
            hessian_method=hessian_method,
            reason=reason,
        )

    def objective_and_gradient(
        parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        offset = float(parameters[0])
        scale = math.exp(float(parameters[1]))
        residual = reference - (offset + scale * target)
        variance = reference_variance + scale**2 * channel_variance

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            objective_value = 0.5 * float(
                np.sum(
                    np.log(variance)
                    + residual**2 / variance
                )
            )
            gradient_offset = float(
                np.sum(-residual / variance)
            )
            gradient_scale = float(
                np.sum(
                    -residual * target / variance
                    + scale
                    * channel_variance
                    * (
                        1.0 / variance
                        - residual**2 / variance**2
                    )
                )
            )

        transformed_gradient = np.asarray(
            [
                gradient_offset,
                gradient_scale * scale,
            ],
            dtype=float,
        )

        if (
            not math.isfinite(objective_value)
            or np.any(~np.isfinite(transformed_gradient))
        ):
            return (
                float(np.finfo(float).max),
                np.zeros(2, dtype=float),
            )

        return objective_value, transformed_gradient

    try:
        result = optimize.minimize(
            objective_and_gradient,
            np.asarray(
                [
                    normalized_initial_offset,
                    initial_log_scale,
                ],
                dtype=float,
            ),
            method="L-BFGS-B",
            jac=True,
            bounds=(
                (None, None),
                (log_scale_lower, log_scale_upper),
            ),
            options={
                "ftol": 1.0e-12,
                "gtol": 1.0e-8,
                "maxiter": 1000,
                "maxls": 50,
            },
        )
    except Exception as exc:
        return unavailable(
            "Scale-dependent calibration optimization raised "
            f"{type(exc).__name__}: {exc}"
        )

    if (
        not bool(result.success)
        or np.asarray(result.x).shape != (2,)
        or np.any(~np.isfinite(result.x))
    ):
        message = str(getattr(result, "message", "")).strip()
        if not message:
            message = "no optimizer diagnostic was supplied"
        return unavailable(
            "Scale-dependent calibration optimization did not converge: "
            f"{message}."
        )

    offset = float(result.x[0])
    log_scale = float(result.x[1])
    scale = math.exp(log_scale)

    boundary_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(log_scale),
        abs(log_scale_lower),
        abs(log_scale_upper),
    )
    if (
        log_scale <= log_scale_lower + boundary_tolerance
        or log_scale >= log_scale_upper - boundary_tolerance
    ):
        return unavailable(
            "Scale-dependent calibration optimization converged on a "
            "numerical log-scale boundary."
        )

    residual = reference - (offset + scale * target)
    variance = reference_variance + scale**2 * channel_variance

    if (
        not math.isfinite(offset)
        or not math.isfinite(scale)
        or scale <= 0.0
        or np.any(~np.isfinite(residual))
        or np.any(~np.isfinite(variance) | (variance <= 0.0))
    ):
        return unavailable(
            "Scale-dependent calibration optimization produced invalid "
            "coefficients, residuals, or effective variances."
        )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        objective_value = 0.5 * float(
            np.sum(
                np.log(variance)
                + residual**2 / variance
            )
        )

        gradient_offset = float(
            np.sum(-residual / variance)
        )
        gradient_scale = float(
            np.sum(
                -residual * target / variance
                + scale
                * channel_variance
                * (
                    1.0 / variance
                    - residual**2 / variance**2
                )
            )
        )
        gradient = np.asarray(
            [gradient_offset, gradient_scale],
            dtype=float,
        )

        hessian_offset_offset = float(
            np.sum(1.0 / variance)
        )
        hessian_offset_scale = float(
            np.sum(
                target / variance
                + 2.0
                * scale
                * channel_variance
                * residual
                / variance**2
            )
        )
        hessian_scale_scale = float(
            np.sum(
                target**2 / variance
                + channel_variance / variance
                - 2.0
                * scale**2
                * channel_variance**2
                / variance**2
                - channel_variance
                * residual**2
                / variance**2
                + 4.0
                * scale
                * channel_variance
                * residual
                * target
                / variance**2
                + 4.0
                * scale**2
                * channel_variance**2
                * residual**2
                / variance**3
            )
        )

    hessian = np.asarray(
        [
            [hessian_offset_offset, hessian_offset_scale],
            [hessian_offset_scale, hessian_scale_scale],
        ],
        dtype=float,
    )
    hessian = 0.5 * (hessian + hessian.T)
    gradient_norm = float(np.linalg.norm(gradient, ord=2))

    if (
        not math.isfinite(objective_value)
        or not math.isfinite(gradient_norm)
        or np.any(~np.isfinite(hessian))
    ):
        return unavailable(
            "Scale-dependent calibration objective, gradient, or observed "
            "Hessian contains non-finite values."
        )

    try:
        hessian_eigenvalues = np.linalg.eigvalsh(hessian)
    except np.linalg.LinAlgError:
        return unavailable(
            "Observed Hessian eigendecomposition failed."
        )

    if (
        hessian_eigenvalues.shape != (2,)
        or np.any(~np.isfinite(hessian_eigenvalues))
        or float(hessian_eigenvalues[0]) <= 0.0
    ):
        return unavailable(
            "Observed Hessian is not finite and positive definite."
        )

    largest_eigenvalue = float(hessian_eigenvalues[-1])
    smallest_eigenvalue = float(hessian_eigenvalues[0])
    if (
        smallest_eigenvalue / largest_eigenvalue
        <= math.sqrt(np.finfo(float).eps)
    ):
        return unavailable(
            "Observed Hessian is numerically singular."
        )

    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return unavailable(
            "Observed Hessian inversion failed."
        )

    covariance = 0.5 * (covariance + covariance.T)
    if np.any(~np.isfinite(covariance)):
        return unavailable(
            "Inverse observed-Hessian covariance contains non-finite values."
        )

    return InstrumentChannelCalibrationScaleDependentUncertaintyEstimate(
        schema_version=(
            INSTRUMENT_CHANNEL_CALIBRATION_SCALE_DEPENDENT_UNCERTAINTY_SCHEMA_VERSION
        ),
        status=InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE,
        uncertainty_source=uncertainty_source,
        n_inliers=n_inliers,
        coefficient_covariance=(
            (
                float(covariance[0, 0]),
                float(covariance[0, 1]),
            ),
            (
                float(covariance[1, 0]),
                float(covariance[1, 1]),
            ),
        ),
        offset=offset,
        scale=scale,
        objective_value=objective_value,
        optimizer=optimizer_name,
        optimizer_converged=True,
        gradient_method=gradient_method,
        gradient_norm=gradient_norm,
        hessian_method=hessian_method,
        hessian_eigenvalues=(
            smallest_eigenvalue,
            largest_eigenvalue,
        ),
    )


def fit_instrument_channel_calibration(
    reference_flux: Any,
    channel_flux: Any,
    *,
    reference_channel: str,
    channel: str,
    wavelength: float,
    reference_error: Any | None = None,
    channel_error: Any | None = None,
    sigma_clip: float = 3.5,
    max_iter: int = 8,
    min_pairs: int = 3,
) -> InstrumentChannelCalibration:
    """Fit an explicit paired affine instrument-channel calibration.

    Parameters
    ----------
    reference_flux, channel_flux
        One-dimensional arrays containing measurements already paired by
        the caller. PGMUVI does not infer simultaneity or construct pairs.
    reference_channel, channel
        Distinct observational-channel identifiers.
    wavelength
        Shared finite wavelength coordinate for the two channels.
    reference_error, channel_error
        Optional one-sigma measurement uncertainties. When supplied, they
        contribute to iterative weighted least squares.
    sigma_clip
        Positive MAD-based clipping threshold.
    max_iter
        Maximum number of fitting and clipping iterations.
    min_pairs
        Minimum number of finite paired measurements required.

    Returns
    -------
    InstrumentChannelCalibration
        An immutable affine mapping satisfying
        ``reference_flux = offset + scale * channel_flux``.

    Notes
    -----
    Coefficient covariance is conditioned on the final MAD-clipped inlier set.
    No-error and reference-error-only fits use the final fixed-weight affine
    solve. Fits with channel-axis errors activate the scale-dependent Gaussian
    full objective so the reported offset, scale, and inverse observed-Hessian
    covariance share one optimum. If that optimization is unavailable, the
    iterative scale-frozen affine estimate is retained with explicit fallback
    provenance and unavailable coefficient covariance.
    """

    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int):
        raise TypeError("min_pairs must be an integer.")
    if min_pairs < 3:
        raise ValueError("min_pairs must be at least 3.")

    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise TypeError("max_iter must be an integer.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    sigma_clip = float(sigma_clip)
    if not np.isfinite(sigma_clip) or sigma_clip <= 0.0:
        raise ValueError("sigma_clip must be finite and positive.")

    reference = _as_calibration_vector(
        reference_flux,
        name="reference_flux",
    )
    target = _as_calibration_vector(
        channel_flux,
        name="channel_flux",
    )

    if reference.shape != target.shape:
        raise ValueError(
            "reference_flux and channel_flux must have the same shape."
        )

    n_input_pairs = int(reference.size)

    reference_sigma = _validate_optional_calibration_error(
        reference_error,
        name="reference_error",
        expected_shape=reference.shape,
    )
    channel_sigma = _validate_optional_calibration_error(
        channel_error,
        name="channel_error",
        expected_shape=reference.shape,
    )

    finite = np.isfinite(reference) & np.isfinite(target)

    for error in (reference_sigma, channel_sigma):
        if error is None:
            continue
        finite &= np.isfinite(error) & (error > 0.0)

    finite_pair_indices_array = np.flatnonzero(finite)
    reference = reference[finite]
    target = target[finite]

    if reference_sigma is not None:
        reference_sigma = reference_sigma[finite]
    if channel_sigma is not None:
        channel_sigma = channel_sigma[finite]

    n_pairs = int(reference.size)
    if n_pairs < min_pairs:
        raise ValueError(
            "Insufficient finite paired measurements: "
            f"received {n_pairs}, require at least {min_pairs}."
        )

    if not np.isfinite(float(wavelength)):
        raise ValueError("wavelength must be finite.")

    normalized_reference_channel = _normalize_calibration_channel(
        reference_channel,
        name="reference_channel",
    )
    normalized_channel = _normalize_calibration_channel(
        channel,
        name="channel",
    )

    if normalized_reference_channel == normalized_channel:
        raise ValueError(
            "reference_channel and channel must identify different "
            "observational channels."
        )

    target_span = float(np.ptp(target))
    if not np.isfinite(target_span) or target_span <= 0.0:
        raise ValueError(
            "channel_flux must span more than one finite value."
        )

    keep = np.ones(n_pairs, dtype=bool)
    scale_for_weights = 1.0
    offset = 0.0
    scale = 1.0

    for _ in range(max_iter):
        if np.count_nonzero(keep) < min_pairs:
            break

        if reference_sigma is None and channel_sigma is None:
            weights = np.ones(np.count_nonzero(keep), dtype=float)
        else:
            variance = np.zeros(n_pairs, dtype=float)

            if reference_sigma is not None:
                variance += reference_sigma**2
            if channel_sigma is not None:
                variance += (
                    scale_for_weights * channel_sigma
                ) ** 2

            selected_variance = variance[keep]
            if np.any(
                ~np.isfinite(selected_variance)
                | (selected_variance <= 0.0)
            ):
                raise ValueError(
                    "Combined calibration variances must be finite "
                    "and positive."
                )

            weights = 1.0 / selected_variance

        offset, scale = _weighted_affine_solution(
            target[keep],
            reference[keep],
            weights,
        )

        if not np.isfinite(offset) or not np.isfinite(scale):
            raise RuntimeError(
                "Instrument-channel calibration fit produced "
                "non-finite coefficients."
            )
        if scale <= 0.0:
            raise ValueError(
                "Instrument-channel calibration requires a strictly "
                "positive fitted scale."
            )

        scale_for_weights = scale
        residual = reference - (offset + scale * target)
        selected_residual = residual[keep]
        centre = float(np.median(selected_residual))
        residual_scale = _calibration_mad_sigma(
            selected_residual
        )

        absolute_deviation = np.abs(residual - centre)
        numerical_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(
                1.0,
                float(np.max(np.abs(reference))),
                float(
                    np.max(
                        np.abs(offset + scale * target)
                    )
                ),
            )
        )

        if residual_scale <= numerical_tolerance:
            updated = (
                absolute_deviation <= numerical_tolerance
            )

            if (
                np.count_nonzero(updated) < min_pairs
                or float(np.ptp(target[updated])) <= 0.0
            ):
                positive_deviation = absolute_deviation[
                    absolute_deviation > numerical_tolerance
                ]

                if positive_deviation.size == 0:
                    break

                fallback_scale = float(
                    np.min(positive_deviation)
                )
                updated = (
                    absolute_deviation
                    <= sigma_clip * fallback_scale
                )
        else:
            updated = (
                absolute_deviation
                <= sigma_clip * residual_scale
            )

        if np.count_nonzero(updated) < min_pairs:
            break
        if np.array_equal(updated, keep):
            keep = updated
            break

        keep = updated

    n_inliers = int(np.count_nonzero(keep))
    if n_inliers < min_pairs:
        raise RuntimeError(
            "Instrument-channel calibration clipping left fewer "
            "than the required number of paired measurements."
        )

    if reference_sigma is None and channel_sigma is None:
        final_weights = np.ones(n_inliers, dtype=float)
    else:
        final_variance = np.zeros(n_pairs, dtype=float)

        if reference_sigma is not None:
            final_variance += reference_sigma**2
        if channel_sigma is not None:
            final_variance += (scale * channel_sigma) ** 2

        final_weights = 1.0 / final_variance[keep]

    offset, scale = _weighted_affine_solution(
        target[keep],
        reference[keep],
        final_weights,
    )

    if scale <= 0.0:
        raise ValueError(
            "Instrument-channel calibration requires a strictly "
            "positive fitted scale."
        )

    final_residual = (
        reference[keep]
        - (offset + scale * target[keep])
    )
    selected_uncertainty_estimator = (
        select_instrument_channel_calibration_uncertainty_estimator(
            reference_error_supplied=reference_sigma is not None,
            channel_error_supplied=channel_sigma is not None,
        )
    )

    if channel_sigma is None:
        coefficient_uncertainty = _estimate_affine_coefficient_uncertainty(
            target[keep],
            final_residual,
            final_weights,
            measurement_errors_supplied=(
                reference_sigma is not None
            ),
            channel_errors_supplied=False,
        )
        integration_status = (
            InstrumentChannelCalibrationUncertaintyIntegrationStatus.ACTIVE
        )
        point_estimate_source = "pgmuvi_affine_fit_final_inliers"
        point_estimate_objective = "fixed_weight_least_squares"
        point_estimate_matches_uncertainty_objective = True
        integration_reason = None
    else:
        scale_dependent_estimate = (
            estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty(
                reference[keep],
                target[keep],
                reference_error=(
                    None
                    if reference_sigma is None
                    else reference_sigma[keep]
                ),
                channel_error=channel_sigma[keep],
                initial_offset=offset,
                initial_scale=scale,
            )
        )

        if (
            scale_dependent_estimate.status
            is InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE
        ):
            offset = float(scale_dependent_estimate.offset)
            scale = float(scale_dependent_estimate.scale)
            final_residual = (
                reference[keep]
                - (offset + scale * target[keep])
            )
            coefficient_uncertainty = (
                InstrumentChannelCalibrationCoefficientUncertainty(
                    schema_version=(
                        INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION
                    ),
                    status=(
                        InstrumentChannelCalibrationUncertaintyStatus.AVAILABLE
                    ),
                    uncertainty_source=(
                        scale_dependent_estimate.uncertainty_source
                    ),
                    coefficient_covariance=(
                        scale_dependent_estimate.coefficient_covariance
                    ),
                    estimation_method=(
                        "inverse_observed_hessian_"
                        "scale_dependent_full_objective"
                    ),
                )
            )
            integration_status = (
                InstrumentChannelCalibrationUncertaintyIntegrationStatus.ACTIVE
            )
            point_estimate_source = (
                "pgmuvi_scale_dependent_full_objective_final_inliers"
            )
            point_estimate_objective = (
                "gaussian_negative_log_likelihood_"
                "scale_dependent_effective_variance"
            )
            point_estimate_matches_uncertainty_objective = True
            integration_reason = None
        else:
            estimator_reason = scale_dependent_estimate.reason
            integration_reason = (
                "Scale-dependent full-objective optimization was attempted "
                "on the final inlier set but coefficient uncertainty is "
                f"unavailable: {estimator_reason}"
            )
            coefficient_uncertainty = (
                InstrumentChannelCalibrationCoefficientUncertainty(
                    schema_version=(
                        INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION
                    ),
                    status=(
                        InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE
                    ),
                    uncertainty_source=(
                        scale_dependent_estimate.uncertainty_source
                    ),
                    coefficient_covariance=None,
                    reason=integration_reason,
                )
            )
            integration_status = (
                InstrumentChannelCalibrationUncertaintyIntegrationStatus
                .ATTEMPTED_UNAVAILABLE_FALLBACK
            )
            point_estimate_source = (
                "pgmuvi_iterative_mad_clipped_affine_fallback_final_inliers"
            )
            point_estimate_objective = (
                "iterative_scale_frozen_weighted_least_squares"
            )
            point_estimate_matches_uncertainty_objective = False

    fit_provenance = InstrumentChannelCalibrationFitProvenance(
        schema_version=(
            INSTRUMENT_CHANNEL_CALIBRATION_FIT_PROVENANCE_SCHEMA_VERSION
        ),
        selected_uncertainty_estimator=selected_uncertainty_estimator,
        integration_status=integration_status,
        reference_error_supplied=reference_sigma is not None,
        channel_error_supplied=channel_sigma is not None,
        n_input_pairs=n_input_pairs,
        finite_pair_indices=tuple(
            int(index) for index in finite_pair_indices_array
        ),
        final_inlier_indices=tuple(
            int(index) for index in finite_pair_indices_array[keep]
        ),
        point_estimate_source=point_estimate_source,
        point_estimate_objective=point_estimate_objective,
        point_estimate_matches_uncertainty_objective=(
            point_estimate_matches_uncertainty_objective
        ),
        reason=integration_reason,
    )

    return InstrumentChannelCalibration(
        schema_version=(
            INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION
        ),
        reference_channel=normalized_reference_channel,
        channel=normalized_channel,
        wavelength=float(wavelength),
        offset=float(offset),
        scale=float(scale),
        n_pairs=n_pairs,
        n_inliers=n_inliers,
        residual_mad_sigma=_calibration_mad_sigma(
            final_residual
        ),
        coefficient_uncertainty=coefficient_uncertainty,
        fit_provenance=fit_provenance,
    )


def apply_instrument_channel_calibration(
    flux: Any,
    calibration: InstrumentChannelCalibration,
    *,
    flux_error: Any | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Map one observational channel onto its reference-channel scale.

    Measurement uncertainties are multiplied by ``abs(scale)``. This
    propagation does not include uncertainty in the fitted calibration
    coefficients.
    """

    if not isinstance(
        calibration,
        InstrumentChannelCalibration,
    ):
        raise TypeError(
            "calibration must be an "
            "InstrumentChannelCalibration instance."
        )

    values = np.asarray(flux, dtype=float)
    calibrated_flux = (
        calibration.offset + calibration.scale * values
    )

    if flux_error is None:
        return calibrated_flux

    errors = np.asarray(flux_error, dtype=float)

    if errors.shape != values.shape:
        raise ValueError(
            "flux_error must have the same shape as flux."
        )
    if np.any(~np.isfinite(errors) | (errors < 0.0)):
        raise ValueError(
            "flux_error values must be finite and non-negative."
        )

    calibrated_error = abs(calibration.scale) * errors
    return calibrated_flux, calibrated_error



def apply_instrument_channel_calibration_with_predictive_uncertainty(
    flux: Any,
    calibration: InstrumentChannelCalibration,
    *,
    flux_error: Any | None = None,
    covariance_mode: (
        InstrumentChannelCalibrationPredictiveCovarianceMode | str
    ),
) -> tuple[
    np.ndarray,
    InstrumentChannelCalibrationPredictiveUncertainty,
]:
    """Apply affine calibration and propagate predictive uncertainty.

    The calibrated flux is identical to
    :func:`apply_instrument_channel_calibration`. Independent input
    measurement errors contribute ``scale**2 * flux_error_i**2`` only on the
    predictive-covariance diagonal. Fitted coefficient covariance contributes
    ``J_i C J_j.T`` with ``J_i = [1, x_i]`` in fixed coefficient order
    ``offset, scale``.

    ``marginal_variance`` returns only flattened row-major variance
    components. ``full_covariance`` additionally returns the complete
    covariance between calibrated predictions sharing the fitted
    coefficients. When coefficient covariance is unavailable, the result is
    explicitly unavailable and contains no numerical variance components;
    measurement-only fallback is not performed.
    """

    if isinstance(
        covariance_mode,
        InstrumentChannelCalibrationPredictiveCovarianceMode,
    ):
        normalized_mode = covariance_mode
    else:
        try:
            normalized_mode = (
                InstrumentChannelCalibrationPredictiveCovarianceMode(
                    str(covariance_mode)
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Unsupported predictive covariance mode: "
                f"{covariance_mode!r}."
            ) from exc

    applied = apply_instrument_channel_calibration(
        flux,
        calibration,
        flux_error=flux_error,
    )
    input_flux = np.asarray(flux, dtype=float)

    if flux_error is None:
        calibrated_flux = np.asarray(applied, dtype=float)
        measurement_variance = np.zeros(
            calibrated_flux.size,
            dtype=float,
        )
    else:
        calibrated_flux_value, calibrated_error = applied
        calibrated_flux = np.asarray(calibrated_flux_value, dtype=float)
        measurement_variance = np.square(
            np.asarray(calibrated_error, dtype=float).reshape(
                -1,
                order="C",
            )
        )

    input_shape = tuple(int(item) for item in calibrated_flux.shape)
    flattened_flux = input_flux.reshape(-1, order="C")
    coefficient_uncertainty = calibration.coefficient_uncertainty

    if coefficient_uncertainty.status is (
        InstrumentChannelCalibrationUncertaintyStatus.UNAVAILABLE
    ):
        return (
            calibrated_flux,
            InstrumentChannelCalibrationPredictiveUncertainty(
                schema_version=(
                    INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_UNCERTAINTY_SCHEMA_VERSION
                ),
                status=(
                    InstrumentChannelCalibrationPredictiveUncertaintyStatus.UNAVAILABLE
                ),
                covariance_mode=normalized_mode,
                input_shape=input_shape,
                input_measurement_uncertainty_supplied=(
                    flux_error is not None
                ),
                coefficient_uncertainty_status=(
                    coefficient_uncertainty.status
                ),
                coefficient_uncertainty_source=(
                    coefficient_uncertainty.uncertainty_source
                ),
                measurement_variance=None,
                offset_variance=None,
                scale_variance=None,
                offset_scale_covariance_term=None,
                predictive_covariance=None,
                reason=coefficient_uncertainty.reason,
            ),
        )

    coefficient_covariance = np.asarray(
        coefficient_uncertainty.coefficient_covariance,
        dtype=float,
    )
    offset_variance = np.full(
        flattened_flux.size,
        coefficient_covariance[0, 0],
        dtype=float,
    )
    scale_variance = (
        np.square(flattened_flux) * coefficient_covariance[1, 1]
    )
    offset_scale_covariance_term = (
        2.0 * flattened_flux * coefficient_covariance[0, 1]
    )

    predictive_covariance = None
    if normalized_mode is (
        InstrumentChannelCalibrationPredictiveCovarianceMode.FULL_COVARIANCE
    ):
        design = np.column_stack(
            (
                np.ones(flattened_flux.size, dtype=float),
                flattened_flux,
            )
        )
        predictive_covariance_array = (
            design @ coefficient_covariance @ design.T
        )
        predictive_covariance_array = 0.5 * (
            predictive_covariance_array
            + predictive_covariance_array.T
        )
        diagonal = np.diag_indices_from(predictive_covariance_array)
        predictive_covariance_array[diagonal] += measurement_variance
        predictive_covariance = predictive_covariance_array

    return (
        calibrated_flux,
        InstrumentChannelCalibrationPredictiveUncertainty(
            schema_version=(
                INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_UNCERTAINTY_SCHEMA_VERSION
            ),
            status=(
                InstrumentChannelCalibrationPredictiveUncertaintyStatus.AVAILABLE
            ),
            covariance_mode=normalized_mode,
            input_shape=input_shape,
            input_measurement_uncertainty_supplied=(flux_error is not None),
            coefficient_uncertainty_status=coefficient_uncertainty.status,
            coefficient_uncertainty_source=(
                coefficient_uncertainty.uncertainty_source
            ),
            measurement_variance=measurement_variance,
            offset_variance=offset_variance,
            scale_variance=scale_variance,
            offset_scale_covariance_term=(
                offset_scale_covariance_term
            ),
            predictive_covariance=predictive_covariance,
        ),
    )
