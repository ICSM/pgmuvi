"""Contracts for future observational-channel calibration.

An observational channel identifies an instrument, detector, filter, or data
stream.  It is distinct from the numeric physical wavelength coordinate used by
the GP, and multiple observational channels may share one physical wavelength.

PGMUVI does not yet estimate or apply channel-specific offsets, scales,
throughput corrections, or noise corrections.  This module provides an
immutable, JSON-safe assessment of whether such calibration is required and
public placeholder callables that fail explicitly rather than silently
approximating a correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, NoReturn

import numpy as np


INSTRUMENT_CHANNEL_CALIBRATION_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER = (
    "TBD[instrument-channel-calibration]"
)


__all__ = [
    "INSTRUMENT_CHANNEL_CALIBRATION_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER",
    "InstrumentChannelCalibrationAssessment",
    "InstrumentChannelCalibrationStatus",
    "SharedWavelengthChannelGroup",
    "apply_instrument_channel_calibration",
    "assess_instrument_channel_calibration_requirement",
    "fit_instrument_channel_calibration",
]


class _StringEnum(str, Enum):
    """Enum whose members serialize as stable public strings."""

    def __str__(self) -> str:
        return self.value


class InstrumentChannelCalibrationStatus(_StringEnum):
    """Implementation state for an observational-channel calibration need."""

    NOT_REQUIRED = "not_required"
    REQUIRED_NOT_IMPLEMENTED = "required_not_implemented"


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
                "Instrument-channel calibration is not implemented."
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


def _raise_calibration_not_implemented(action: str) -> NoReturn:
    raise NotImplementedError(
        f"Instrument-channel calibration {action} is not implemented. "
        f"{INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER} requires an explicit, "
        "tested model; PGMUVI will not silently infer or apply a correction."
    )


def fit_instrument_channel_calibration(
    physical_wavelengths: Any,
    fluxes: Any,
    observational_channel_labels: Any,
    uncertainties: Any | None = None,
) -> NoReturn:
    """Fit a channel calibration model.

    This callable defines the public failure contract only.  No calibration
    model is currently implemented.
    """

    _raise_calibration_not_implemented("fitting")


def apply_instrument_channel_calibration(
    fluxes: Any,
    observational_channel_labels: Any,
    calibration: Any,
) -> NoReturn:
    """Apply a fitted channel calibration model.

    This callable defines the public failure contract only.  No calibration
    model is currently implemented.
    """

    _raise_calibration_not_implemented("application")
