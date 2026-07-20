"""Explicit observational-channel calibration primitives.

An observational channel identifies an instrument, detector, filter, or data
stream.  It is distinct from the numeric physical wavelength coordinate used by
the GP, and multiple observational channels may share one physical wavelength.

This module provides an immutable, JSON-safe requirement assessment together
with low-level fitting and application primitives for an explicit affine
mapping between caller-paired measurements.  It does not construct temporal
pairs, choose a calibration family, merge channels, alter wavelengths, or
integrate calibration automatically into a light-curve fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np


INSTRUMENT_CHANNEL_CALIBRATION_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER = (
    "TBD[instrument-channel-calibration]"
)


__all__ = [
    "INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER",
    "InstrumentChannelCalibration",
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


INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION = "1.0"


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
    fit_method: str = "iterative_mad_clipped_affine"

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
            "application_equation": (
                "reference_flux = offset + scale * channel_flux"
            ),
            "automatic_time_matching": False,
            "automatic_model_selection": False,
        }


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

        if residual_scale <= 0.0:
            break

        updated = (
            np.abs(residual - centre)
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
