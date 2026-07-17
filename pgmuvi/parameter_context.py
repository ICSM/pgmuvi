"""Diagnostic containers for parameter estimation.

This module defines model-independent containers for data-derived
diagnostics that may later be used to build parameter guesses and
constraints. It does not compute diagnostics or apply values to models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LightcurveDiagnostics:
    """Global diagnostics for a light curve or light-curve collection."""

    baseline: float | None = None
    cadence: float | None = None
    median_flux: float | None = None
    baseline_duration: float | None = None
    median_cadence: float | None = None
    mad_flux: float | None = None
    flux_percentiles: dict[float, float] = field(default_factory=dict)
    n_points: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BandDiagnostics:
    """Diagnostics for one band in a multiband light curve."""

    band: str
    wavelength: float | None = None
    baseline: float | None = None
    cadence: float | None = None
    median_flux: float | None = None
    mad_flux: float | None = None
    flux_percentiles: dict[float, float] = field(default_factory=dict)
    n_points: int | None = None
    median_uncertainty: float | None = None
    uncertainty_percentiles: dict[float, float] = field(default_factory=dict)
    robust_scatter: float | None = None
    raw_half_amplitude_q02_5_q97_5: float | None = None
    raw_half_amplitude_q05_q95: float | None = None
    raw_half_amplitude_q10_q90: float | None = None
    fractional_half_amplitude_q02_5_q97_5: float | None = None
    fractional_half_amplitude_q05_q95: float | None = None
    fractional_half_amplitude_q10_q90: float | None = None
    noise_corrected_robust_scatter: float | None = None
    noise_corrected_half_amplitude_q02_5_q97_5: float | None = None
    noise_corrected_half_amplitude_q05_q95: float | None = None
    noise_corrected_half_amplitude_q10_q90: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


WAVELENGTH_ESTIMATION_SCHEMA_VERSION = "pgmuvi-wavelength-estimation-v2"


@dataclass(frozen=True)
class WavelengthEstimationDiagnostics:
    """Data-derived wavelength sampling and trend diagnostics.

    Raw wavelength scales are expressed in the same coordinate units as the
    second column of the input light curve.  The model-coordinate fields retain
    the corresponding values after the scale-only part of an input transform is
    applied, allowing separable wavelength kernels to consume the estimates
    without losing raw-coordinate provenance.
    """

    schema_version: str = WAVELENGTH_ESTIMATION_SCHEMA_VERSION
    available: bool = False
    coordinate_space: str = "raw_input"
    n_observations: int = 0
    n_distinct_wavelengths: int = 0
    n_usable_bands: int = 0
    min_points_per_band: int = 3
    wavelengths: tuple[float, ...] = ()
    wavelength_min: float | None = None
    wavelength_max: float | None = None
    wavelength_span: float | None = None
    adjacent_spacings: tuple[float, ...] = ()
    minimum_adjacent_spacing: float | None = None
    median_adjacent_spacing: float | None = None
    maximum_adjacent_spacing: float | None = None
    largest_gap: float | None = None
    largest_gap_ratio_to_median_spacing: float | None = None
    spacing_ratio_max_to_min: float | None = None
    coverage_class: str = "unavailable"
    median_flux_monotonicity_class: str = "unavailable"
    amplitude_monotonicity_class: str = "unavailable"
    scatter_monotonicity_class: str = "unavailable"
    median_flux_ratio_max_to_min_abs: float | None = None
    amplitude_ratio_max_to_min: float | None = None
    scatter_ratio_max_to_min: float | None = None
    recommended_lengthscale_initial: float | None = None
    recommended_lengthscale_bounds: tuple[float, float] | None = None
    recommendation_method: str | None = None
    model_coordinate_space: str = "raw_input"
    model_recommended_lengthscale_initial: float | None = None
    model_recommended_lengthscale_bounds: tuple[float, float] | None = None
    lengthscale_transform_status: str = "identity"
    lengthscale_transform_name: str | None = None
    excluded_bands: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)


WAVELENGTH_MEAN_ESTIMATION_SCHEMA_VERSION = "pgmuvi-wavelength-mean-estimation-v1"


@dataclass(frozen=True)
class WavelengthMeanEstimationDiagnostics:
    """Model-ready wavelength-mean initialization and constraint candidates.

    ``physical_wavelength`` recommendations use the raw positive wavelength
    coordinate together with the transformed GP target.  ``model_wavelength``
    recommendations use the wavelength coordinate actually supplied to the GP.
    This distinction keeps dust and power-law parameters physically interpretable
    while allowing the flexible quadratic mean to follow the fitted coordinate.
    """

    schema_version: str = WAVELENGTH_MEAN_ESTIMATION_SCHEMA_VERSION
    available: bool = False
    n_usable_bands: int = 0
    raw_wavelengths: tuple[float, ...] = ()
    model_wavelengths: tuple[float, ...] = ()
    model_median_fluxes: tuple[float, ...] = ()
    recommendations: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class ConsensusDiagnostics:
    """Period/frequency diagnostics from a cross-band consensus analysis."""

    method: str
    periods: Any | None = None
    frequencies: Any | None = None
    period_widths: Any | None = None
    frequency_widths: Any | None = None
    powers: Any | None = None
    component_indices: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParameterEstimationContext:
    """Container for diagnostics used by future parameter-estimation logic."""

    is_multiband: bool
    global_diagnostics: LightcurveDiagnostics | None = None
    band_diagnostics: dict[str, BandDiagnostics] = field(default_factory=dict)
    consensus_diagnostics: ConsensusDiagnostics | None = None
    wavelength_diagnostics: WavelengthEstimationDiagnostics | None = None
    wavelength_mean_diagnostics: WavelengthMeanEstimationDiagnostics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def bands(self) -> list[str]:
        """Return the available band names."""
        return list(self.band_diagnostics.keys())

    def get_band(self, band: str) -> BandDiagnostics:
        """Return diagnostics for one band."""
        try:
            return self.band_diagnostics[band]
        except KeyError as exc:
            raise KeyError(f"No diagnostics found for band {band!r}.") from exc
