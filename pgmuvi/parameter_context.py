"""Diagnostic containers for parameter estimation.

This module defines model-independent containers for data-derived
diagnostics that may later be used to build parameter guesses and
constraints. It does not compute diagnostics or apply values to models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LightcurveDiagnostics:
    """Global diagnostics for a light curve or light-curve collection."""

    baseline: float | None = None
    cadence: float | None = None
    median_flux: float | None = None
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
    metadata: dict[str, Any] = field(default_factory=dict)


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
