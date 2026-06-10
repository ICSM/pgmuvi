"""Model-independent parameter specification schema.

This module defines lightweight containers for describing GP model parameters
in data/physical space. It does not apply values to GPyTorch models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

import numpy as np


class ParameterRole(str, Enum):
    PERIOD = "period"
    FREQUENCY = "frequency"
    AMPLITUDE = "amplitude"
    OFFSET = "offset"
    TIMESCALE = "timescale"
    LENGTHSCALE = "lengthscale"
    WEIGHT = "weight"
    NOISE = "noise"
    SHAPE = "shape"
    WAVELENGTH_SCALE = "wavelength_scale"
    OTHER = "other"


class ParameterDomain(str, Enum):
    TIME = "time"
    FREQUENCY = "frequency"
    FLUX = "flux"
    WAVELENGTH = "wavelength"
    VARIANCE = "variance"
    DIMENSIONLESS = "dimensionless"
    OTHER = "other"


class ParameterScale(str, Enum):
    LINEAR = "linear"
    LOG = "log"
    LOG10 = "log10"
    LOGIT = "logit"
    OTHER = "other"


class GuessStrategy(str, Enum):
    USER = "user"
    DEFAULT = "default"

    MEDIAN_FLUX = "median_flux"
    ROBUST_FLUX_RANGE = "robust_flux_range"
    ROBUST_FLUX_SPAN = "robust_flux_span"
    FLUX_STD = "flux_std"
    MAD = "mad"

    BASELINE = "baseline"
    CADENCE = "cadence"

    LS_PERIOD = "ls_period"
    ACF_PERIOD = "acf_period"
    CONSENSUS_PERIOD = "consensus_period"
    CONSENSUS_MULTICOMP_PERIOD = "consensus_multicomp_period"

    VARIABILITY_TIMESCALE = "variability_timescale"

    WAVELENGTH_RANGE = "wavelength_range"

    CUSTOM = "custom"

class ConstraintStrategy(str, Enum):
    USER = "user"
    DEFAULT = "default"

    MEDIAN_FLUX = "median_flux"
    ROBUST_FLUX_RANGE = "robust_flux_range"
    ROBUST_POSITIVE_FLUX_SPAN = "robust_positive_flux_span"
    ROBUST_FLUX_SPAN = "robust_flux_span"
    FLUX_STD = "flux_std"
    MAD = "mad"

    BASELINE = "baseline"
    CADENCE = "cadence"

    LS_PERIOD = "ls_period"
    ACF_PERIOD = "acf_period"
    CONSENSUS_PERIOD = "consensus_period"
    CONSENSUS_MULTICOMP_PERIOD = "consensus_multicomp_period"

    VARIABILITY_TIMESCALE = "variability_timescale"

    WAVELENGTH_RANGE = "wavelength_range"

    CUSTOM = "custom"


@dataclass(frozen=True)
class ParameterSpec:
    """Description of one trainable model parameter in data/physical space.

    The specification describes the parameter semantically and records
    how initial values and constraints should be constructed. It does
    not contain any model-specific transformation logic.
    """

    name: str
    role: ParameterRole
    domain: ParameterDomain
    scale: ParameterScale = ParameterScale.LINEAR

    required: bool = True
    trainable: bool = True

    initial_value: Any | None = None
    constraint: tuple[Any, Any] | None = None
    shape: tuple[int, ...] | None = None

    units: str | None = None
    description: str | None = None

    guess_strategy: GuessStrategy | None = None
    constraint_strategy: ConstraintStrategy | None = None

    guess_source: str | None = None
    constraint_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate internal consistency of this parameter specification."""
        if not self.name:
            raise ValueError("ParameterSpec.name must be non-empty.")

        if self.constraint is not None:
            lower, upper = self.constraint
            lower_arr = np.asarray(lower)
            upper_arr = np.asarray(upper)

            if np.any(lower_arr >= upper_arr):
                raise ValueError(
                    f"Invalid constraint for {self.name!r}: lower bound must be "
                    "strictly smaller than upper bound."
                )

            if self.initial_value is not None:
                value_arr = np.asarray(self.initial_value)
                if np.any(value_arr < lower_arr) or np.any(value_arr > upper_arr):
                    raise ValueError(
                        f"Initial value for {self.name!r} lies outside its constraint."
                    )

        if self.shape is not None and self.initial_value is not None:
            value_shape = np.asarray(self.initial_value).shape
            if value_shape != self.shape:
                raise ValueError(
                    f"Shape mismatch for {self.name!r}: expected {self.shape}, "
                    f"got {value_shape}."
                )


@dataclass
class ParameterSpecCollection:
    """Container for a set of named parameter specifications."""

    specs: list[ParameterSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def __iter__(self) -> Iterator[ParameterSpec]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, name: str) -> bool:
        return any(spec.name == name for spec in self.specs)

    def __getitem__(self, name: str) -> ParameterSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def validate(self) -> None:
        names = self.names()
        if len(names) != len(set(names)):
            raise ValueError("ParameterSpecCollection contains duplicate names.")

        for spec in self.specs:
            spec.validate()

    def add(self, spec: ParameterSpec) -> None:
        if spec.name in self:
            raise ValueError(f"Duplicate parameter specification: {spec.name!r}")
        spec.validate()
        self.specs.append(spec)

    def as_dict(self) -> dict[str, ParameterSpec]:
        return {spec.name: spec for spec in self.specs}
