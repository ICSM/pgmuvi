"""Data-derived parameter estimates.

This module defines containers for parameter values and constraints
estimated from data in physical space. It does not transform values into
GPyTorch parameter space or apply them to models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from pgmuvi.parameter_specs import ParameterSpec


@dataclass(frozen=True)
class ParameterEstimate:
    """Data-derived estimate for one parameter specification."""

    spec: ParameterSpec

    value: Any | None = None
    constraint: tuple[Any, Any] | None = None

    value_source: str | None = None
    constraint_source: str | None = None

    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Return the parameter name from the associated specification."""
        return self.spec.name

    def validate(self) -> None:
        """Validate internal consistency of this parameter estimate."""
        self.spec.validate()

        if self.constraint is not None:
            lower, upper = self.constraint
            lower_arr = np.asarray(lower)
            upper_arr = np.asarray(upper)

            if np.any(lower_arr >= upper_arr):
                raise ValueError(
                    f"Invalid constraint for {self.name!r}: lower bound must be "
                    "strictly smaller than upper bound."
                )

            if self.value is not None:
                value_arr = np.asarray(self.value)
                if np.any(value_arr < lower_arr) or np.any(value_arr > upper_arr):
                    raise ValueError(
                        f"Estimated value for {self.name!r} lies outside its constraint."
                    )

        if self.spec.shape is not None and self.value is not None:
            value_shape = np.asarray(self.value).shape
            if value_shape != self.spec.shape:
                raise ValueError(
                    f"Shape mismatch for {self.name!r}: expected {self.spec.shape}, "
                    f"got {value_shape}."
                )


@dataclass
class ParameterEstimateCollection:
    """Container for a set of named parameter estimates."""

    estimates: list[ParameterEstimate] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def __iter__(self) -> Iterator[ParameterEstimate]:
        return iter(self.estimates)

    def __len__(self) -> int:
        return len(self.estimates)

    def __contains__(self, name: str) -> bool:
        return any(estimate.name == name for estimate in self.estimates)

    def __getitem__(self, name: str) -> ParameterEstimate:
        for estimate in self.estimates:
            if estimate.name == name:
                return estimate
        raise KeyError(name)

    def names(self) -> list[str]:
        """Return parameter names in collection order."""
        return [estimate.name for estimate in self.estimates]

    def validate(self) -> None:
        """Validate all estimates and reject duplicate names."""
        names = self.names()
        if len(names) != len(set(names)):
            raise ValueError("ParameterEstimateCollection contains duplicate names.")

        for estimate in self.estimates:
            estimate.validate()

    def add(self, estimate: ParameterEstimate) -> None:
        """Add one estimate after validating it."""
        if estimate.name in self:
            raise ValueError(f"Duplicate parameter estimate: {estimate.name!r}")
        estimate.validate()
        self.estimates.append(estimate)

    def as_dict(self) -> dict[str, ParameterEstimate]:
        """Return estimates keyed by parameter name."""
        return {estimate.name: estimate for estimate in self.estimates}