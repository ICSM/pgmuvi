"""Explicit scientific-hypothesis metadata for wavelength GP models.

The public model strings used by PGMUVI describe complete GP configurations,
not isolated scientific mechanisms.  In particular, the LPV-focused
``2DDustMean`` and ``2DPowerLawMean`` models combine a wavelength-dependent
mean with the same family of smooth separable wavelength covariance used by
``2DWavelengthDependent``.  This module records those orthogonal mean and
covariance roles explicitly so advisory reports do not imply that a score
change can automatically be attributed to one mechanism.

The taxonomy is descriptive only.  It does not rank, select, instantiate, or
fit a model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

WAVELENGTH_HYPOTHESIS_SCHEMA_VERSION = "pgmuvi-wavelength-hypotheses-v1"

LPV_WAVELENGTH_MODEL_PRIORITY = (
    "2DWavelengthDependent",
    "2DDustMean",
    "2DPowerLawMean",
    "2DSeparable",
    "2D",
)

__all__ = [
    "LPV_WAVELENGTH_MODEL_PRIORITY",
    "WAVELENGTH_HYPOTHESIS_SCHEMA_VERSION",
    "WavelengthCovarianceStructure",
    "WavelengthHypothesisRole",
    "WavelengthMeanStructure",
    "WavelengthModelHypothesis",
    "describe_wavelength_model_hypothesis",
]


class _StringEnum(str, Enum):
    """Enum whose values serialize as stable public strings."""

    def __str__(self) -> str:
        return self.value


class WavelengthMeanStructure(_StringEnum):
    """How a complete model represents wavelength dependence in its mean."""

    CONSTANT = "constant"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    DUST_ATTENUATION = "dust_attenuation"
    POWER_LAW = "power_law"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class WavelengthCovarianceStructure(_StringEnum):
    """How a complete model represents cross-wavelength covariance."""

    JOINT_2D_SPECTRAL_MIXTURE = "joint_2d_spectral_mixture"
    SEPARABLE_PRODUCT = "separable_product"
    SEPARABLE_SMOOTH_WAVELENGTH = "separable_smooth_wavelength"
    ACHROMATIC_CONSTANT_WAVELENGTH = "achromatic_constant_wavelength"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class WavelengthHypothesisRole(_StringEnum):
    """Broad interpretive role of a complete wavelength-model configuration."""

    JOINT_BASELINE = "joint_baseline"
    COVARIANCE_FOCUSED = "covariance_focused"
    MEAN_AND_COVARIANCE = "mean_and_covariance"
    ACHROMATIC_CONTROL = "achromatic_control"
    UNKNOWN = "unknown"


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-safe copy of public metadata."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe(item) for item in value]
    return str(value)


def _coerce_enum(enum_type, value: Any, default):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError):
        return default


def _mean_structure_from_override(value: Any) -> WavelengthMeanStructure | None:
    """Interpret a user-visible ``mean_module`` override when possible."""
    if value is None:
        return None
    if not isinstance(value, str):
        return WavelengthMeanStructure.CUSTOM
    normalized = value.strip().lower()
    if normalized in {"constant", "constant_mean"}:
        return WavelengthMeanStructure.CONSTANT
    if normalized in {"linear", "linear_mean"}:
        return WavelengthMeanStructure.LINEAR
    if normalized in {"quad", "quadratic", "quad_constant"}:
        return WavelengthMeanStructure.QUADRATIC
    if normalized in {"dust", "dust_mean"}:
        return WavelengthMeanStructure.DUST_ATTENUATION
    if normalized in {"power_law", "power_law_mean"}:
        return WavelengthMeanStructure.POWER_LAW
    return WavelengthMeanStructure.CUSTOM


def _mean_is_wavelength_dependent(
    structure: WavelengthMeanStructure,
) -> bool | None:
    if structure is WavelengthMeanStructure.UNKNOWN:
        return None
    return structure is not WavelengthMeanStructure.CONSTANT


@dataclass(frozen=True)
class WavelengthModelHypothesis:
    """Orthogonal mean/covariance description of one complete GP model.

    ``lpv_advisory_priority`` is descriptive ordering metadata only.  It is not
    a model score, selection decision, or guarantee that a model is suitable
    for a particular light curve.
    """

    model: str
    role: WavelengthHypothesisRole
    mean_structure: WavelengthMeanStructure
    covariance_structure: WavelengthCovarianceStructure
    mean_wavelength_dependent: bool | None
    covariance_wavelength_dependent: bool | None
    covariance_separable: bool | None
    temporal_kernel_configurable: bool | None
    baseline: bool = False
    lpv_advisory_priority: int | None = None
    summary: str | None = None
    comparison_cautions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WAVELENGTH_HYPOTHESIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise ValueError("WavelengthModelHypothesis.model must be non-empty.")
        object.__setattr__(self, "model", str(self.model))
        object.__setattr__(
            self, "comparison_cautions", _string_tuple(self.comparison_cautions)
        )
        object.__setattr__(
            self,
            "metadata",
            _json_safe(dict(self.metadata)),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WavelengthModelHypothesis:
        """Construct from the stable public mapping representation."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        model = str(payload.get("model") or "unknown")
        default = describe_wavelength_model_hypothesis(model)
        priority = payload.get(
            "lpv_advisory_priority", default.lpv_advisory_priority
        )
        try:
            priority = int(priority) if priority is not None else None
        except (TypeError, ValueError):
            priority = None
        return cls(
            model=model,
            role=_coerce_enum(
                WavelengthHypothesisRole,
                payload.get("role"),
                default.role,
            ),
            mean_structure=_coerce_enum(
                WavelengthMeanStructure,
                payload.get("mean_structure"),
                default.mean_structure,
            ),
            covariance_structure=_coerce_enum(
                WavelengthCovarianceStructure,
                payload.get("covariance_structure"),
                default.covariance_structure,
            ),
            mean_wavelength_dependent=payload.get(
                "mean_wavelength_dependent",
                default.mean_wavelength_dependent,
            ),
            covariance_wavelength_dependent=payload.get(
                "covariance_wavelength_dependent",
                default.covariance_wavelength_dependent,
            ),
            covariance_separable=payload.get(
                "covariance_separable",
                default.covariance_separable,
            ),
            temporal_kernel_configurable=payload.get(
                "temporal_kernel_configurable",
                default.temporal_kernel_configurable,
            ),
            baseline=bool(payload.get("baseline", default.baseline)),
            lpv_advisory_priority=priority,
            summary=payload.get("summary") or default.summary,
            comparison_cautions=_string_tuple(
                payload.get("comparison_cautions")
                or default.comparison_cautions
            ),
            metadata=payload.get("metadata") or {},
            schema_version=str(
                payload.get("schema_version")
                or WAVELENGTH_HYPOTHESIS_SCHEMA_VERSION
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-safe dictionary."""
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "role": self.role.value,
            "mean_structure": self.mean_structure.value,
            "covariance_structure": self.covariance_structure.value,
            "mean_wavelength_dependent": self.mean_wavelength_dependent,
            "covariance_wavelength_dependent": (
                self.covariance_wavelength_dependent
            ),
            "covariance_separable": self.covariance_separable,
            "temporal_kernel_configurable": self.temporal_kernel_configurable,
            "baseline": self.baseline,
            "lpv_advisory_priority": self.lpv_advisory_priority,
            "summary": self.summary,
            "comparison_cautions": list(self.comparison_cautions),
            "metadata": _json_safe(self.metadata),
        }


_COMMON_COMPLETE_CONFIG_CAUTION = (
    "This label describes a complete mean-plus-covariance GP configuration, "
    "not an isolated physical mechanism."
)

_BASE_HYPOTHESES = {
    "2D": WavelengthModelHypothesis(
        model="2D",
        role=WavelengthHypothesisRole.JOINT_BASELINE,
        mean_structure=WavelengthMeanStructure.CONSTANT,
        covariance_structure=(
            WavelengthCovarianceStructure.JOINT_2D_SPECTRAL_MIXTURE
        ),
        mean_wavelength_dependent=False,
        covariance_wavelength_dependent=True,
        covariance_separable=False,
        temporal_kernel_configurable=False,
        baseline=True,
        lpv_advisory_priority=5,
        summary=(
            "Constant mean with one joint non-separable two-dimensional "
            "spectral-mixture covariance over time and wavelength."
        ),
        comparison_cautions=(
            _COMMON_COMPLETE_CONFIG_CAUTION,
            "Its joint spectral-mixture covariance is not the same parameterization "
            "as the separable LPV configurations.",
        ),
    ),
    "2DSeparable": WavelengthModelHypothesis(
        model="2DSeparable",
        role=WavelengthHypothesisRole.COVARIANCE_FOCUSED,
        mean_structure=WavelengthMeanStructure.CONSTANT,
        covariance_structure=WavelengthCovarianceStructure.SEPARABLE_PRODUCT,
        mean_wavelength_dependent=False,
        covariance_wavelength_dependent=True,
        covariance_separable=True,
        temporal_kernel_configurable=True,
        lpv_advisory_priority=4,
        summary=(
            "Constant mean with a separable product of configurable time and "
            "wavelength covariance kernels."
        ),
        comparison_cautions=(
            _COMMON_COMPLETE_CONFIG_CAUTION,
            "A constant mean can leave wavelength-dependent baseline structure to "
            "the covariance model or residuals.",
        ),
    ),
    "2DWavelengthDependent": WavelengthModelHypothesis(
        model="2DWavelengthDependent",
        role=WavelengthHypothesisRole.MEAN_AND_COVARIANCE,
        mean_structure=WavelengthMeanStructure.QUADRATIC,
        covariance_structure=(
            WavelengthCovarianceStructure.SEPARABLE_SMOOTH_WAVELENGTH
        ),
        mean_wavelength_dependent=True,
        covariance_wavelength_dependent=True,
        covariance_separable=True,
        temporal_kernel_configurable=True,
        lpv_advisory_priority=1,
        summary=(
            "Quadratic wavelength-dependent mean plus separable configurable "
            "time covariance and smooth wavelength covariance."
        ),
        comparison_cautions=(
            _COMMON_COMPLETE_CONFIG_CAUTION,
            "A score difference from 2DSeparable changes both the mean structure "
            "and potentially the instantiated covariance configuration.",
        ),
    ),
    "2DDustMean": WavelengthModelHypothesis(
        model="2DDustMean",
        role=WavelengthHypothesisRole.MEAN_AND_COVARIANCE,
        mean_structure=WavelengthMeanStructure.DUST_ATTENUATION,
        covariance_structure=(
            WavelengthCovarianceStructure.SEPARABLE_SMOOTH_WAVELENGTH
        ),
        mean_wavelength_dependent=True,
        covariance_wavelength_dependent=True,
        covariance_separable=True,
        temporal_kernel_configurable=True,
        lpv_advisory_priority=2,
        summary=(
            "Dust-attenuation wavelength mean plus the same broad family of "
            "separable configurable time and smooth wavelength covariance."
        ),
        comparison_cautions=(
            _COMMON_COMPLETE_CONFIG_CAUTION,
            "This is not a mean-only hypothesis; it also contains wavelength "
            "covariance.",
            "A better complete-fit score alone does not isolate evidence for the "
            "dust mean law.",
        ),
    ),
    "2DPowerLawMean": WavelengthModelHypothesis(
        model="2DPowerLawMean",
        role=WavelengthHypothesisRole.MEAN_AND_COVARIANCE,
        mean_structure=WavelengthMeanStructure.POWER_LAW,
        covariance_structure=(
            WavelengthCovarianceStructure.SEPARABLE_SMOOTH_WAVELENGTH
        ),
        mean_wavelength_dependent=True,
        covariance_wavelength_dependent=True,
        covariance_separable=True,
        temporal_kernel_configurable=True,
        lpv_advisory_priority=3,
        summary=(
            "Power-law wavelength mean plus the same broad family of separable "
            "configurable time and smooth wavelength covariance."
        ),
        comparison_cautions=(
            _COMMON_COMPLETE_CONFIG_CAUTION,
            "This is not a mean-only hypothesis; it also contains wavelength "
            "covariance.",
            "A better complete-fit score alone does not isolate evidence for the "
            "power-law mean.",
        ),
    ),
    "2DAchromatic": WavelengthModelHypothesis(
        model="2DAchromatic",
        role=WavelengthHypothesisRole.ACHROMATIC_CONTROL,
        mean_structure=WavelengthMeanStructure.CONSTANT,
        covariance_structure=(
            WavelengthCovarianceStructure.ACHROMATIC_CONSTANT_WAVELENGTH
        ),
        mean_wavelength_dependent=False,
        covariance_wavelength_dependent=False,
        covariance_separable=True,
        temporal_kernel_configurable=True,
        lpv_advisory_priority=None,
        summary=(
            "Constant mean with a separable temporal covariance and constant "
            "wavelength kernel enforcing the same variability pattern across bands."
        ),
        comparison_cautions=(
            _COMMON_COMPLETE_CONFIG_CAUTION,
            "This model is an achromatic control and is not part of the default "
            "LPV advisory priority ordering.",
        ),
    ),
}


def describe_wavelength_model_hypothesis(
    model: str,
    *,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> WavelengthModelHypothesis:
    """Return descriptive hypothesis metadata for a public model string.

    Parameters
    ----------
    model : str
        Public PGMUVI model name.
    fit_kwargs : mapping, optional
        Fit configuration used to refine metadata such as an explicit
        ``mean_module`` override.  The mapping is inspected only; it is never
        modified.

    Returns
    -------
    WavelengthModelHypothesis
        Immutable descriptive metadata.  Unknown model strings receive an
        explicit ``unknown`` classification rather than being guessed.
    """
    model_name = str(model).strip() or "unknown"
    hypothesis = _BASE_HYPOTHESES.get(model_name)
    if hypothesis is None:
        return WavelengthModelHypothesis(
            model=model_name,
            role=WavelengthHypothesisRole.UNKNOWN,
            mean_structure=WavelengthMeanStructure.UNKNOWN,
            covariance_structure=WavelengthCovarianceStructure.UNKNOWN,
            mean_wavelength_dependent=None,
            covariance_wavelength_dependent=None,
            covariance_separable=None,
            temporal_kernel_configurable=None,
            summary="No maintained wavelength-hypothesis taxonomy is available.",
            comparison_cautions=(
                "Do not infer mean or covariance semantics from an unknown model name.",
            ),
        )

    kwargs = dict(fit_kwargs or {})
    override = _mean_structure_from_override(kwargs.get("mean_module"))
    if override is None or model_name in {
        "2D",
        "2DAchromatic",
        "2DDustMean",
        "2DPowerLawMean",
    }:
        return hypothesis

    return replace(
        hypothesis,
        mean_structure=override,
        mean_wavelength_dependent=_mean_is_wavelength_dependent(override),
        summary=(
            f"{hypothesis.summary} The fit configuration overrides the mean "
            f"structure as {override.value!r}."
        ),
        metadata={
            **dict(hypothesis.metadata),
            "mean_module_override": str(kwargs.get("mean_module")),
        },
    )
