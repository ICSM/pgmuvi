"""Truth-preserving synthetic cases for wavelength-model validation.

The builders in this module construct deterministic D1 validation cases without
running a fit.  Every case contains the observation arrays, the noiseless mean,
the sampled latent process, the reported uncertainties, and a versioned
:class:`~pgmuvi.wavelength_validation.WavelengthValidationScenario` with
explicit generating truth.

The generated light curves remain in linear flux.  A
:class:`~pgmuvi.lightcurve.Lightcurve` is created only when
:meth:`SyntheticWavelengthValidationCase.to_lightcurve` is called, keeping the
scenario-construction layer usable without importing the fitting stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np
import torch

from .wavelength_validation import (
    WavelengthValidationPhase,
    WavelengthValidationProvenance,
    WavelengthValidationScenario,
    WavelengthValidationSourceKind,
    WavelengthValidationTruth,
)

SYNTHETIC_WAVELENGTH_VALIDATION_SCHEMA_VERSION = (
    "pgmuvi-synthetic-wavelength-validation-v1"
)

DEFAULT_VALIDATION_WAVELENGTHS = (0.55, 0.80, 1.25, 2.20, 3.40, 4.60)
DEFAULT_VALIDATION_BANDS = ("V", "I", "J", "K", "W1", "W2")
DEFAULT_VALIDATION_CANDIDATES = (
    "2DWavelengthDependent",
    "2DDustMean",
    "2DPowerLawMean",
    "2DSeparable",
    "2D",
)

__all__ = [
    "DEFAULT_VALIDATION_BANDS",
    "DEFAULT_VALIDATION_CANDIDATES",
    "DEFAULT_VALIDATION_WAVELENGTHS",
    "SYNTHETIC_WAVELENGTH_VALIDATION_SCHEMA_VERSION",
    "SyntheticWavelengthCovarianceKind",
    "SyntheticWavelengthDependenceStrength",
    "SyntheticWavelengthMeanKind",
    "SyntheticWavelengthValidationCase",
    "canonical_synthetic_wavelength_validation_cases",
    "make_synthetic_wavelength_validation_case",
]


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SyntheticWavelengthMeanKind(_StringEnum):
    """Wavelength-dependent mean law used to generate a synthetic case."""

    CONSTANT = "constant"
    QUADRATIC = "quadratic"
    DUST = "dust"
    POWER_LAW = "power_law"


class SyntheticWavelengthCovarianceKind(_StringEnum):
    """Covariance family used to sample the latent synthetic process."""

    SEPARABLE_QUASI_PERIODIC_RBF = "separable_quasi_periodic_rbf"
    JOINT_SPECTRAL_MIXTURE_ARD = "joint_spectral_mixture_ard"


class SyntheticWavelengthDependenceStrength(_StringEnum):
    """Qualitative wavelength-dependence strength for canonical cases."""

    NEGLIGIBLE = "negligible"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


_STRENGTH_SCALE = {
    SyntheticWavelengthDependenceStrength.NEGLIGIBLE: 0.0,
    SyntheticWavelengthDependenceStrength.WEAK: 0.30,
    SyntheticWavelengthDependenceStrength.MODERATE: 0.65,
    SyntheticWavelengthDependenceStrength.STRONG: 1.0,
}

_WAVELENGTH_LENGTHSCALE_FACTOR = {
    SyntheticWavelengthDependenceStrength.NEGLIGIBLE: 20.0,
    SyntheticWavelengthDependenceStrength.WEAK: 2.0,
    SyntheticWavelengthDependenceStrength.MODERATE: 0.75,
    SyntheticWavelengthDependenceStrength.STRONG: 0.30,
}


def _coerce_enum(enum_type, value: Any, *, field_name: str):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}.") from exc


def _float_tuple(values: Any, *, field_name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be an iterable of numbers.")
    try:
        output = tuple(float(item) for item in values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of numbers.") from exc
    return output


def _string_tuple(values: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    try:
        output = tuple(str(item) for item in values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings.") from exc
    return output


def _json_safe(value: Any) -> Any:
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
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError, RuntimeError):
            pass
    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            return _json_safe(tolist_method())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _mapping_copy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _json_safe(item) for key, item in value.items()}


def _resolved_seed_bundle(
    seed: int | None,
    *,
    sampling_seed: int | None,
    process_seed: int | None,
    noise_seed: int | None,
) -> dict[str, int]:
    if seed is None and any(
        item is None for item in (sampling_seed, process_seed, noise_seed)
    ):
        raise ValueError(
            "Provide a master seed or all three purpose-specific seeds."
        )
    if seed is not None:
        seed_sequence = np.random.SeedSequence(int(seed))
        generated = [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in seed_sequence.spawn(3)
        ]
    else:
        generated = [0, 0, 0]
    return {
        "master": int(seed) if seed is not None else -1,
        "sampling": (
            int(sampling_seed) if sampling_seed is not None else generated[0]
        ),
        "process": int(process_seed) if process_seed is not None else generated[1],
        "noise": int(noise_seed) if noise_seed is not None else generated[2],
    }


def _resolve_band_counts(
    n_per_band: int | tuple[int, int] | Sequence[int],
    *,
    n_bands: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    if isinstance(n_per_band, int):
        counts = (int(n_per_band),) * n_bands
    elif isinstance(n_per_band, tuple):
        if len(n_per_band) != 2:
            raise ValueError("n_per_band tuple must contain exactly (min, max).")
        lower, upper = (int(n_per_band[0]), int(n_per_band[1]))
        if lower < 1 or upper < lower:
            raise ValueError(
                "n_per_band range requires 1 <= minimum <= maximum."
            )
        counts = tuple(
            int(rng.integers(lower, upper + 1)) for _ in range(n_bands)
        )
    else:
        counts = tuple(int(item) for item in n_per_band)
        if len(counts) != n_bands:
            raise ValueError(
                "Explicit n_per_band values must match the number of bands."
            )
    if any(item < 2 for item in counts):
        raise ValueError("Every synthetic band must contain at least two rows.")
    return counts


def _coordinate_transform(
    time_values: np.ndarray,
    wavelength_values: np.ndarray,
    *,
    xtransform: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if xtransform is None:
        return (
            time_values.copy(),
            wavelength_values.copy(),
            {
                "kind": "identity",
                "raw_to_model": "model = raw",
                "time": {"origin": 0.0, "scale": 1.0},
                "wavelength": {"origin": 0.0, "scale": 1.0},
            },
        )
    if xtransform != "minmax":
        raise ValueError("PR127 synthetic cases support xtransform=None or 'minmax'.")

    def transform(values: np.ndarray, name: str):
        origin = float(np.min(values))
        scale = float(np.max(values) - origin)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{name} requires a positive finite span.")
        return (values - origin) / scale, {"origin": origin, "scale": scale}

    model_time, time_record = transform(time_values, "time coordinate")
    model_wavelength, wavelength_record = transform(
        wavelength_values, "wavelength coordinate"
    )
    return (
        model_time,
        model_wavelength,
        {
            "kind": "affine_minmax",
            "raw_to_model": "model = (raw - origin) / scale",
            "model_to_raw": "raw = origin + scale * model",
            "time": time_record,
            "wavelength": wavelength_record,
        },
    )


def _default_mean_parameters(
    mean_kind: SyntheticWavelengthMeanKind,
    strength: SyntheticWavelengthDependenceStrength,
    *,
    turning_point: bool,
) -> dict[str, float | list[float]]:
    scale = _STRENGTH_SCALE[strength]
    if mean_kind is SyntheticWavelengthMeanKind.CONSTANT:
        return {"offset": 20.0}
    if mean_kind is SyntheticWavelengthMeanKind.QUADRATIC:
        if turning_point:
            linear = 8.0 * max(scale, 0.30)
            quadratic = -8.0 * max(scale, 0.30)
        else:
            linear = 8.0 * scale
            quadratic = 1.5 * scale
        return {"bias": 20.0, "weights": [linear, quadratic]}
    if mean_kind is SyntheticWavelengthMeanKind.DUST:
        tau = {
            SyntheticWavelengthDependenceStrength.NEGLIGIBLE: 0.01,
            SyntheticWavelengthDependenceStrength.WEAK: 0.20,
            SyntheticWavelengthDependenceStrength.MODERATE: 0.80,
            SyntheticWavelengthDependenceStrength.STRONG: 2.00,
        }[strength]
        return {
            "offset": 5.0,
            "amplitude": 20.0,
            "tau": tau,
            "alpha": 1.7,
        }
    return {
        "offset": 8.0,
        "weight": 5.0 * scale,
        "exponent": 0.8,
    }


def _evaluate_mean(
    mean_kind: SyntheticWavelengthMeanKind,
    *,
    physical_wavelength: np.ndarray,
    model_wavelength: np.ndarray,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    if mean_kind is SyntheticWavelengthMeanKind.CONSTANT:
        return np.full_like(physical_wavelength, float(parameters["offset"]))
    if mean_kind is SyntheticWavelengthMeanKind.QUADRATIC:
        weights = _float_tuple(parameters["weights"], field_name="weights")
        if len(weights) != 2:
            raise ValueError("Quadratic mean requires two weights.")
        return (
            float(parameters["bias"])
            + weights[0] * model_wavelength
            + weights[1] * model_wavelength**2
        )
    if mean_kind is SyntheticWavelengthMeanKind.DUST:
        wavelength = np.clip(physical_wavelength, 1.0e-6, None)
        return float(parameters["offset"]) + float(parameters["amplitude"]) * np.exp(
            -float(parameters["tau"])
            * wavelength ** (-float(parameters["alpha"]))
        )
    wavelength = np.clip(physical_wavelength, 1.0e-12, None)
    exponent = float(parameters["exponent"])
    return (
        float(parameters["offset"])
        + float(parameters["weight"]) * wavelength**exponent
    )


def _mean_truth_parameters(
    mean_kind: SyntheticWavelengthMeanKind,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    output = _mapping_copy(parameters)
    output["mean_kind"] = mean_kind.value
    output["linear_flux"] = True
    if mean_kind is SyntheticWavelengthMeanKind.QUADRATIC:
        output["model_parameter_values"] = {
            "mean_module.bias": float(parameters["bias"]),
            "mean_module.weights": list(parameters["weights"]),
        }
    elif mean_kind is SyntheticWavelengthMeanKind.DUST:
        output["model_parameter_values"] = {
            "mean_module.offset": float(parameters["offset"]),
            "mean_module.log_amplitude": math.log(float(parameters["amplitude"])),
            "mean_module.log_tau": math.log(float(parameters["tau"])),
            "mean_module.log_alpha": math.log(float(parameters["alpha"])),
        }
    elif mean_kind is SyntheticWavelengthMeanKind.POWER_LAW:
        output["model_parameter_values"] = {
            "mean_module.offset": float(parameters["offset"]),
            "mean_module.weight": float(parameters["weight"]),
            "mean_module.exponent": float(parameters["exponent"]),
        }
    else:
        output["model_parameter_values"] = {"mean": float(parameters["offset"])}
    return output


def _resolve_temporal_components(
    temporal_components: Sequence[Mapping[str, Any]] | None,
    *,
    period: float,
) -> tuple[dict[str, float], ...]:
    if temporal_components is None:
        temporal_components = (
            {
                "period": period,
                "variance_fraction": 1.0,
                "coherence_time": 4.0 * period,
                "periodic_lengthscale": 0.65,
            },
        )
    resolved: list[dict[str, float]] = []
    for index, component in enumerate(temporal_components):
        item = {
            "period": float(component.get("period", period)),
            "variance_fraction": float(
                component.get("variance_fraction", 1.0)
            ),
            "coherence_time": float(
                component.get("coherence_time", 4.0 * period)
            ),
            "periodic_lengthscale": float(
                component.get("periodic_lengthscale", 0.65)
            ),
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in item.values()):
            raise ValueError(
                f"Temporal component {index} must contain finite positive values."
            )
        resolved.append(item)
    if not resolved:
        raise ValueError("At least one temporal component is required.")
    total = sum(item["variance_fraction"] for item in resolved)
    for item in resolved:
        item["variance_fraction"] /= total
    return tuple(resolved)


def _quasi_periodic_time_covariance(
    time_values: np.ndarray,
    components: Sequence[Mapping[str, float]],
) -> np.ndarray:
    delta = time_values[:, None] - time_values[None, :]
    output = np.zeros((time_values.size, time_values.size), dtype=float)
    for component in components:
        periodic = np.exp(
            -2.0
            * np.sin(math.pi * delta / component["period"]) ** 2
            / component["periodic_lengthscale"] ** 2
        )
        coherence = np.exp(
            -0.5 * (delta / component["coherence_time"]) ** 2
        )
        output += component["variance_fraction"] * periodic * coherence
    return output


def _separable_covariance(
    time_values: np.ndarray,
    wavelength_values: np.ndarray,
    *,
    components: Sequence[Mapping[str, float]],
    signal_variance: float,
    wavelength_lengthscale: float,
) -> np.ndarray:
    time_covariance = _quasi_periodic_time_covariance(time_values, components)
    wavelength_delta = wavelength_values[:, None] - wavelength_values[None, :]
    wavelength_covariance = np.exp(
        -0.5 * (wavelength_delta / wavelength_lengthscale) ** 2
    )
    return signal_variance * time_covariance * wavelength_covariance


def _joint_spectral_mixture_covariance(
    time_values: np.ndarray,
    wavelength_values: np.ndarray,
    *,
    components: Sequence[Mapping[str, float]],
    signal_variance: float,
    wavelength_lengthscale: float,
    wavelength_frequency: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    coordinates = np.column_stack([time_values, wavelength_values])
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    covariance = np.zeros((coordinates.shape[0], coordinates.shape[0]))
    means = []
    scales = []
    weights = []
    for component in components:
        component_means = np.asarray(
            [1.0 / component["period"], wavelength_frequency], dtype=float
        )
        component_scales = np.asarray(
            [
                1.0 / (2.0 * math.pi * component["coherence_time"]),
                1.0 / (2.0 * math.pi * wavelength_lengthscale),
            ],
            dtype=float,
        )
        weight = signal_variance * component["variance_fraction"]
        exponential = np.exp(
            -2.0
            * math.pi**2
            * np.sum(component_scales**2 * delta**2, axis=-1)
        )
        cosine = np.cos(2.0 * math.pi * np.sum(component_means * delta, axis=-1))
        covariance += weight * exponential * cosine
        means.append([[float(item) for item in component_means]])
        scales.append([[float(item) for item in component_scales]])
        weights.append(float(weight))
    return covariance, {
        "parameterization": "gpytorch_spectral_mixture_frequency_space",
        "coordinate_order": ["temporal_frequency", "wavelength_frequency"],
        "ard_index": {"temporal_frequency": 0, "wavelength_frequency": 1},
        "mixture_means": means,
        "mixture_scales": scales,
        "mixture_weights": weights,
        "lengthscale_to_spectral_scale": "1/(2*pi*lengthscale)",
    }


def _sample_latent_process(
    covariance: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    covariance = 0.5 * (covariance + covariance.T)
    diagonal_scale = max(float(np.max(np.diag(covariance))), 1.0)
    jitter = 1.0e-10 * diagonal_scale
    identity = np.eye(covariance.shape[0])
    for _ in range(8):
        try:
            factor = np.linalg.cholesky(covariance + jitter * identity)
            return factor @ rng.standard_normal(covariance.shape[0]), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise ValueError("Synthetic covariance was not numerically positive definite.")


def _parameter_ownership(
    generating_model: str,
    mean_kind: SyntheticWavelengthMeanKind,
    covariance_kind: SyntheticWavelengthCovarianceKind,
) -> dict[str, Any]:
    mean_parameters = {
        SyntheticWavelengthMeanKind.CONSTANT: ("mean",),
        SyntheticWavelengthMeanKind.QUADRATIC: (
            "mean_module.bias",
            "mean_module.weights",
        ),
        SyntheticWavelengthMeanKind.DUST: (
            "mean_module.offset",
            "mean_module.log_amplitude",
            "mean_module.log_tau",
            "mean_module.log_alpha",
        ),
        SyntheticWavelengthMeanKind.POWER_LAW: (
            "mean_module.offset",
            "mean_module.weight",
            "mean_module.exponent",
        ),
    }[mean_kind]
    if covariance_kind is SyntheticWavelengthCovarianceKind.JOINT_SPECTRAL_MIXTURE_ARD:
        covariance_parameters = (
            "covar_module.mixture_means",
            "covar_module.mixture_scales",
            "covar_module.mixture_weights",
        )
    else:
        covariance_parameters = (
            "time_kernel.period_length",
            "time_kernel.lengthscale",
            "wavelength_kernel.lengthscale",
        )
    return {
        "generating_model": generating_model,
        "mean_parameters": list(mean_parameters),
        "covariance_parameters": list(covariance_parameters),
        "temporal_frequency_ard_index": 0,
        "wavelength_frequency_ard_index": 1,
    }


@dataclass(frozen=True)
class SyntheticWavelengthValidationCase:
    """One synthetic dataset plus its complete validation scenario."""

    scenario: WavelengthValidationScenario
    time_values: tuple[float, ...]
    wavelength_values: tuple[float, ...]
    band_labels: tuple[str, ...]
    noiseless_mean: tuple[float, ...]
    latent_process: tuple[float, ...]
    noiseless_flux: tuple[float, ...]
    observed_flux: tuple[float, ...]
    uncertainties: tuple[float, ...]
    lightcurve_configuration: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SYNTHETIC_WAVELENGTH_VALIDATION_SCHEMA_VERSION
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, WavelengthValidationScenario):
            object.__setattr__(
                self,
                "scenario",
                WavelengthValidationScenario.from_mapping(self.scenario),
            )
        if self.scenario.source_kind is not WavelengthValidationSourceKind.SYNTHETIC:
            raise ValueError("Synthetic cases require a synthetic scenario.")
        fields = (
            "time_values",
            "wavelength_values",
            "noiseless_mean",
            "latent_process",
            "noiseless_flux",
            "observed_flux",
            "uncertainties",
        )
        for name in fields:
            object.__setattr__(
                self, name, _float_tuple(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "band_labels",
            _string_tuple(self.band_labels, field_name="band_labels"),
        )
        n_rows = len(self.time_values)
        if n_rows == 0:
            raise ValueError("Synthetic cases must contain at least one row.")
        for name in (*fields[1:], "band_labels"):
            if len(getattr(self, name)) != n_rows:
                raise ValueError("All synthetic observation arrays must align.")
        numeric_fields = fields
        if any(
            not math.isfinite(item)
            for name in numeric_fields
            for item in getattr(self, name)
        ):
            raise ValueError("Synthetic observation arrays must be finite.")
        if any(item <= 0.0 for item in self.wavelength_values):
            raise ValueError("Synthetic wavelengths must be positive.")
        if any(item <= 0.0 for item in self.uncertainties):
            raise ValueError("Synthetic uncertainties must be positive.")
        if any(item <= 0.0 for item in self.observed_flux):
            raise ValueError("Synthetic observed flux must remain strictly positive.")
        reconstructed = np.asarray(self.noiseless_mean) + np.asarray(
            self.latent_process
        )
        if not np.allclose(reconstructed, self.noiseless_flux, rtol=1e-10, atol=1e-10):
            raise ValueError("noiseless_flux must equal mean plus latent process.")
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(
            self,
            "lightcurve_configuration",
            _mapping_copy(self.lightcurve_configuration),
        )
        object.__setattr__(self, "extra_fields", _mapping_copy(self.extra_fields))

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> SyntheticWavelengthValidationCase:
        """Reconstruct a synthetic case from its JSON-safe mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        known = {
            "schema_version",
            "scenario",
            "time_values",
            "wavelength_values",
            "band_labels",
            "noiseless_mean",
            "latent_process",
            "noiseless_flux",
            "observed_flux",
            "uncertainties",
            "lightcurve_configuration",
            "n_observations",
            "extra_fields",
        }
        extras = _mapping_copy(payload.get("extra_fields"))
        for key, value in payload.items():
            if key not in known:
                extras[str(key)] = _json_safe(value)
        return cls(
            schema_version=str(
                payload.get("schema_version")
                or SYNTHETIC_WAVELENGTH_VALIDATION_SCHEMA_VERSION
            ),
            scenario=WavelengthValidationScenario.from_mapping(
                payload.get("scenario") or {}
            ),
            time_values=payload.get("time_values") or (),
            wavelength_values=payload.get("wavelength_values") or (),
            band_labels=payload.get("band_labels") or (),
            noiseless_mean=payload.get("noiseless_mean") or (),
            latent_process=payload.get("latent_process") or (),
            noiseless_flux=payload.get("noiseless_flux") or (),
            observed_flux=payload.get("observed_flux") or (),
            uncertainties=payload.get("uncertainties") or (),
            lightcurve_configuration=payload.get("lightcurve_configuration") or {},
            extra_fields=extras,
        )

    def to_dict(self, *, include_observations: bool = True) -> dict[str, Any]:
        """Return a JSON-safe representation of the case."""
        payload = {
            "schema_version": self.schema_version,
            "scenario": self.scenario.to_dict(),
            "lightcurve_configuration": _mapping_copy(
                self.lightcurve_configuration
            ),
            "n_observations": len(self.time_values),
            "extra_fields": _mapping_copy(self.extra_fields),
        }
        if include_observations:
            payload.update(
                {
                    "time_values": list(self.time_values),
                    "wavelength_values": list(self.wavelength_values),
                    "band_labels": list(self.band_labels),
                    "noiseless_mean": list(self.noiseless_mean),
                    "latent_process": list(self.latent_process),
                    "noiseless_flux": list(self.noiseless_flux),
                    "observed_flux": list(self.observed_flux),
                    "uncertainties": list(self.uncertainties),
                }
            )
        return payload

    def to_lightcurve(self, **overrides):
        """Build a :class:`pgmuvi.lightcurve.Lightcurve` from this case."""
        from pgmuvi.lightcurve import Lightcurve

        configuration = dict(self.lightcurve_configuration)
        configuration.update(overrides)
        x = torch.tensor(
            np.column_stack([self.time_values, self.wavelength_values]),
            dtype=torch.float64,
        )
        y = torch.tensor(self.observed_flux, dtype=torch.float64)
        yerr = torch.tensor(self.uncertainties, dtype=torch.float64)
        return Lightcurve(
            x,
            y,
            yerr=yerr,
            band=np.asarray(self.band_labels, dtype=str),
            **configuration,
        )


def make_synthetic_wavelength_validation_case(
    *,
    scenario_id: str,
    generating_model: str,
    mean_kind: SyntheticWavelengthMeanKind | str,
    covariance_kind: SyntheticWavelengthCovarianceKind | str,
    strength: SyntheticWavelengthDependenceStrength | str = "moderate",
    wavelengths: Sequence[float] = DEFAULT_VALIDATION_WAVELENGTHS,
    band_labels: Sequence[str] = DEFAULT_VALIDATION_BANDS,
    n_per_band: int | tuple[int, int] | Sequence[int] = 36,
    period: float = 500.0,
    n_cycles: float = 3.2,
    irregular: bool = True,
    shared_time_grid: bool = False,
    temporal_components: Sequence[Mapping[str, Any]] | None = None,
    mean_parameters: Mapping[str, Any] | None = None,
    covariance_parameters: Mapping[str, Any] | None = None,
    turning_point: bool = False,
    noise_sigma: float = 0.15,
    heteroscedastic_fraction: float = 0.0,
    seed: int | None = 0,
    sampling_seed: int | None = None,
    process_seed: int | None = None,
    noise_seed: int | None = None,
    xtransform: str | None = "minmax",
    description: str | None = None,
    tags: Sequence[str] = (),
) -> SyntheticWavelengthValidationCase:
    """Construct one deterministic, truth-preserving D1 synthetic case."""
    scenario_id = str(scenario_id).strip()
    generating_model = str(generating_model).strip()
    if not scenario_id or not generating_model:
        raise ValueError("scenario_id and generating_model must be non-empty.")
    mean_kind = _coerce_enum(
        SyntheticWavelengthMeanKind, mean_kind, field_name="mean_kind"
    )
    covariance_kind = _coerce_enum(
        SyntheticWavelengthCovarianceKind,
        covariance_kind,
        field_name="covariance_kind",
    )
    strength = _coerce_enum(
        SyntheticWavelengthDependenceStrength,
        strength,
        field_name="strength",
    )
    physical_wavelengths = np.asarray(
        _float_tuple(wavelengths, field_name="wavelengths"), dtype=float
    )
    labels = _string_tuple(band_labels, field_name="band_labels")
    if physical_wavelengths.size < 2:
        raise ValueError("Synthetic wavelength validation requires at least two bands.")
    if len(labels) != physical_wavelengths.size:
        raise ValueError("band_labels and wavelengths must have equal lengths.")
    if np.any(~np.isfinite(physical_wavelengths)) or np.any(
        physical_wavelengths <= 0.0
    ):
        raise ValueError("wavelengths must be finite and positive.")
    if np.any(np.diff(physical_wavelengths) <= 0.0):
        raise ValueError("wavelengths must be strictly increasing.")
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive.")
    if not math.isfinite(n_cycles) or n_cycles <= 1.0:
        raise ValueError("n_cycles must be finite and greater than one.")
    if not math.isfinite(noise_sigma) or noise_sigma <= 0.0:
        raise ValueError("noise_sigma must be finite and positive.")
    if not math.isfinite(heteroscedastic_fraction) or not (
        0.0 <= heteroscedastic_fraction <= 1.0
    ):
        raise ValueError("heteroscedastic_fraction must be in [0, 1].")

    seeds = _resolved_seed_bundle(
        seed,
        sampling_seed=sampling_seed,
        process_seed=process_seed,
        noise_seed=noise_seed,
    )
    sampling_rng = np.random.default_rng(seeds["sampling"])
    process_rng = np.random.default_rng(seeds["process"])
    noise_rng = np.random.default_rng(seeds["noise"])
    counts = _resolve_band_counts(
        n_per_band,
        n_bands=physical_wavelengths.size,
        rng=sampling_rng,
    )
    t_span = float(n_cycles * period)
    if shared_time_grid and len(set(counts)) != 1:
        raise ValueError(
            "shared_time_grid=True requires equal observation counts in every band."
        )
    shared_times = None
    if shared_time_grid:
        shared_count = counts[0]
        if irregular:
            shared_times = np.sort(
                sampling_rng.uniform(0.0, t_span, shared_count)
            )
        else:
            shared_times = np.linspace(0.0, t_span, shared_count)

    time_blocks = []
    wavelength_blocks = []
    label_blocks = []
    for wavelength, label, count in zip(
        physical_wavelengths, labels, counts, strict=True
    ):
        if shared_times is not None:
            time_band = shared_times.copy()
        elif irregular:
            time_band = np.sort(sampling_rng.uniform(0.0, t_span, count))
        else:
            time_band = np.linspace(0.0, t_span, count)
        time_blocks.append(time_band)
        wavelength_blocks.append(np.full(count, wavelength))
        label_blocks.extend([label] * count)
    time_values = np.concatenate(time_blocks)
    wavelength_values = np.concatenate(wavelength_blocks)
    _, model_wavelength, coordinate_record = _coordinate_transform(
        time_values,
        wavelength_values,
        xtransform=xtransform,
    )

    resolved_mean_parameters = _default_mean_parameters(
        mean_kind, strength, turning_point=turning_point
    )
    if mean_parameters is not None:
        resolved_mean_parameters.update(_mapping_copy(mean_parameters))
    noiseless_mean = _evaluate_mean(
        mean_kind,
        physical_wavelength=wavelength_values,
        model_wavelength=model_wavelength,
        parameters=resolved_mean_parameters,
    )

    components = _resolve_temporal_components(
        temporal_components,
        period=period,
    )
    covariance_configuration = _mapping_copy(covariance_parameters)
    wavelength_span = float(np.ptp(physical_wavelengths))
    wavelength_lengthscale = float(
        covariance_configuration.get(
            "wavelength_lengthscale",
            _WAVELENGTH_LENGTHSCALE_FACTOR[strength] * wavelength_span,
        )
    )
    signal_std = float(covariance_configuration.get("signal_std", 1.25))
    signal_variance = signal_std**2
    if not math.isfinite(wavelength_lengthscale) or wavelength_lengthscale <= 0.0:
        raise ValueError("wavelength_lengthscale must be finite and positive.")
    if not math.isfinite(signal_std) or signal_std <= 0.0:
        raise ValueError("signal_std must be finite and positive.")

    if covariance_kind is (
        SyntheticWavelengthCovarianceKind.SEPARABLE_QUASI_PERIODIC_RBF
    ):
        covariance = _separable_covariance(
            time_values,
            wavelength_values,
            components=components,
            signal_variance=signal_variance,
            wavelength_lengthscale=wavelength_lengthscale,
        )
        covariance_truth = {
            "covariance_kind": covariance_kind.value,
            "signal_std": signal_std,
            "signal_variance": signal_variance,
            "time_kernel": "sum_quasi_periodic",
            "temporal_components": [dict(item) for item in components],
            "wavelength_kernel": "rbf",
            "wavelength_lengthscale": wavelength_lengthscale,
            "wavelength_lengthscale_coordinate": "physical",
        }
    else:
        wavelength_frequency = float(
            covariance_configuration.get("wavelength_frequency", 0.0)
        )
        covariance, sm_truth = _joint_spectral_mixture_covariance(
            time_values,
            wavelength_values,
            components=components,
            signal_variance=signal_variance,
            wavelength_lengthscale=wavelength_lengthscale,
            wavelength_frequency=wavelength_frequency,
        )
        covariance_truth = {
            "covariance_kind": covariance_kind.value,
            "signal_std": signal_std,
            "signal_variance": signal_variance,
            "wavelength_lengthscale": wavelength_lengthscale,
            "wavelength_lengthscale_coordinate": "physical",
            **sm_truth,
        }

    latent_process, applied_jitter = _sample_latent_process(
        covariance, rng=process_rng
    )
    noiseless_flux = noiseless_mean + latent_process
    relative_mean = np.abs(noiseless_mean) / max(
        float(np.median(np.abs(noiseless_mean))), 1.0e-12
    )
    uncertainties = noise_sigma * (
        1.0 + heteroscedastic_fraction * (relative_mean - 1.0)
    )
    uncertainties = np.clip(uncertainties, 0.25 * noise_sigma, None)
    observed_flux = noiseless_flux + noise_rng.standard_normal(
        noiseless_flux.size
    ) * uncertainties

    positive_flux_shift = 0.0
    minimum_observed = float(np.min(observed_flux))
    if minimum_observed <= 0.0:
        positive_flux_shift = 1.0 - minimum_observed
        noiseless_mean = noiseless_mean + positive_flux_shift
        noiseless_flux = noiseless_flux + positive_flux_shift
        observed_flux = observed_flux + positive_flux_shift
        offset_key = (
            "bias"
            if mean_kind is SyntheticWavelengthMeanKind.QUADRATIC
            else "offset"
        )
        resolved_mean_parameters[offset_key] = (
            float(resolved_mean_parameters[offset_key]) + positive_flux_shift
        )

    unique_model_wavelengths = []
    mean_by_band = []
    for wavelength in physical_wavelengths:
        mask = wavelength_values == wavelength
        unique_model_wavelengths.append(float(model_wavelength[mask][0]))
        mean_by_band.append(float(np.mean(noiseless_mean[mask])))

    mean_truth = _mean_truth_parameters(mean_kind, resolved_mean_parameters)
    mean_truth["positive_flux_shift"] = positive_flux_shift
    truth = WavelengthValidationTruth(
        generating_model=generating_model,
        truth_kind="mean_and_covariance",
        physical_wavelengths=tuple(float(item) for item in physical_wavelengths),
        band_labels=labels,
        temporal_parameters={
            "fundamental_period": float(period),
            "n_cycles": float(n_cycles),
            "components": [dict(item) for item in components],
            "fundamental_plus_harmonic": len(components) > 1,
        },
        wavelength_mean_parameters=mean_truth,
        wavelength_covariance_parameters=covariance_truth,
        latent_parameters={
            "sampling_seed": seeds["sampling"],
            "process_seed": seeds["process"],
            "noise_seed": seeds["noise"],
            "applied_covariance_jitter": applied_jitter,
        },
        noiseless_summary={
            "linear_flux": True,
            "strictly_positive_observed_flux": True,
            "mean_by_band": mean_by_band,
            "model_wavelength_by_band": unique_model_wavelengths,
            "latent_standard_deviation": float(np.std(latent_process)),
            "noiseless_flux_minimum": float(np.min(noiseless_flux)),
            "noiseless_flux_maximum": float(np.max(noiseless_flux)),
        },
        coordinate_transforms=coordinate_record,
        parameter_ownership=_parameter_ownership(
            generating_model, mean_kind, covariance_kind
        ),
        metadata={
            "mean_kind": mean_kind.value,
            "covariance_kind": covariance_kind.value,
            "dependence_strength": strength.value,
            "turning_point": bool(turning_point),
        },
    )
    description = description or (
        f"{strength.value.capitalize()} {mean_kind.value} mean with "
        f"{covariance_kind.value} covariance."
    )
    scenario = WavelengthValidationScenario(
        scenario_id=scenario_id,
        phase=WavelengthValidationPhase.D1_SYNTHETIC_RECOVERY,
        source_kind=WavelengthValidationSourceKind.SYNTHETIC,
        description=description,
        truth=truth,
        sampling_configuration={
            "n_per_band": list(counts),
            "n_observations": int(time_values.size),
            "n_cycles": float(n_cycles),
            "time_span": t_span,
            "irregular": bool(irregular),
            "shared_time_grid": bool(shared_time_grid),
            "purpose_specific_seeds": seeds,
        },
        noise_configuration={
            "noise_type": "gaussian",
            "noise_sigma": noise_sigma,
            "heteroscedastic_fraction": heteroscedastic_fraction,
            "reported_uncertainties": True,
        },
        preprocessing_configuration={
            "linear_flux": True,
            "positive_flux_required": True,
            "xtransform": xtransform,
        },
        fit_configuration={
            "candidate_models": list(DEFAULT_VALIDATION_CANDIDATES),
            "fit_strategy": "consensus",
            "time_kernel_type": "quasi_periodic",
            "learn_additional_noise": True,
            "selection_performed": False,
        },
        tags=(
            "d1",
            "synthetic",
            mean_kind.value,
            covariance_kind.value,
            strength.value,
            *tuple(str(item) for item in tags),
        ),
        provenance=WavelengthValidationProvenance(
            source_identifier=scenario_id,
            configuration={
                "generator_schema_version": (
                    SYNTHETIC_WAVELENGTH_VALIDATION_SCHEMA_VERSION
                ),
                "master_seed": seeds["master"],
            },
        ),
        metadata={
            "generator": "make_synthetic_wavelength_validation_case",
            "generator_schema_version": (
                SYNTHETIC_WAVELENGTH_VALIDATION_SCHEMA_VERSION
            ),
        },
    )
    lightcurve_configuration = {
        "xtransform": xtransform,
        "center_time": False if xtransform is None else "auto",
        "max_samples": None,
        "max_samples_per_band": None,
        "check_sampling": False,
        "name": scenario_id,
    }
    return SyntheticWavelengthValidationCase(
        scenario=scenario,
        time_values=tuple(float(item) for item in time_values),
        wavelength_values=tuple(float(item) for item in wavelength_values),
        band_labels=tuple(label_blocks),
        noiseless_mean=tuple(float(item) for item in noiseless_mean),
        latent_process=tuple(float(item) for item in latent_process),
        noiseless_flux=tuple(float(item) for item in noiseless_flux),
        observed_flux=tuple(float(item) for item in observed_flux),
        uncertainties=tuple(float(item) for item in uncertainties),
        lightcurve_configuration=lightcurve_configuration,
    )


def canonical_synthetic_wavelength_validation_cases(
    *, seed: int = 0
) -> tuple[SyntheticWavelengthValidationCase, ...]:
    """Return the canonical nominal D1 scenario set.

    The set spans the maintained wavelength-model families and includes one
    quadratic turning-point case plus one known fundamental-and-harmonic case.
    Nominal recovery uses 72 shared-grid observations per band over 3.2 cycles;
    sparse and uneven designs are reserved for D2 robustness validation.  Seeds
    are offset deterministically so cases remain independent while the complete
    suite remains reproducible.
    """
    common = {
        "wavelengths": DEFAULT_VALIDATION_WAVELENGTHS,
        "band_labels": DEFAULT_VALIDATION_BANDS,
        "n_per_band": 72,
        "period": 500.0,
        "n_cycles": 3.2,
        "noise_sigma": 0.15,
        "xtransform": "minmax",
        "shared_time_grid": True,
    }
    specifications = (
        {
            "scenario_id": "d1-separable-constant-moderate",
            "generating_model": "2DSeparable",
            "mean_kind": "constant",
            "covariance_kind": "separable_quasi_periodic_rbf",
            "strength": "moderate",
        },
        {
            "scenario_id": "d1-quadratic-monotonic-moderate",
            "generating_model": "2DWavelengthDependent",
            "mean_kind": "quadratic",
            "covariance_kind": "separable_quasi_periodic_rbf",
            "strength": "moderate",
        },
        {
            "scenario_id": "d1-quadratic-turning-point",
            "generating_model": "2DWavelengthDependent",
            "mean_kind": "quadratic",
            "covariance_kind": "separable_quasi_periodic_rbf",
            "strength": "strong",
            "turning_point": True,
        },
        {
            "scenario_id": "d1-dust-mean-strong",
            "generating_model": "2DDustMean",
            "mean_kind": "dust",
            "covariance_kind": "separable_quasi_periodic_rbf",
            "strength": "strong",
        },
        {
            "scenario_id": "d1-power-law-mean-moderate",
            "generating_model": "2DPowerLawMean",
            "mean_kind": "power_law",
            "covariance_kind": "separable_quasi_periodic_rbf",
            "strength": "moderate",
        },
        {
            "scenario_id": "d1-joint-sm-ard-moderate",
            "generating_model": "2D",
            "mean_kind": "constant",
            "covariance_kind": "joint_spectral_mixture_ard",
            "strength": "moderate",
        },
        {
            "scenario_id": "d1-fundamental-harmonic",
            "generating_model": "2DSeparable",
            "mean_kind": "constant",
            "covariance_kind": "separable_quasi_periodic_rbf",
            "strength": "moderate",
            "temporal_components": (
                {
                    "period": 500.0,
                    "variance_fraction": 0.80,
                    "coherence_time": 2000.0,
                    "periodic_lengthscale": 0.65,
                },
                {
                    "period": 250.0,
                    "variance_fraction": 0.20,
                    "coherence_time": 1500.0,
                    "periodic_lengthscale": 0.65,
                },
            ),
        },
    )
    return tuple(
        make_synthetic_wavelength_validation_case(
            **common,
            **specification,
            seed=seed + index,
        )
        for index, specification in enumerate(specifications)
    )
