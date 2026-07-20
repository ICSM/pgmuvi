"""Representative observed-LPV wavelength-validation orchestration.

This module adapts the maintained period-independent wavelength advisory
workflow to D3 representative observed sources.  It records source structure,
fit evidence, warnings, constraints, and failures without constructing
truth-recovery metrics or selecting a model automatically.
"""

from __future__ import annotations

import math
import platform
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from .wavelength_validation import (
    WavelengthValidationPhase,
    WavelengthValidationScenario,
    WavelengthValidationSourceKind,
)

REPRESENTATIVE_LPV_VALIDATION_SCHEMA_VERSION = "1.0"

DEFAULT_REPRESENTATIVE_LPV_MODELS = (
    "2DWavelengthDependent",
    "2DDustMean",
    "2DPowerLawMean",
    "2DSeparable",
    "2D",
)


def _json_safe(value: Any) -> Any:
    """Return a recursively JSON-safe value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_json_safe(item) for item in value]

    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach()
            cpu = getattr(value, "cpu", None)
            if callable(cpu):
                value = cpu()
        except (TypeError, ValueError, RuntimeError):
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError, RuntimeError):
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError, RuntimeError):
            pass

    return str(value)


def _representative_lpv_package_version(
    package_name: str,
) -> str:
    """Return an installed package version or an explicit fallback."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def build_representative_lpv_runtime_environment() -> dict[str, Any]:
    """Return compact JSON-safe runtime provenance for D3 execution."""
    try:
        import gpytorch
    except ImportError:
        gpytorch_version = "unavailable"
    else:
        gpytorch_version = str(
            getattr(gpytorch, "__version__", "unknown")
        )

    try:
        import torch
    except ImportError:
        torch_version = "unavailable"
        cuda_available = False
    else:
        torch_version = str(
            getattr(torch, "__version__", "unknown")
        )
        cuda_available = bool(torch.cuda.is_available())

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pgmuvi_version": _representative_lpv_package_version(
            "pgmuvi"
        ),
        "numpy_version": str(np.__version__),
        "torch_version": torch_version,
        "gpytorch_version": gpytorch_version,
        "cuda_available": cuda_available,
    }


@contextmanager
def _temporary_representative_lpv_seed(seed: int):
    """Apply one source seed and restore all supported RNG states."""
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _numpy_array(value: Any) -> np.ndarray:
    """Convert a tensor/array-like value to a NumPy array."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
    return np.asarray(value)


def _observational_channel_labels(lightcurve: Any) -> np.ndarray:
    """Return row-aligned observational-channel labels."""
    labels = getattr(
        lightcurve,
        "observational_channel_labels",
        None,
    )
    if labels is None:
        labels = getattr(lightcurve, "band", None)
    if labels is None:
        raise ValueError(
            "Representative LPV validation requires row-aligned "
            "observational-channel labels."
        )

    labels = np.asarray(labels, dtype=str)
    if labels.ndim != 1:
        raise ValueError(
            "Observational-channel labels must be one-dimensional."
        )
    return labels


def build_representative_lpv_source_summary(
    lightcurve: Any,
) -> dict[str, Any]:
    """Summarize observed source rows, channels, and wavelengths.

    Multiple observational channels at one physical wavelength remain
    distinct.  No instrumental calibration, averaging, merging, flux
    correction, or artificial wavelength reassignment is performed.
    """
    xdata = getattr(lightcurve, "_xdata_raw", None)
    ydata = getattr(lightcurve, "_ydata_raw", None)
    if xdata is None or ydata is None:
        raise ValueError(
            "Representative LPV validation requires a Lightcurve-like "
            "object with raw xdata and ydata."
        )

    xarray = _numpy_array(xdata)
    yarray = _numpy_array(ydata).reshape(-1)
    if xarray.ndim != 2 or xarray.shape[1] < 2:
        raise ValueError(
            "Representative LPV validation requires two-dimensional "
            "time/wavelength coordinates."
        )
    if xarray.shape[0] != yarray.shape[0]:
        raise ValueError("Raw xdata and ydata row counts must match.")

    channels = _observational_channel_labels(lightcurve)
    if channels.shape[0] != xarray.shape[0]:
        raise ValueError(
            "Observational-channel labels must align with data rows."
        )

    wavelengths = np.asarray(xarray[:, 1], dtype=float)
    finite = np.isfinite(wavelengths)
    if not np.all(finite):
        raise ValueError(
            "Physical wavelength coordinates must be finite."
        )

    unique_channels = sorted(
        str(item) for item in np.unique(channels)
    )
    unique_wavelengths = np.unique(wavelengths)

    channels_by_wavelength: dict[str, list[str]] = {}
    shared: dict[str, list[str]] = {}
    for wavelength in unique_wavelengths:
        mask = wavelengths == wavelength
        attached_channels = sorted(
            str(item) for item in np.unique(channels[mask])
        )
        key = format(float(wavelength), ".17g")
        channels_by_wavelength[key] = attached_channels
        if len(attached_channels) > 1:
            shared[key] = attached_channels

    return _json_safe(
        {
            "kind": "representative_lpv_source_summary",
            "n_rows": int(xarray.shape[0]),
            "n_observational_channels": len(unique_channels),
            "observational_channels": unique_channels,
            "n_distinct_physical_wavelengths": int(
                unique_wavelengths.size
            ),
            "physical_wavelengths": [
                float(item) for item in unique_wavelengths
            ],
            "observational_channels_by_physical_wavelength": (
                channels_by_wavelength
            ),
            "multiple_observational_channels_per_wavelength": bool(
                shared
            ),
            "observational_channels_by_shared_wavelength": shared,
            "instrument_calibration_status": "not_implemented",
            "instrument_calibration_tbd": True,
            "instrument_calibration_marker": (
                "TBD[instrument-channel-calibration]"
            ),
            "shared_wavelength_policy": (
                "preserve_channels_without_calibration"
            ),
            "linear_flux": True,
        }
    )


def _workflow_attempt_rows(
    workflow_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    run_report = workflow_report.get("run_report")
    run_report = (
        run_report if isinstance(run_report, Mapping) else {}
    )
    rows = (
        run_report.get("model_kernel_config_results")
        or run_report.get("outcomes")
        or ()
    )
    return [
        dict(item)
        for item in rows
        if isinstance(item, Mapping)
    ]


def _period_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        period = row.get("consensus_period")
        frequency = row.get("consensus_frequency")
        if period is None and frequency is None:
            continue
        model = str(row.get("model") or "").strip()
        if not model:
            continue
        by_model[model] = _json_safe(
            {
                "model_kernel_config_id": row.get(
                    "model_kernel_config_id"
                ),
                "consensus_success": row.get("consensus_success"),
                "consensus_period": period,
                "consensus_frequency": frequency,
                "consensus_time_kernel_constraint_mode": row.get(
                    "consensus_time_kernel_constraint_mode"
                ),
                "n_accepted_observational_channels": row.get(
                    "n_accepted_bands"
                ),
                "n_rejected_observational_channels": row.get(
                    "n_rejected_bands"
                ),
            }
        )

    return {
        "n_models_with_period_evidence": len(by_model),
        "by_model": by_model,
        "truth_recovery_evidence": False,
    }


def _fit_quality_evidence(
    workflow_report: Mapping[str, Any],
) -> dict[str, Any]:
    quality = workflow_report.get("quality_report")
    quality = quality if isinstance(quality, Mapping) else {}

    ranking_status = (
        quality.get("fit_quality_ranking_status")
        or workflow_report.get("fit_quality_ranking_status")
    )
    ranking_available = (
        quality.get("fit_quality_ranking_available")
        if "fit_quality_ranking_available" in quality
        else workflow_report.get(
            "fit_quality_ranking_available",
            False,
        )
    )

    return _json_safe(
        {
            "score_kind": (
                quality.get("score_kind")
                or workflow_report.get("score_kind")
            ),
            "ranking_status": ranking_status,
            "ranking_available": bool(ranking_available),
            "n_with_fit_quality": quality.get(
                "n_with_fit_quality"
            ),
            "top_ranked_model": (
                quality.get("top_ranked_model")
                or workflow_report.get("top_ranked_model")
            ),
            "top_ranked_fit_quality_score": (
                quality.get("top_ranked_fit_quality_score")
                or workflow_report.get(
                    "top_ranked_fit_quality_score"
                )
            ),
            "ranked_results": quality.get("ranked_results") or [],
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "truth_recovery_evidence": False,
            "interpretation": (
                "Descriptive training-space comparison evidence only."
            ),
        }
    )


def _residual_wavelength_evidence(
    workflow_report: Mapping[str, Any],
    source_summary: Mapping[str, Any],
) -> dict[str, Any]:
    config_report = workflow_report.get(
        "model_kernel_config_report"
    )
    config_report = (
        config_report
        if isinstance(config_report, Mapping)
        else {}
    )
    diagnostics = config_report.get(
        "period_independent_diagnostics"
    )
    diagnostics = (
        diagnostics if isinstance(diagnostics, Mapping) else {}
    )

    result = dict(_json_safe(diagnostics))
    result.setdefault(
        "n_observational_channels",
        source_summary.get("n_observational_channels"),
    )
    result.setdefault(
        "n_distinct_physical_wavelengths",
        source_summary.get(
            "n_distinct_physical_wavelengths"
        ),
    )
    result["truth_recovery_evidence"] = False
    return result


def _constraint_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = str(row.get("model") or "").strip()
        if not model:
            continue
        by_model[model] = _json_safe(
            {
                "model_kernel_config_id": row.get(
                    "model_kernel_config_id"
                ),
                "parameter_workflow": row.get(
                    "parameter_workflow"
                )
                or {},
                "n_sm_ard_boundary_hits": row.get(
                    "n_sm_ard_boundary_hits"
                ),
                "sm_ard_boundary_hits": row.get(
                    "sm_ard_boundary_hits"
                )
                or [],
                "n_constrained_sm_ard_components": row.get(
                    "n_constrained_sm_ard_components"
                ),
                "constrained_sm_ard_components": row.get(
                    "constrained_sm_ard_components"
                )
                or [],
                "fit_kwargs": row.get("fit_kwargs") or {},
            }
        )
    return {
        "by_model": by_model,
        "truth_recovery_evidence": False,
    }


def _warning_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = []
    by_category: dict[str, int] = {}

    for row in rows:
        model = row.get("model")
        config_id = row.get("model_kernel_config_id")
        for warning in row.get("warnings") or ():
            if not isinstance(warning, Mapping):
                continue
            record = {
                **dict(warning),
                "model": model,
                "model_kernel_config_id": config_id,
            }
            records.append(_json_safe(record))
            category = str(
                warning.get("category") or "unclassified"
            )
            by_category[category] = (
                by_category.get(category, 0) + 1
            )

    return {
        "n_warning_records": len(records),
        "by_category": by_category,
        "records": records,
    }


def _failure_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures = []
    by_code: dict[str, int] = {}
    by_stage: dict[str, int] = {}

    for row in rows:
        failed = (
            row.get("fit_failed") is True
            or row.get("status") == "failed"
            or row.get("technical_outcome") == "failed"
        )
        if not failed:
            continue

        code = str(
            row.get("failure_code") or "unclassified_failure"
        )
        stage = str(
            row.get("failure_stage") or "unclassified"
        )
        by_code[code] = by_code.get(code, 0) + 1
        by_stage[stage] = by_stage.get(stage, 0) + 1

        failures.append(
            _json_safe(
                {
                    "model_kernel_config_id": row.get(
                        "model_kernel_config_id"
                    ),
                    "model": row.get("model"),
                    "failure_code": code,
                    "failure_stage": stage,
                    "failure_substage": row.get(
                        "failure_substage"
                    ),
                    "failure_stage_reason": row.get(
                        "failure_stage_reason"
                    ),
                    "exception_type": row.get("exception_type"),
                    "exception_message": row.get(
                        "exception_message"
                    ),
                    "is_consensus_failure": row.get(
                        "is_consensus_failure"
                    ),
                    "is_numerical_failure": row.get(
                        "is_numerical_failure"
                    ),
                    "is_input_validation_failure": row.get(
                        "is_input_validation_failure"
                    ),
                }
            )
        )

    return {
        "n_failed_attempts": len(failures),
        "by_code": by_code,
        "by_stage": by_stage,
        "records": failures,
    }


def _build_source_evidence(
    workflow_report: Mapping[str, Any],
    source_summary: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _workflow_attempt_rows(workflow_report)
    return {
        "period_evidence": _period_evidence(rows),
        "fit_quality": _fit_quality_evidence(workflow_report),
        "residual_wavelength_structure": (
            _residual_wavelength_evidence(
                workflow_report,
                source_summary,
            )
        ),
        "constraint_diagnostics": _constraint_evidence(rows),
        "warnings": _warning_evidence(rows),
        "failures": _failure_evidence(rows),
    }


def _validate_nonselecting_workflow(
    workflow_report: Mapping[str, Any],
) -> None:
    if workflow_report.get(
        "automatic_model_selection_applied"
    ) is not False:
        raise ValueError(
            "Representative LPV validation forbids automatic model "
            "selection."
        )
    if workflow_report.get("selected_model") is not None:
        raise ValueError(
            "Representative LPV validation forbids automatic model "
            "selection and requires selected_model=None."
        )


@dataclass(frozen=True)
class RepresentativeLPVValidationReport:
    """Typed D3 report for one representative observed LPV."""

    report_id: str
    scenario: WavelengthValidationScenario
    source_summary: Mapping[str, Any]
    workflow_report: Mapping[str, Any]
    source_evidence: Mapping[str, Any]
    schema_version: str = (
        REPRESENTATIVE_LPV_VALIDATION_SCHEMA_VERSION
    )
    advisory_only: bool = True
    automatic_model_selection_applied: bool = False
    selected_model: str | None = None
    notes: tuple[str, ...] = ()
    extra_fields: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        report_id = str(self.report_id or "").strip()
        if not report_id:
            raise ValueError("report_id must be non-empty.")
        object.__setattr__(self, "report_id", report_id)

        scenario = self.scenario
        if not isinstance(scenario, WavelengthValidationScenario):
            scenario = WavelengthValidationScenario.from_mapping(
                scenario
            )
            object.__setattr__(self, "scenario", scenario)

        if (
            scenario.phase
            is not WavelengthValidationPhase.D3_REPRESENTATIVE_LPV
        ):
            raise ValueError(
                "Representative LPV reports require a D3 scenario."
            )
        if (
            scenario.source_kind
            is not WavelengthValidationSourceKind.OBSERVED
        ):
            raise ValueError(
                "Representative LPV reports require an observed source."
            )
        if scenario.truth is not None:
            raise ValueError(
                "Observed representative LPV reports cannot claim a "
                "truth record."
            )

        if self.advisory_only is not True:
            raise ValueError(
                "Representative LPV reports must remain advisory-only."
            )
        if self.automatic_model_selection_applied is not False:
            raise ValueError(
                "Representative LPV reports cannot select a model."
            )
        if self.selected_model is not None:
            raise ValueError(
                "Representative LPV reports require "
                "selected_model=None."
            )

        workflow = dict(_json_safe(self.workflow_report))
        _validate_nonselecting_workflow(workflow)

        object.__setattr__(
            self,
            "schema_version",
            str(self.schema_version),
        )
        object.__setattr__(
            self,
            "source_summary",
            dict(_json_safe(self.source_summary)),
        )
        object.__setattr__(self, "workflow_report", workflow)
        object.__setattr__(
            self,
            "source_evidence",
            dict(_json_safe(self.source_evidence)),
        )
        object.__setattr__(
            self,
            "notes",
            tuple(str(item) for item in self.notes),
        )
        object.__setattr__(
            self,
            "extra_fields",
            dict(_json_safe(self.extra_fields)),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> RepresentativeLPVValidationReport:
        """Reconstruct a report from a JSON-safe mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")

        known = {
            "schema_version",
            "report_id",
            "scenario",
            "source_summary",
            "workflow_report",
            "source_evidence",
            "advisory_only",
            "automatic_model_selection_applied",
            "selected_model",
            "notes",
            "extra_fields",
        }
        extras = dict(payload.get("extra_fields") or {})
        for key, value in payload.items():
            if key not in known:
                extras[str(key)] = _json_safe(value)

        return cls(
            schema_version=str(
                payload.get("schema_version")
                or REPRESENTATIVE_LPV_VALIDATION_SCHEMA_VERSION
            ),
            report_id=str(payload.get("report_id") or ""),
            scenario=WavelengthValidationScenario.from_mapping(
                payload.get("scenario") or {}
            ),
            source_summary=payload.get("source_summary") or {},
            workflow_report=payload.get("workflow_report") or {},
            source_evidence=payload.get("source_evidence") or {},
            advisory_only=payload.get("advisory_only", True),
            automatic_model_selection_applied=payload.get(
                "automatic_model_selection_applied",
                False,
            ),
            selected_model=payload.get("selected_model"),
            notes=tuple(payload.get("notes") or ()),
            extra_fields=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-safe D3 report envelope."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "scenario": self.scenario.to_dict(),
            "source_summary": _json_safe(self.source_summary),
            "workflow_report": _json_safe(self.workflow_report),
            "source_evidence": _json_safe(self.source_evidence),
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "notes": list(self.notes),
            "extra_fields": _json_safe(self.extra_fields),
        }


def build_representative_lpv_validation_report(
    *,
    source_id: str,
    description: str,
    source_summary: Mapping[str, Any],
    workflow_report: Mapping[str, Any],
    report_id: str | None = None,
    notes: Sequence[str] = (),
) -> RepresentativeLPVValidationReport:
    """Build a D3 report from an already-completed advisory workflow."""
    source_id = str(source_id or "").strip()
    description = str(description or "").strip()
    if not source_id:
        raise ValueError("source_id must be non-empty.")
    if not description:
        raise ValueError("description must be non-empty.")
    if not isinstance(workflow_report, Mapping):
        raise TypeError("workflow_report must be a mapping.")

    normalized_workflow = dict(_json_safe(workflow_report))
    _validate_nonselecting_workflow(normalized_workflow)
    normalized_summary = dict(_json_safe(source_summary))

    scenario = WavelengthValidationScenario(
        scenario_id=source_id,
        phase=WavelengthValidationPhase.D3_REPRESENTATIVE_LPV,
        source_kind=WavelengthValidationSourceKind.OBSERVED,
        description=description,
    )

    return RepresentativeLPVValidationReport(
        report_id=report_id or f"d3-representative-lpv:{source_id}",
        scenario=scenario,
        source_summary=normalized_summary,
        workflow_report=normalized_workflow,
        source_evidence=_build_source_evidence(
            normalized_workflow,
            normalized_summary,
        ),
        notes=(
            "Observed-source D3 evidence is descriptive and contains "
            "no truth-recovery gates.",
            "Candidate rankings remain advisory and do not install a "
            "model.",
            *tuple(str(item) for item in notes),
        ),
    )


def run_representative_lpv_validation(
    lightcurve: Any,
    *,
    source_id: str,
    description: str = "Representative observed LPV source.",
    models: Sequence[str] = DEFAULT_REPRESENTATIVE_LPV_MODELS,
    workflow_runner: Callable[..., Mapping[str, Any]] | None = None,
    workflow_kwargs: Mapping[str, Any] | None = None,
    report_id: str | None = None,
    notes: Sequence[str] = (),
) -> RepresentativeLPVValidationReport:
    """Execute the maintained advisory workflow and build one D3 report."""
    resolved_models = tuple(str(model) for model in models)
    if not resolved_models:
        raise ValueError("models must contain at least one model.")
    if "2DAchromatic" in resolved_models:
        raise ValueError(
            "2DAchromatic is outside the representative LPV model "
            "priority set."
        )

    if workflow_runner is None:
        from .wavelength_diagnostics import (
            run_period_independent_wavelength_advisory_workflow,
        )

        workflow_runner = (
            run_period_independent_wavelength_advisory_workflow
        )

    kwargs = dict(workflow_kwargs or {})
    kwargs.setdefault("include_models", resolved_models)
    kwargs.setdefault("include_2d_baseline", True)
    kwargs.setdefault("make_text_report", True)
    kwargs.setdefault("make_plots", False)

    workflow_report = workflow_runner(lightcurve, **kwargs)
    if not isinstance(workflow_report, Mapping):
        raise TypeError(
            "workflow_runner must return a workflow mapping."
        )
    _validate_nonselecting_workflow(workflow_report)

    source_summary = build_representative_lpv_source_summary(
        lightcurve
    )
    return build_representative_lpv_validation_report(
        source_id=source_id,
        description=description,
        source_summary=source_summary,
        workflow_report=workflow_report,
        report_id=report_id,
        notes=notes,
    )


@dataclass(frozen=True)
class RepresentativeLPVSourceSpecification:
    """Manifest entry for one representative observed LPV source."""

    source_id: str
    source_path: str
    description: str
    sample_role: str
    selection_reason: str
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "source_path",
            "description",
            "sample_role",
            "selection_reason",
        ):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty.")
            object.__setattr__(self, field_name, value)

        if isinstance(self.seed, bool):
            raise ValueError("seed must be a nonnegative integer.")
        try:
            seed = int(self.seed)
        except (TypeError, ValueError) as exception:
            raise ValueError(
                "seed must be a nonnegative integer."
            ) from exception
        if seed < 0:
            raise ValueError("seed must be nonnegative.")
        object.__setattr__(self, "seed", seed)

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(
            self,
            "metadata",
            dict(_json_safe(self.metadata)),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> RepresentativeLPVSourceSpecification:
        """Build a source specification from a JSON-style mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("source specification must be a mapping.")

        return cls(
            source_id=payload.get("source_id", ""),
            source_path=payload.get("source_path", ""),
            description=payload.get("description", ""),
            sample_role=payload.get("sample_role", ""),
            selection_reason=payload.get("selection_reason", ""),
            seed=payload.get("seed", 0),
            metadata=payload.get("metadata") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe manifest entry."""
        return {
            "source_id": self.source_id,
            "source_path": self.source_path,
            "description": self.description,
            "sample_role": self.sample_role,
            "selection_reason": self.selection_reason,
            "seed": self.seed,
            "metadata": _json_safe(self.metadata),
        }


def validate_representative_lpv_source_manifest(
    manifest: Sequence[
        RepresentativeLPVSourceSpecification | Mapping[str, Any]
    ],
) -> tuple[RepresentativeLPVSourceSpecification, ...]:
    """Validate and normalize an ordered representative-source manifest."""
    if isinstance(manifest, (str, bytes, bytearray)):
        raise TypeError("manifest must be a sequence of source entries.")

    try:
        entries = tuple(manifest)
    except TypeError as exception:
        raise TypeError(
            "manifest must be a sequence of source entries."
        ) from exception

    if not entries:
        raise ValueError("manifest must contain at least one source.")

    normalized = []
    seen_source_ids = set()

    for index, entry in enumerate(entries):
        if isinstance(entry, RepresentativeLPVSourceSpecification):
            specification = entry
        elif isinstance(entry, Mapping):
            specification = (
                RepresentativeLPVSourceSpecification.from_mapping(entry)
            )
        else:
            raise TypeError(
                "manifest entry "
                f"{index} must be a source specification or mapping."
            )

        if specification.source_id in seen_source_ids:
            raise ValueError(
                "duplicate source_id in representative LPV manifest: "
                f"{specification.source_id}"
            )
        seen_source_ids.add(specification.source_id)
        normalized.append(specification)

    return tuple(normalized)


def _representative_lpv_safe_path_component(
    value: Any,
    *,
    fallback: str = "source",
) -> str:
    """Return a deterministic filesystem-safe path component."""
    text = str(value or "").strip()
    safe = "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in text
    ).strip("._")
    return safe or fallback


def _representative_lpv_source_output_paths(
    output_root: str,
    source_id: str,
) -> tuple[str, str]:
    root = str(output_root or "").strip().rstrip("/")
    if not root:
        raise ValueError("output_root must be non-empty.")

    safe_source_id = _representative_lpv_safe_path_component(
        source_id
    )
    source_output_dir = f"{root}/sources/{safe_source_id}"
    source_report_path = f"{source_output_dir}/report.json"
    return source_output_dir, source_report_path


def _representative_lpv_source_failure(
    *,
    stage: str,
    exception: Exception,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "failure_code": f"{stage}_failed",
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
    }


@dataclass(frozen=True)
class RepresentativeLPVBatchReport:
    """Failure-aware, non-selecting D3 report for an ordered source set."""

    batch_id: str
    source_manifest: Sequence[
        RepresentativeLPVSourceSpecification | Mapping[str, Any]
    ]
    source_results: Sequence[Mapping[str, Any]]
    output_root: str = "validation_outputs/d3_real_lpv"
    models: Sequence[str] = DEFAULT_REPRESENTATIVE_LPV_MODELS
    workflow_configuration: Mapping[str, Any] = field(
        default_factory=dict
    )
    runtime_environment: Mapping[str, Any] = field(
        default_factory=dict
    )
    schema_version: str = (
        REPRESENTATIVE_LPV_VALIDATION_SCHEMA_VERSION
    )
    advisory_only: bool = True
    automatic_model_selection_applied: bool = False
    selected_model: str | None = None
    writes_outputs: bool = False
    automatic_constraints_applied: bool = False
    automatic_initialization_applied: bool = False
    notes: Sequence[str] = ()
    extra_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        batch_id = str(self.batch_id or "").strip()
        if not batch_id:
            raise ValueError("batch_id must be non-empty.")
        object.__setattr__(self, "batch_id", batch_id)

        output_root = str(self.output_root or "").strip().rstrip("/")
        if not output_root:
            raise ValueError("output_root must be non-empty.")
        object.__setattr__(self, "output_root", output_root)

        manifest = validate_representative_lpv_source_manifest(
            self.source_manifest
        )
        object.__setattr__(self, "source_manifest", manifest)

        models = tuple(str(model).strip() for model in self.models)
        if not models or any(not model for model in models):
            raise ValueError("models must contain non-empty model names.")
        if len(set(models)) != len(models):
            raise ValueError("models must not contain duplicates.")
        if "2DAchromatic" in models:
            raise ValueError(
                "2DAchromatic is outside the representative LPV "
                "candidate set."
            )
        object.__setattr__(self, "models", models)

        if not isinstance(self.workflow_configuration, Mapping):
            raise TypeError(
                "workflow_configuration must be a mapping."
            )
        object.__setattr__(
            self,
            "workflow_configuration",
            dict(_json_safe(self.workflow_configuration)),
        )

        if not isinstance(self.runtime_environment, Mapping):
            raise TypeError("runtime_environment must be a mapping.")
        object.__setattr__(
            self,
            "runtime_environment",
            dict(_json_safe(self.runtime_environment)),
        )

        if len(self.source_results) != len(manifest):
            raise ValueError(
                "source_results must contain one row per manifest source."
            )

        normalized_results = []
        expected_ids = tuple(item.source_id for item in manifest)
        actual_ids = []

        for row in self.source_results:
            if not isinstance(row, Mapping):
                raise TypeError(
                    "each source result must be a mapping."
                )
            normalized = dict(_json_safe(row))
            source_id = str(normalized.get("source_id") or "")
            status = normalized.get("status")
            if status not in {"completed", "failed"}:
                raise ValueError(
                    "source result status must be 'completed' or "
                    "'failed'."
                )
            if status == "completed":
                if normalized.get("validation_report") is None:
                    raise ValueError(
                        "completed source results require a validation "
                        "report."
                    )
                if normalized.get("failure") is not None:
                    raise ValueError(
                        "completed source results cannot contain a "
                        "failure."
                    )
            else:
                if normalized.get("failure") is None:
                    raise ValueError(
                        "failed source results require a failure record."
                    )

            actual_ids.append(source_id)
            normalized_results.append(normalized)

        if tuple(actual_ids) != expected_ids:
            raise ValueError(
                "source_results must preserve manifest source order."
            )

        object.__setattr__(
            self,
            "source_results",
            tuple(normalized_results),
        )

        if self.advisory_only is not True:
            raise ValueError(
                "representative LPV batch reports must remain "
                "advisory-only."
            )
        if self.automatic_model_selection_applied is not False:
            raise ValueError(
                "representative LPV batch reports cannot apply "
                "automatic model selection."
            )
        if self.selected_model is not None:
            raise ValueError(
                "representative LPV batch reports require "
                "selected_model=None."
            )
        if self.writes_outputs is not False:
            raise ValueError(
                "the protocol runner must not write output artifacts."
            )
        if self.automatic_constraints_applied is not False:
            raise ValueError(
                "the D3 protocol runner cannot apply constraints "
                "automatically."
            )
        if self.automatic_initialization_applied is not False:
            raise ValueError(
                "the D3 protocol runner cannot apply initialization "
                "automatically."
            )

        object.__setattr__(
            self,
            "schema_version",
            str(self.schema_version),
        )
        object.__setattr__(
            self,
            "notes",
            tuple(str(item) for item in self.notes),
        )
        object.__setattr__(
            self,
            "extra_fields",
            dict(_json_safe(self.extra_fields)),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> RepresentativeLPVBatchReport:
        """Reconstruct a batch report from its JSON-safe envelope."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")

        known = {
            "schema_version",
            "batch_id",
            "source_manifest",
            "source_results",
            "output_root",
            "models",
            "workflow_configuration",
            "runtime_environment",
            "n_sources",
            "n_completed_sources",
            "n_failed_sources",
            "source_status_counts",
            "advisory_only",
            "automatic_model_selection_applied",
            "selected_model",
            "writes_outputs",
            "automatic_constraints_applied",
            "automatic_initialization_applied",
            "notes",
            "extra_fields",
        }
        extras = dict(payload.get("extra_fields") or {})
        for key, value in payload.items():
            if key not in known:
                extras[str(key)] = _json_safe(value)

        return cls(
            schema_version=payload.get(
                "schema_version",
                REPRESENTATIVE_LPV_VALIDATION_SCHEMA_VERSION,
            ),
            batch_id=payload.get("batch_id", ""),
            source_manifest=payload.get("source_manifest") or (),
            source_results=payload.get("source_results") or (),
            output_root=payload.get(
                "output_root",
                "validation_outputs/d3_real_lpv",
            ),
            models=payload.get(
                "models",
                DEFAULT_REPRESENTATIVE_LPV_MODELS,
            ),
            workflow_configuration=payload.get(
                "workflow_configuration"
            )
            or {},
            runtime_environment=payload.get("runtime_environment")
            or {},
            advisory_only=payload.get("advisory_only", True),
            automatic_model_selection_applied=payload.get(
                "automatic_model_selection_applied",
                False,
            ),
            selected_model=payload.get("selected_model"),
            writes_outputs=payload.get("writes_outputs", False),
            automatic_constraints_applied=payload.get(
                "automatic_constraints_applied",
                False,
            ),
            automatic_initialization_applied=payload.get(
                "automatic_initialization_applied",
                False,
            ),
            notes=payload.get("notes") or (),
            extra_fields=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-safe batch report envelope."""
        completed = sum(
            row["status"] == "completed"
            for row in self.source_results
        )
        failed = sum(
            row["status"] == "failed"
            for row in self.source_results
        )

        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "source_manifest": [
                item.to_dict() for item in self.source_manifest
            ],
            "source_results": _json_safe(self.source_results),
            "output_root": self.output_root,
            "models": list(self.models),
            "workflow_configuration": _json_safe(
                self.workflow_configuration
            ),
            "runtime_environment": _json_safe(
                self.runtime_environment
            ),
            "n_sources": len(self.source_results),
            "n_completed_sources": completed,
            "n_failed_sources": failed,
            "source_status_counts": {
                "completed": completed,
                "failed": failed,
            },
            "advisory_only": True,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "writes_outputs": False,
            "automatic_constraints_applied": False,
            "automatic_initialization_applied": False,
            "notes": list(self.notes),
            "extra_fields": _json_safe(self.extra_fields),
        }


def _default_representative_lpv_source_loader(
    specification: RepresentativeLPVSourceSpecification,
) -> Any:
    from .lightcurve import Lightcurve

    return Lightcurve.from_csv(
        specification.source_path,
        check_sampling=False,
        max_samples=None,
        max_samples_per_band=None,
    )


def run_representative_lpv_validation_batch(
    manifest: Sequence[
        RepresentativeLPVSourceSpecification | Mapping[str, Any]
    ],
    *,
    source_loader: Callable[
        [RepresentativeLPVSourceSpecification],
        Any,
    ] | None = None,
    workflow_runner: Callable[..., Mapping[str, Any]] | None = None,
    models: Sequence[str] = DEFAULT_REPRESENTATIVE_LPV_MODELS,
    workflow_kwargs: Mapping[str, Any] | None = None,
    output_root: str = "validation_outputs/d3_real_lpv",
    batch_id: str = "d3-representative-observed-lpv",
    notes: Sequence[str] = (),
) -> RepresentativeLPVBatchReport:
    """Run a failure-aware D3 protocol over an ordered source manifest.

    This protocol layer computes reports and deterministic intended output
    paths.  It does not write files, install a model, apply constraints, or
    apply initialization automatically.
    """
    specifications = validate_representative_lpv_source_manifest(
        manifest
    )

    resolved_models = tuple(str(model).strip() for model in models)
    if not resolved_models or any(
        not model for model in resolved_models
    ):
        raise ValueError("models must contain non-empty model names.")
    if len(set(resolved_models)) != len(resolved_models):
        raise ValueError("models must not contain duplicates.")
    if "2DAchromatic" in resolved_models:
        raise ValueError(
            "2DAchromatic is outside the representative LPV "
            "candidate set."
        )

    loader = (
        source_loader
        if source_loader is not None
        else _default_representative_lpv_source_loader
    )
    if not callable(loader):
        raise TypeError("source_loader must be callable.")
    if workflow_runner is not None and not callable(workflow_runner):
        raise TypeError("workflow_runner must be callable.")

    base_workflow_kwargs = dict(workflow_kwargs or {})
    runtime_environment = (
        build_representative_lpv_runtime_environment()
    )
    results = []

    for specification in specifications:
        source_output_dir, source_report_path = (
            _representative_lpv_source_output_paths(
                output_root,
                specification.source_id,
            )
        )

        base_result = {
            "source_id": specification.source_id,
            "source_specification": specification.to_dict(),
            "source_output_dir": source_output_dir,
            "source_report_path": source_report_path,
        }

        with _temporary_representative_lpv_seed(
            specification.seed
        ):
            try:
                lightcurve = loader(specification)
            except Exception as exception:
                results.append(
                    {
                        **base_result,
                        "status": "failed",
                        "execution_seed": specification.seed,
                        "seed_applied": True,
                        "seed_scope": (
                            "source_loading_and_advisory_workflow"
                        ),
                        "validation_report": None,
                        "failure": (
                            _representative_lpv_source_failure(
                                stage="source_loading",
                                exception=exception,
                            )
                        ),
                    }
                )
                continue

            try:
                validation_report = (
                    run_representative_lpv_validation(
                        lightcurve,
                        source_id=specification.source_id,
                        description=specification.description,
                        models=resolved_models,
                        workflow_runner=workflow_runner,
                        workflow_kwargs=base_workflow_kwargs,
                        notes=(
                            (
                                "sample_role="
                                f"{specification.sample_role}"
                            ),
                            (
                                "selection_reason="
                                f"{specification.selection_reason}"
                            ),
                            f"seed={specification.seed}",
                        ),
                    )
                )
            except Exception as exception:
                results.append(
                    {
                        **base_result,
                        "status": "failed",
                        "execution_seed": specification.seed,
                        "seed_applied": True,
                        "seed_scope": (
                            "source_loading_and_advisory_workflow"
                        ),
                        "validation_report": None,
                        "failure": (
                            _representative_lpv_source_failure(
                                stage="advisory_workflow",
                                exception=exception,
                            )
                        ),
                    }
                )
                continue

            results.append(
                {
                    **base_result,
                    "status": "completed",
                    "execution_seed": specification.seed,
                    "seed_applied": True,
                    "seed_scope": (
                        "source_loading_and_advisory_workflow"
                    ),
                    "validation_report": validation_report.to_dict(),
                    "failure": None,
                }
            )

    return RepresentativeLPVBatchReport(
        batch_id=batch_id,
        source_manifest=specifications,
        source_results=results,
        output_root=output_root,
        models=resolved_models,
        workflow_configuration=base_workflow_kwargs,
        runtime_environment=runtime_environment,
        notes=(
            "Source order is preserved from the validated manifest.",
            "A source loading or advisory workflow failure does not "
            "abort later sources.",
            "Output paths are deterministic protocol metadata; this "
            "runner does not write files.",
            "Each source seed controls source loading and advisory "
            "execution while caller RNG states are restored.",
            "Workflow configuration and compact runtime environment "
            "provenance are retained in the batch report.",
            *tuple(str(item) for item in notes),
        ),
    )


def _representative_lpv_write_json(
    path: Any,
    payload: Any,
) -> str:
    """Write strict JSON and return the written path."""
    import json
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(output_path)


def _representative_lpv_write_summary_csv(
    path: Any,
    source_results: Sequence[Mapping[str, Any]],
) -> str:
    """Write the compact source-level D3 status table."""
    import csv
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        "source_index",
        "source_id",
        "status",
        "sample_role",
        "seed",
        "failure_stage",
        "failure_code",
        "exception_type",
        "exception_message",
        "source_artifact_path",
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for source_index, row in enumerate(source_results):
            specification = row.get("source_specification") or {}
            failure = row.get("failure") or {}
            writer.writerow(
                {
                    "source_index": source_index,
                    "source_id": row.get("source_id"),
                    "status": row.get("status"),
                    "sample_role": specification.get("sample_role"),
                    "seed": specification.get("seed"),
                    "failure_stage": failure.get("stage"),
                    "failure_code": failure.get("failure_code"),
                    "exception_type": failure.get("exception_type"),
                    "exception_message": failure.get(
                        "exception_message"
                    ),
                    "source_artifact_path": row.get(
                        "exported_source_artifact_path"
                    ),
                }
            )

    return str(output_path)


def export_representative_lpv_batch_report(
    report: RepresentativeLPVBatchReport | Mapping[str, Any],
    output_dir: Any | None = None,
) -> dict[str, Any]:
    """Serialize an already-computed D3 batch report.

    This is an output layer only.  It does not load light curves, run fits,
    score or rank candidates, select or install a model, or automatically
    apply constraints or initialization.
    """
    from pathlib import Path

    if isinstance(report, Mapping):
        resolved_report = RepresentativeLPVBatchReport.from_mapping(
            report
        )
    elif isinstance(report, RepresentativeLPVBatchReport):
        resolved_report = report
    else:
        raise TypeError(
            "report must be a RepresentativeLPVBatchReport or mapping."
        )

    destination = (
        resolved_report.output_root
        if output_dir is None
        else output_dir
    )
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)

    payload = resolved_report.to_dict()

    batch_report_path = _representative_lpv_write_json(
        root / "batch_report.json",
        payload,
    )
    source_manifest_path = _representative_lpv_write_json(
        root / "source_manifest.json",
        payload["source_manifest"],
    )

    exported_source_rows = []
    source_artifacts = []

    for source_index, row in enumerate(payload["source_results"]):
        source_id = str(row["source_id"])
        safe_source_id = _representative_lpv_safe_path_component(
            source_id,
            fallback=f"source_{source_index:04d}",
        )
        source_dir = root / "sources" / safe_source_id
        source_dir.mkdir(parents=True, exist_ok=True)

        source_result_path = _representative_lpv_write_json(
            source_dir / "source_result.json",
            row,
        )

        if row["status"] == "completed":
            source_artifact_kind = "validation_report"
            source_artifact_path = _representative_lpv_write_json(
                source_dir / "report.json",
                row["validation_report"],
            )
        else:
            source_artifact_kind = "failure"
            source_artifact_path = _representative_lpv_write_json(
                source_dir / "failure.json",
                row["failure"],
            )

        exported_row = {
            **row,
            "exported_source_result_path": source_result_path,
            "exported_source_artifact_kind": source_artifact_kind,
            "exported_source_artifact_path": source_artifact_path,
        }
        exported_source_rows.append(exported_row)
        source_artifacts.append(
            {
                "source_index": source_index,
                "source_id": source_id,
                "status": row["status"],
                "source_dir": str(source_dir),
                "source_result_path": source_result_path,
                "artifact_kind": source_artifact_kind,
                "artifact_path": source_artifact_path,
            }
        )

    source_summary_path = _representative_lpv_write_summary_csv(
        root / "source_summary.csv",
        exported_source_rows,
    )

    export_manifest = {
        "kind": "representative_lpv_batch_report_export",
        "schema_version": resolved_report.schema_version,
        "batch_id": resolved_report.batch_id,
        "output_dir": str(root),
        "batch_report_path": batch_report_path,
        "source_manifest_path": source_manifest_path,
        "source_summary_path": source_summary_path,
        "source_artifacts": source_artifacts,
        "n_sources": payload["n_sources"],
        "n_completed_sources": payload["n_completed_sources"],
        "n_failed_sources": payload["n_failed_sources"],
        "advisory_only": True,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "automatic_constraints_applied": False,
        "automatic_initialization_applied": False,
        "source_report_writes_only": True,
        "runs_fits": False,
    }
    export_manifest_path = root / "export_manifest.json"
    export_manifest["export_manifest_path"] = str(
        export_manifest_path
    )
    _representative_lpv_write_json(
        export_manifest_path,
        export_manifest,
    )
    return dict(_json_safe(export_manifest))

