"""Structured, conservative conclusions for wavelength-advisory workflows.

The wavelength-advisory workflow compares complete Gaussian-process
configurations.  Its current training-residual score is heuristic rather than a
formal model-comparison statistic.  This module converts existing attempt
statuses, hypothesis metadata, and ranking availability into explicit
conclusions without selecting a model or attributing a score difference to a
mean or covariance mechanism that was not isolated by the compared configs.

The synthesis is report-only.  It does not fit models, change scores, alter
comparison eligibility, apply constraints, or mutate a light curve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from .wavelength_hypotheses import (
    WavelengthCovarianceStructure,
    WavelengthMeanStructure,
)
from .wavelength_results import WavelengthModelAttemptResult
from .wavelength_status import (
    AttemptDisposition,
    ComparisonEligibility,
    ScientificUsability,
    TechnicalOutcome,
)

WAVELENGTH_ADVISORY_CONCLUSION_SCHEMA_VERSION = (
    "pgmuvi-wavelength-advisory-conclusions-v1"
)

__all__ = [
    "WAVELENGTH_ADVISORY_CONCLUSION_SCHEMA_VERSION",
    "WavelengthAdvisoryConclusion",
    "WavelengthConclusionDisposition",
    "WavelengthConclusionScope",
    "WavelengthUnresolvedAmbiguity",
    "synthesize_wavelength_advisory_conclusions",
]


class _StringEnum(str, Enum):
    """Enum whose values serialize and display as stable public strings."""

    def __str__(self) -> str:
        return self.value


class WavelengthConclusionScope(_StringEnum):
    """Scientific level addressed by one advisory conclusion."""

    COMPLETE_CONFIGURATION = "complete_configuration"
    MEAN_STRUCTURE = "mean_structure"
    COVARIANCE_STRUCTURE = "covariance_structure"
    WORKFLOW = "workflow"


class WavelengthConclusionDisposition(_StringEnum):
    """Conservative state assigned by the advisory synthesis."""

    REMAINS_PLAUSIBLE = "remains_plausible"
    WEAKENED = "weakened"
    TECHNICALLY_UNEVALUABLE = "technically_unevaluable"
    SCIENTIFICALLY_AMBIGUOUS = "scientifically_ambiguous"
    INCOMPARABLE = "incomparable"


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
    return str(value)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


@dataclass(frozen=True)
class WavelengthAdvisoryConclusion:
    """One explicit, JSON-safe advisory conclusion."""

    conclusion_id: str
    scope: WavelengthConclusionScope
    subject: str
    disposition: WavelengthConclusionDisposition
    summary: str
    models: tuple[str, ...] = ()
    model_kernel_config_ids: tuple[str, ...] = ()
    mean_structures: tuple[str, ...] = ()
    covariance_structures: tuple[str, ...] = ()
    evidence_basis: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    schema_version: str = WAVELENGTH_ADVISORY_CONCLUSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("conclusion_id", "subject", "summary"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty.")
        object.__setattr__(self, "conclusion_id", str(self.conclusion_id))
        object.__setattr__(self, "subject", str(self.subject))
        object.__setattr__(self, "summary", str(self.summary))
        object.__setattr__(self, "models", _string_tuple(self.models))
        object.__setattr__(
            self,
            "model_kernel_config_ids",
            _string_tuple(self.model_kernel_config_ids),
        )
        object.__setattr__(
            self, "mean_structures", _string_tuple(self.mean_structures)
        )
        object.__setattr__(
            self,
            "covariance_structures",
            _string_tuple(self.covariance_structures),
        )
        object.__setattr__(
            self, "evidence_basis", _json_safe(dict(self.evidence_basis))
        )
        object.__setattr__(self, "limitations", _string_tuple(self.limitations))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable public mapping representation."""
        return {
            "schema_version": self.schema_version,
            "kind": "wavelength_advisory_conclusion",
            "conclusion_id": self.conclusion_id,
            "scope": self.scope.value,
            "subject": self.subject,
            "disposition": self.disposition.value,
            "summary": self.summary,
            "models": list(self.models),
            "model_kernel_config_ids": list(self.model_kernel_config_ids),
            "mean_structures": list(self.mean_structures),
            "covariance_structures": list(self.covariance_structures),
            "evidence_basis": _json_safe(self.evidence_basis),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class WavelengthUnresolvedAmbiguity:
    """One unresolved limitation that prevents a stronger interpretation."""

    code: str
    summary: str
    affected_scopes: tuple[WavelengthConclusionScope, ...]
    models: tuple[str, ...] = ()
    resolution_requirements: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WAVELENGTH_ADVISORY_CONCLUSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.code).strip():
            raise ValueError("code must be non-empty.")
        if not str(self.summary).strip():
            raise ValueError("summary must be non-empty.")
        object.__setattr__(self, "code", str(self.code))
        object.__setattr__(self, "summary", str(self.summary))
        object.__setattr__(self, "affected_scopes", tuple(self.affected_scopes))
        object.__setattr__(self, "models", _string_tuple(self.models))
        object.__setattr__(
            self,
            "resolution_requirements",
            _string_tuple(self.resolution_requirements),
        )
        object.__setattr__(self, "details", _json_safe(dict(self.details)))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable public mapping representation."""
        return {
            "schema_version": self.schema_version,
            "kind": "wavelength_unresolved_ambiguity",
            "code": self.code,
            "summary": self.summary,
            "affected_scopes": [scope.value for scope in self.affected_scopes],
            "models": list(self.models),
            "resolution_requirements": list(self.resolution_requirements),
            "details": _json_safe(self.details),
        }


def _outcomes_from_workflow(workflow: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    run_report = workflow.get("run_report")
    run_report = run_report if isinstance(run_report, Mapping) else {}
    outcomes = (
        run_report.get("model_kernel_config_results")
        or run_report.get("outcomes")
        or workflow.get("model_kernel_config_results")
        or workflow.get("outcomes")
        or []
    )
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, str):
        return []
    return [item for item in outcomes if isinstance(item, Mapping)]


def _quality_rows_from_workflow(
    workflow: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    quality_report = workflow.get("quality_report")
    quality_report = quality_report if isinstance(quality_report, Mapping) else {}
    rows = quality_report.get("ranked_results") or quality_report.get("results") or []
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return []
    return [item for item in rows if isinstance(item, Mapping)]


def _quality_row_for_attempt(
    attempt: WavelengthModelAttemptResult,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    for row in rows:
        if (
            attempt.model_kernel_config_id is not None
            and row.get("model_kernel_config_id") == attempt.model_kernel_config_id
        ):
            return row
    matching = [row for row in rows if row.get("model") == attempt.model]
    if len(matching) == 1:
        return matching[0]
    if attempt.rank is not None:
        for row in matching:
            if row.get("rank") == attempt.rank:
                return row
    return {}


def _configuration_disposition(
    attempt: WavelengthModelAttemptResult,
) -> WavelengthConclusionDisposition:
    status = attempt.status
    if status.disposition in {
        AttemptDisposition.NOT_ATTEMPTED,
        AttemptDisposition.SKIPPED,
    }:
        return WavelengthConclusionDisposition.TECHNICALLY_UNEVALUABLE
    if status.technical_outcome is TechnicalOutcome.FAILED:
        return WavelengthConclusionDisposition.TECHNICALLY_UNEVALUABLE
    if status.scientific_usability is ScientificUsability.UNUSABLE:
        return WavelengthConclusionDisposition.TECHNICALLY_UNEVALUABLE
    if attempt.legacy_payload.get("hard_exclusion") is True:
        return WavelengthConclusionDisposition.WEAKENED
    if status.comparison_eligibility is not ComparisonEligibility.ELIGIBLE:
        return WavelengthConclusionDisposition.INCOMPARABLE
    return WavelengthConclusionDisposition.REMAINS_PLAUSIBLE


def _configuration_summary(
    attempt: WavelengthModelAttemptResult,
    disposition: WavelengthConclusionDisposition,
    ranking_status: str,
    top_ranked_model: str | None,
) -> str:
    model = attempt.model or attempt.model_kernel_config_id or "unknown configuration"
    if disposition is WavelengthConclusionDisposition.TECHNICALLY_UNEVALUABLE:
        return (
            f"{model} could not be scientifically evaluated because the attempt "
            "did not produce a usable completed fit."
        )
    if disposition is WavelengthConclusionDisposition.WEAKENED:
        return (
            f"{model} is weakened by an explicit advisory exclusion, but this "
            "does not constitute a formal model rejection."
        )
    if disposition is WavelengthConclusionDisposition.INCOMPARABLE:
        return (
            f"{model} produced some usable information but is not eligible for "
            "the current like-for-like fit-quality comparison."
        )
    if ranking_status == "available" and model == top_ranked_model:
        return (
            f"{model} remains plausible and is top ranked by the current "
            "training-residual heuristic; it is not selected automatically."
        )
    if ranking_status == "available":
        return (
            f"{model} remains plausible despite not being top ranked by the "
            "current training-residual heuristic."
        )
    return (
        f"{model} remains plausible as a completed comparable configuration, "
        "but no comparative ranking is available."
    )


def _has_isolating_contrast(
    attempt: WavelengthModelAttemptResult,
    eligible_attempts: Sequence[WavelengthModelAttemptResult],
    *,
    scope: WavelengthConclusionScope,
) -> bool:
    hypothesis = attempt.hypothesis
    if hypothesis is None:
        return False
    for other in eligible_attempts:
        if other is attempt or other.hypothesis is None:
            continue
        if scope is WavelengthConclusionScope.MEAN_STRUCTURE:
            if (
                other.hypothesis.covariance_structure
                is hypothesis.covariance_structure
                and other.hypothesis.mean_structure is not hypothesis.mean_structure
            ):
                return True
        elif scope is WavelengthConclusionScope.COVARIANCE_STRUCTURE:
            if (
                other.hypothesis.mean_structure is hypothesis.mean_structure
                and other.hypothesis.covariance_structure
                is not hypothesis.covariance_structure
            ):
                return True
    return False


def _aggregate_structure_conclusions(
    attempts: Sequence[WavelengthModelAttemptResult],
    configuration_dispositions: Mapping[str, WavelengthConclusionDisposition],
    *,
    scope: WavelengthConclusionScope,
) -> list[WavelengthAdvisoryConclusion]:
    eligible = [
        attempt
        for attempt in attempts
        if attempt.status.comparison_eligibility is ComparisonEligibility.ELIGIBLE
    ]
    grouped: dict[str, list[WavelengthModelAttemptResult]] = {}
    for attempt in attempts:
        if attempt.hypothesis is None:
            structure = "unknown"
        elif scope is WavelengthConclusionScope.MEAN_STRUCTURE:
            structure = attempt.hypothesis.mean_structure.value
        else:
            structure = attempt.hypothesis.covariance_structure.value
        grouped.setdefault(structure, []).append(attempt)

    conclusions = []
    for structure, members in grouped.items():
        member_keys = [
            member.model_kernel_config_id or member.model or "unknown" for member in members
        ]
        member_dispositions = [
            configuration_dispositions[key] for key in member_keys
        ]
        comparable = [
            member
            for member in members
            if member.status.comparison_eligibility is ComparisonEligibility.ELIGIBLE
        ]
        isolating = any(
            _has_isolating_contrast(member, eligible, scope=scope)
            for member in comparable
        )

        if not comparable and all(
            item is WavelengthConclusionDisposition.TECHNICALLY_UNEVALUABLE
            for item in member_dispositions
        ):
            disposition = WavelengthConclusionDisposition.TECHNICALLY_UNEVALUABLE
            summary = (
                f"The {structure} hypothesis could not be evaluated because no "
                "configuration representing it produced a usable comparable fit."
            )
        elif not comparable:
            disposition = WavelengthConclusionDisposition.INCOMPARABLE
            summary = (
                f"The {structure} hypothesis is represented, but none of its "
                "configurations is eligible for the current comparison."
            )
        elif not isolating:
            disposition = WavelengthConclusionDisposition.SCIENTIFICALLY_AMBIGUOUS
            mechanism = "mean" if scope is WavelengthConclusionScope.MEAN_STRUCTURE else "covariance"
            summary = (
                f"The {structure} {mechanism} hypothesis is represented by a "
                "comparable fit, but the current candidate set does not isolate "
                f"that {mechanism} mechanism from the other GP components."
            )
        else:
            disposition = WavelengthConclusionDisposition.REMAINS_PLAUSIBLE
            mechanism = "mean" if scope is WavelengthConclusionScope.MEAN_STRUCTURE else "covariance"
            summary = (
                f"The {structure} {mechanism} hypothesis remains plausible and "
                f"has at least one same-other-axis contrast in the candidate set."
            )

        conclusions.append(
            WavelengthAdvisoryConclusion(
                conclusion_id=f"{scope.value}:{structure}",
                scope=scope,
                subject=structure,
                disposition=disposition,
                summary=summary,
                models=tuple(
                    member.model or member.model_kernel_config_id or "unknown"
                    for member in members
                ),
                model_kernel_config_ids=tuple(
                    member.model_kernel_config_id
                    or member.model
                    or "unknown"
                    for member in members
                ),
                mean_structures=(structure,)
                if scope is WavelengthConclusionScope.MEAN_STRUCTURE
                else tuple(
                    sorted(
                        {
                            member.hypothesis.mean_structure.value
                            if member.hypothesis is not None
                            else WavelengthMeanStructure.UNKNOWN.value
                            for member in members
                        }
                    )
                ),
                covariance_structures=(structure,)
                if scope is WavelengthConclusionScope.COVARIANCE_STRUCTURE
                else tuple(
                    sorted(
                        {
                            member.hypothesis.covariance_structure.value
                            if member.hypothesis is not None
                            else WavelengthCovarianceStructure.UNKNOWN.value
                            for member in members
                        }
                    )
                ),
                evidence_basis={
                    "n_configurations": len(members),
                    "n_comparison_eligible": len(comparable),
                    "isolating_contrast_available": isolating,
                },
                limitations=(
                    "The workflow compares complete GP configurations.",
                    "The current score is a training-residual heuristic, not a formal comparison statistic.",
                ),
            )
        )
    return conclusions


def _build_ambiguities(
    attempts: Sequence[WavelengthModelAttemptResult],
    conclusions: Sequence[WavelengthAdvisoryConclusion],
    *,
    ranking_status: str,
    top_ranked_model: str | None,
) -> list[WavelengthUnresolvedAmbiguity]:
    ambiguities: list[WavelengthUnresolvedAmbiguity] = []
    models = tuple(
        attempt.model or attempt.model_kernel_config_id or "unknown"
        for attempt in attempts
    )
    failed = [
        attempt
        for attempt in attempts
        if attempt.status.technical_outcome is TechnicalOutcome.FAILED
    ]

    if not attempts:
        ambiguities.append(
            WavelengthUnresolvedAmbiguity(
                code="no_candidate_attempts",
                summary="No model/kernel configuration attempts are available.",
                affected_scopes=(WavelengthConclusionScope.WORKFLOW,),
                resolution_requirements=(
                    "Run at least one wavelength-model configuration.",
                ),
            )
        )
    if ranking_status == "available":
        ambiguities.append(
            WavelengthUnresolvedAmbiguity(
                code="training_residual_ranking_is_heuristic",
                summary=(
                    "The available ranking is based on training-residual fit quality "
                    "and is not a formal or held-out model comparison."
                ),
                affected_scopes=(
                    WavelengthConclusionScope.COMPLETE_CONFIGURATION,
                    WavelengthConclusionScope.MEAN_STRUCTURE,
                    WavelengthConclusionScope.COVARIANCE_STRUCTURE,
                ),
                models=models,
                resolution_requirements=(
                    "Add held-out or cross-validated predictive comparison.",
                    "Assess seed and subsampling stability.",
                ),
                details={"top_ranked_model": top_ranked_model},
            )
        )
    elif ranking_status == "single_valid_candidate":
        ambiguities.append(
            WavelengthUnresolvedAmbiguity(
                code="single_valid_candidate_not_comparative",
                summary=(
                    "Only one candidate has valid fit-quality diagnostics, so it "
                    "cannot establish a comparative preference."
                ),
                affected_scopes=(WavelengthConclusionScope.WORKFLOW,),
                models=models,
                resolution_requirements=(
                    "Obtain valid fit-quality diagnostics for another candidate.",
                ),
            )
        )
    else:
        ambiguities.append(
            WavelengthUnresolvedAmbiguity(
                code="comparative_ranking_unavailable",
                summary="No comparative fit-quality ranking is available.",
                affected_scopes=(WavelengthConclusionScope.WORKFLOW,),
                models=models,
                resolution_requirements=(
                    "Resolve fit failures or missing diagnostics before comparison.",
                ),
            )
        )

    if failed:
        ambiguities.append(
            WavelengthUnresolvedAmbiguity(
                code="technical_failures_limit_hypothesis_coverage",
                summary=(
                    "Technical failures prevent some configured hypotheses from "
                    "contributing comparable evidence."
                ),
                affected_scopes=(
                    WavelengthConclusionScope.COMPLETE_CONFIGURATION,
                    WavelengthConclusionScope.MEAN_STRUCTURE,
                    WavelengthConclusionScope.COVARIANCE_STRUCTURE,
                ),
                models=tuple(
                    attempt.model or attempt.model_kernel_config_id or "unknown"
                    for attempt in failed
                ),
                resolution_requirements=(
                    "Resolve the recorded failure stages and rerun affected candidates.",
                ),
            )
        )

    ambiguous_scopes = {
        conclusion.scope
        for conclusion in conclusions
        if conclusion.disposition
        is WavelengthConclusionDisposition.SCIENTIFICALLY_AMBIGUOUS
    }
    if WavelengthConclusionScope.MEAN_STRUCTURE in ambiguous_scopes:
        ambiguities.append(
            WavelengthUnresolvedAmbiguity(
                code="mean_effect_not_isolated",
                summary=(
                    "At least one wavelength-mean hypothesis lacks a same-covariance "
                    "contrast, so its separate contribution cannot be attributed."
                ),
                affected_scopes=(WavelengthConclusionScope.MEAN_STRUCTURE,),
                models=models,
                resolution_requirements=(
                    "Compare alternative means under the same covariance parameterization.",
                ),
            )
        )
    if WavelengthConclusionScope.COVARIANCE_STRUCTURE in ambiguous_scopes:
        ambiguities.append(
            WavelengthUnresolvedAmbiguity(
                code="covariance_effect_not_isolated",
                summary=(
                    "At least one wavelength-covariance hypothesis lacks a same-mean "
                    "contrast, so its separate contribution cannot be attributed."
                ),
                affected_scopes=(WavelengthConclusionScope.COVARIANCE_STRUCTURE,),
                models=models,
                resolution_requirements=(
                    "Compare alternative covariances under the same mean structure.",
                ),
            )
        )
    return ambiguities


def synthesize_wavelength_advisory_conclusions(
    workflow_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Synthesize explicit conclusions and ambiguities from an advisory report.

    The returned mapping is additive and non-selecting.  It may be attached to
    an existing workflow under ``advisory_conclusions`` and
    ``unresolved_ambiguities`` without changing the legacy fields.
    """
    if not isinstance(workflow_report, Mapping):
        raise TypeError("workflow_report must be a mapping")

    raw_attempts = _outcomes_from_workflow(workflow_report)
    attempts = [
        WavelengthModelAttemptResult.from_mapping(payload)
        for payload in raw_attempts
    ]
    quality_rows = _quality_rows_from_workflow(workflow_report)
    quality_report = workflow_report.get("quality_report")
    quality_report = quality_report if isinstance(quality_report, Mapping) else {}
    ranking_status = str(
        workflow_report.get("fit_quality_ranking_status")
        or quality_report.get("ranking_status")
        or quality_report.get("fit_quality_ranking_status")
        or "unavailable"
    )
    top_ranked_value = workflow_report.get("top_ranked_model")
    if top_ranked_value is None:
        top_ranked_value = quality_report.get("top_ranked_model")
    top_ranked_model = (
        str(top_ranked_value) if top_ranked_value is not None else None
    )

    configuration_conclusions: list[WavelengthAdvisoryConclusion] = []
    dispositions: dict[str, WavelengthConclusionDisposition] = {}
    for attempt in attempts:
        key = attempt.model_kernel_config_id or attempt.model or "unknown"
        disposition = _configuration_disposition(attempt)
        dispositions[key] = disposition
        quality_row = _quality_row_for_attempt(attempt, quality_rows)
        hypothesis = attempt.hypothesis
        configuration_conclusions.append(
            WavelengthAdvisoryConclusion(
                conclusion_id=f"complete_configuration:{key}",
                scope=WavelengthConclusionScope.COMPLETE_CONFIGURATION,
                subject=attempt.model or key,
                disposition=disposition,
                summary=_configuration_summary(
                    attempt,
                    disposition,
                    ranking_status,
                    top_ranked_model,
                ),
                models=(attempt.model or key,),
                model_kernel_config_ids=(key,),
                mean_structures=(
                    hypothesis.mean_structure.value if hypothesis is not None else "unknown",
                ),
                covariance_structures=(
                    hypothesis.covariance_structure.value
                    if hypothesis is not None
                    else "unknown",
                ),
                evidence_basis={
                    **attempt.status.to_dict(),
                    "quality_rank": quality_row.get("quality_rank"),
                    "fit_quality_score": quality_row.get("fit_quality_score"),
                    "score_kind": quality_report.get("score_kind"),
                    "is_top_ranked": quality_row.get("is_top_ranked"),
                },
                limitations=tuple(
                    dict.fromkeys(
                        (
                            "The configuration contains both a mean and a covariance model.",
                            "The current score is a training-residual heuristic, not a formal comparison statistic.",
                            *(hypothesis.comparison_cautions if hypothesis is not None else ()),
                        )
                    )
                ),
            )
        )

    mean_conclusions = _aggregate_structure_conclusions(
        attempts,
        dispositions,
        scope=WavelengthConclusionScope.MEAN_STRUCTURE,
    )
    covariance_conclusions = _aggregate_structure_conclusions(
        attempts,
        dispositions,
        scope=WavelengthConclusionScope.COVARIANCE_STRUCTURE,
    )
    conclusions = [
        *configuration_conclusions,
        *mean_conclusions,
        *covariance_conclusions,
    ]
    ambiguities = _build_ambiguities(
        attempts,
        conclusions,
        ranking_status=ranking_status,
        top_ranked_model=top_ranked_model,
    )

    disposition_counts: dict[str, int] = {}
    for conclusion in conclusions:
        name = conclusion.disposition.value
        disposition_counts[name] = disposition_counts.get(name, 0) + 1

    return {
        "schema_version": WAVELENGTH_ADVISORY_CONCLUSION_SCHEMA_VERSION,
        "kind": "wavelength_advisory_conclusion_synthesis",
        "advisory_only": True,
        "automatic_model_selection_applied": False,
        "selected_model": None,
        "fit_quality_ranking_status": ranking_status,
        "top_ranked_model": top_ranked_model,
        "conclusions": [item.to_dict() for item in conclusions],
        "unresolved_ambiguities": [item.to_dict() for item in ambiguities],
        "summary": {
            "n_attempts": len(attempts),
            "n_conclusions": len(conclusions),
            "n_unresolved_ambiguities": len(ambiguities),
            "conclusion_disposition_counts": disposition_counts,
        },
    }
