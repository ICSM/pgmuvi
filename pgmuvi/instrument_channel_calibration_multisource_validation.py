"""Contracts for maintainer-private multi-source calibration validation.

The public records in this module freeze source eligibility, reproducible
selection, leave-one-source-out validation, source balancing, aggregate gates,
and redacted evidence for one exact candidate observational-channel rule.  They
do not read private data, add Parquet support to the public package, execute
numerical fitting, populate a pairing-rule catalogue, or activate calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

INSTRUMENT_CHANNEL_CALIBRATION_MULTISOURCE_PROTOCOL_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-multisource-protocol-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_MULTISOURCE_SUMMARY_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-multisource-summary-v1"
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

__all__ = [
    "INSTRUMENT_CHANNEL_CALIBRATION_MULTISOURCE_PROTOCOL_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_MULTISOURCE_SUMMARY_SCHEMA_VERSION",
    "InstrumentChannelCalibrationCrossSourceValidationMethod",
    "InstrumentChannelCalibrationMultiSourceSelectionMethod",
    "InstrumentChannelCalibrationMultiSourceValidationDisposition",
    "InstrumentChannelCalibrationMultiSourceValidationProtocol",
    "InstrumentChannelCalibrationMultiSourceValidationSummary",
    "InstrumentChannelCalibrationSourceBalanceMethod",
]


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class InstrumentChannelCalibrationMultiSourceSelectionMethod(_StringEnum):
    """Pre-registered, outcome-independent source-selection method."""

    SEEDED_PERMUTATION_FIRST_N = (
        "sorted_eligible_source_ids_seeded_permutation_first_n"
    )


class InstrumentChannelCalibrationCrossSourceValidationMethod(_StringEnum):
    """Method used to test transfer to an unseen astrophysical source."""

    LEAVE_ONE_SOURCE_OUT = "leave_one_astrophysical_source_out"


class InstrumentChannelCalibrationSourceBalanceMethod(_StringEnum):
    """Method preventing one training source from dominating the fit."""

    EQUAL_PAIR_COUNT = (
        "equal_matched_pair_count_seeded_without_replacement"
    )


class InstrumentChannelCalibrationMultiSourceValidationDisposition(_StringEnum):
    """Outcome encoded by the redacted public summary."""

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


def _normalize_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _normalize_sha256(value: Any, *, name: str) -> str:
    normalized = _normalize_text(value, name=name)
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return normalized


def _normalize_positive_int(
    value: Any,
    *,
    name: str,
    minimum: int = 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _normalize_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _normalize_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not boolean.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None:
        invalid = (
            normalized <= minimum
            if strict_minimum
            else normalized < minimum
        )
        if invalid:
            comparator = (
                "greater than" if strict_minimum else "at least"
            )
            raise ValueError(f"{name} must be {comparator} {minimum}.")
    return normalized


def _normalize_optional_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _normalize_float(value, name=name)


def _normalize_enum(
    value: Any,
    enum_type: type[_StringEnum],
    *,
    name: str,
) -> _StringEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{name} must be one of: {choices}.") from exc


def _normalize_text_tuple(
    value: Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings.")
    try:
        normalized = tuple(
            _normalize_text(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of strings.") from exc
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must contain at least one item.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    return normalized


def _normalize_sha256_tuple(
    value: Any,
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of digests.")
    normalized = tuple(
        _normalize_sha256(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    return normalized


@dataclass(frozen=True)
class InstrumentChannelCalibrationMultiSourceValidationProtocol:
    """Frozen contract for a maintainer-private five-source decision."""

    protocol_id: str
    protocol_version: str
    candidate_protocol_id: str
    candidate_protocol_version: str
    candidate_protocol_sha256: str
    rule_id: str
    reference_channel: str
    channel: str
    physical_wavelength: float
    required_source_count: int
    selection_method: (
        InstrumentChannelCalibrationMultiSourceSelectionMethod | str
    )
    selection_source_ordering_field: str
    minimum_matched_pairs_per_source: int
    temporal_fold_count: int
    minimum_holdout_pairs_per_fold: int
    minimum_holdout_amplitude_to_median_reference_error: float
    physical_wavelength_absolute_tolerance: float
    cross_source_validation_method: (
        InstrumentChannelCalibrationCrossSourceValidationMethod | str
    )
    source_balance_method: InstrumentChannelCalibrationSourceBalanceMethod | str
    training_pair_subsample_seed_derivation: str
    maximum_median_source_holdout_normalized_rmse: float
    maximum_worst_source_holdout_normalized_rmse: float
    maximum_absolute_median_source_holdout_bias_normalized: float
    private_input_policy: tuple[str, ...]
    public_summary_policy: tuple[str, ...]
    selection_without_replacement: bool = True
    eligibility_evaluated_before_selection: bool = True
    calibration_outcomes_excluded_from_eligibility: bool = True
    require_exact_observational_channel_identity: bool = True
    require_finite_strictly_positive_flux_and_error: bool = True
    require_independent_primary_astrophysical_sources: bool = True
    exclude_derived_sources: bool = True
    require_every_selected_source_held_out_once: bool = True
    require_every_heldout_temporal_fold_informative: bool = True
    coefficient_stability_reported_not_gated: bool = True
    private_parquet_runner_distributed: bool = False
    public_package_parquet_support_required: bool = False
    protocol_frozen_before_execution: bool = True
    catalogue_population_performed: bool = False
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_MULTISOURCE_PROTOCOL_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_MULTISOURCE_PROTOCOL_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported multi-source protocol schema.")

        for name in (
            "protocol_id",
            "protocol_version",
            "candidate_protocol_id",
            "candidate_protocol_version",
            "rule_id",
            "reference_channel",
            "channel",
            "selection_source_ordering_field",
            "training_pair_subsample_seed_derivation",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_text(getattr(self, name), name=name),
            )

        if self.reference_channel == self.channel:
            raise ValueError("Reference and target channels must differ.")

        object.__setattr__(
            self,
            "candidate_protocol_sha256",
            _normalize_sha256(
                self.candidate_protocol_sha256,
                name="candidate_protocol_sha256",
            ),
        )
        object.__setattr__(
            self,
            "physical_wavelength",
            _normalize_float(
                self.physical_wavelength,
                name="physical_wavelength",
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        object.__setattr__(
            self,
            "required_source_count",
            _normalize_positive_int(
                self.required_source_count,
                name="required_source_count",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "selection_method",
            _normalize_enum(
                self.selection_method,
                InstrumentChannelCalibrationMultiSourceSelectionMethod,
                name="selection_method",
            ),
        )
        object.__setattr__(
            self,
            "minimum_matched_pairs_per_source",
            _normalize_positive_int(
                self.minimum_matched_pairs_per_source,
                name="minimum_matched_pairs_per_source",
            ),
        )
        object.__setattr__(
            self,
            "temporal_fold_count",
            _normalize_positive_int(
                self.temporal_fold_count,
                name="temporal_fold_count",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "minimum_holdout_pairs_per_fold",
            _normalize_positive_int(
                self.minimum_holdout_pairs_per_fold,
                name="minimum_holdout_pairs_per_fold",
            ),
        )
        if (
            self.minimum_matched_pairs_per_source
            < self.temporal_fold_count
            * self.minimum_holdout_pairs_per_fold
        ):
            raise ValueError(
                "minimum_matched_pairs_per_source must support every fold."
            )
        object.__setattr__(
            self,
            "minimum_holdout_amplitude_to_median_reference_error",
            _normalize_float(
                self.minimum_holdout_amplitude_to_median_reference_error,
                name=(
                    "minimum_holdout_amplitude_to_median_reference_error"
                ),
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        object.__setattr__(
            self,
            "physical_wavelength_absolute_tolerance",
            _normalize_float(
                self.physical_wavelength_absolute_tolerance,
                name="physical_wavelength_absolute_tolerance",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "cross_source_validation_method",
            _normalize_enum(
                self.cross_source_validation_method,
                InstrumentChannelCalibrationCrossSourceValidationMethod,
                name="cross_source_validation_method",
            ),
        )
        object.__setattr__(
            self,
            "source_balance_method",
            _normalize_enum(
                self.source_balance_method,
                InstrumentChannelCalibrationSourceBalanceMethod,
                name="source_balance_method",
            ),
        )

        for name in (
            "maximum_median_source_holdout_normalized_rmse",
            "maximum_worst_source_holdout_normalized_rmse",
            "maximum_absolute_median_source_holdout_bias_normalized",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                ),
            )

        if (
            self.maximum_worst_source_holdout_normalized_rmse
            < self.maximum_median_source_holdout_normalized_rmse
        ):
            raise ValueError(
                "Worst-source RMSE gate cannot be tighter than median gate."
            )

        object.__setattr__(
            self,
            "private_input_policy",
            _normalize_text_tuple(
                self.private_input_policy,
                name="private_input_policy",
            ),
        )
        object.__setattr__(
            self,
            "public_summary_policy",
            _normalize_text_tuple(
                self.public_summary_policy,
                name="public_summary_policy",
            ),
        )

        for name in (
            "selection_without_replacement",
            "eligibility_evaluated_before_selection",
            "calibration_outcomes_excluded_from_eligibility",
            "require_exact_observational_channel_identity",
            "require_finite_strictly_positive_flux_and_error",
            "require_independent_primary_astrophysical_sources",
            "exclude_derived_sources",
            "require_every_selected_source_held_out_once",
            "require_every_heldout_temporal_fold_informative",
            "coefficient_stability_reported_not_gated",
            "protocol_frozen_before_execution",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} must be true.")

        for name in (
            "private_parquet_runner_distributed",
            "public_package_parquet_support_required",
            "catalogue_population_performed",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"{name} must be false.")

    @property
    def canonical_sha256(self) -> str:
        """Return the digest of the strict canonical protocol payload."""

        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-safe protocol representation."""

        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "candidate_protocol_id": self.candidate_protocol_id,
            "candidate_protocol_version": self.candidate_protocol_version,
            "candidate_protocol_sha256": self.candidate_protocol_sha256,
            "rule_id": self.rule_id,
            "reference_channel": self.reference_channel,
            "channel": self.channel,
            "physical_wavelength": self.physical_wavelength,
            "required_source_count": self.required_source_count,
            "selection_method": self.selection_method.value,
            "selection_source_ordering_field": (
                self.selection_source_ordering_field
            ),
            "selection_without_replacement": (
                self.selection_without_replacement
            ),
            "eligibility_evaluated_before_selection": (
                self.eligibility_evaluated_before_selection
            ),
            "calibration_outcomes_excluded_from_eligibility": (
                self.calibration_outcomes_excluded_from_eligibility
            ),
            "minimum_matched_pairs_per_source": (
                self.minimum_matched_pairs_per_source
            ),
            "temporal_fold_count": self.temporal_fold_count,
            "minimum_holdout_pairs_per_fold": (
                self.minimum_holdout_pairs_per_fold
            ),
            "minimum_holdout_amplitude_to_median_reference_error": (
                self.minimum_holdout_amplitude_to_median_reference_error
            ),
            "physical_wavelength_absolute_tolerance": (
                self.physical_wavelength_absolute_tolerance
            ),
            "require_exact_observational_channel_identity": (
                self.require_exact_observational_channel_identity
            ),
            "require_finite_strictly_positive_flux_and_error": (
                self.require_finite_strictly_positive_flux_and_error
            ),
            "require_independent_primary_astrophysical_sources": (
                self.require_independent_primary_astrophysical_sources
            ),
            "exclude_derived_sources": self.exclude_derived_sources,
            "cross_source_validation_method": (
                self.cross_source_validation_method.value
            ),
            "source_balance_method": self.source_balance_method.value,
            "training_pair_subsample_seed_derivation": (
                self.training_pair_subsample_seed_derivation
            ),
            "require_every_selected_source_held_out_once": (
                self.require_every_selected_source_held_out_once
            ),
            "require_every_heldout_temporal_fold_informative": (
                self.require_every_heldout_temporal_fold_informative
            ),
            "maximum_median_source_holdout_normalized_rmse": (
                self.maximum_median_source_holdout_normalized_rmse
            ),
            "maximum_worst_source_holdout_normalized_rmse": (
                self.maximum_worst_source_holdout_normalized_rmse
            ),
            "maximum_absolute_median_source_holdout_bias_normalized": (
                self.maximum_absolute_median_source_holdout_bias_normalized
            ),
            "coefficient_stability_reported_not_gated": (
                self.coefficient_stability_reported_not_gated
            ),
            "private_input_policy": list(self.private_input_policy),
            "public_summary_policy": list(self.public_summary_policy),
            "private_parquet_runner_distributed": (
                self.private_parquet_runner_distributed
            ),
            "public_package_parquet_support_required": (
                self.public_package_parquet_support_required
            ),
            "protocol_frozen_before_execution": (
                self.protocol_frozen_before_execution
            ),
            "catalogue_population_performed": (
                self.catalogue_population_performed
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> InstrumentChannelCalibrationMultiSourceValidationProtocol:
        """Construct a strict protocol from its public representation."""

        if not isinstance(payload, dict):
            raise TypeError("Multi-source protocol must be a dictionary.")

        expected = set(
            cls(
                protocol_id="protocol",
                protocol_version="1",
                candidate_protocol_id="candidate",
                candidate_protocol_version="1",
                candidate_protocol_sha256="0" * 64,
                rule_id="rule",
                reference_channel="reference",
                channel="target",
                physical_wavelength=1.0,
                required_source_count=5,
                selection_method=(
                    InstrumentChannelCalibrationMultiSourceSelectionMethod.
                    SEEDED_PERMUTATION_FIRST_N
                ),
                selection_source_ordering_field="astrophysical_source_id",
                minimum_matched_pairs_per_source=100,
                temporal_fold_count=5,
                minimum_holdout_pairs_per_fold=20,
                minimum_holdout_amplitude_to_median_reference_error=5.0,
                physical_wavelength_absolute_tolerance=1.0e-15,
                cross_source_validation_method=(
                    InstrumentChannelCalibrationCrossSourceValidationMethod.
                    LEAVE_ONE_SOURCE_OUT
                ),
                source_balance_method=(
                    InstrumentChannelCalibrationSourceBalanceMethod.
                    EQUAL_PAIR_COUNT
                ),
                training_pair_subsample_seed_derivation="sha256",
                maximum_median_source_holdout_normalized_rmse=0.1,
                maximum_worst_source_holdout_normalized_rmse=0.2,
                maximum_absolute_median_source_holdout_bias_normalized=0.05,
                private_input_policy=("private",),
                public_summary_policy=("redacted",),
            ).to_dict()
        )
        if set(payload) != expected:
            raise ValueError(
                "Multi-source protocol fields do not match the contract."
            )

        result = cls(
            **{
                **payload,
                "private_input_policy": tuple(
                    payload["private_input_policy"]
                ),
                "public_summary_policy": tuple(
                    payload["public_summary_policy"]
                ),
            }
        )
        if result.to_dict() != payload:
            raise ValueError("Multi-source protocol failed strict round trip.")
        return result


@dataclass(frozen=True)
class InstrumentChannelCalibrationMultiSourceValidationSummary:
    """Redacted public evidence emitted after private execution."""

    summary_id: str
    summary_version: str
    protocol_id: str
    protocol_version: str
    protocol_sha256: str
    candidate_protocol_id: str
    candidate_protocol_version: str
    candidate_protocol_sha256: str
    rule_id: str
    private_input_sha256: str
    selection_seed: int
    eligible_source_count: int
    selected_source_id_sha256s: tuple[str, ...]
    successful_source_holdout_count: int
    median_source_holdout_normalized_rmse: float | None
    worst_source_holdout_normalized_rmse: float | None
    median_source_holdout_bias_normalized: float | None
    fitted_scale_median: float | None
    fitted_scale_mad: float | None
    fitted_offset_median: float | None
    fitted_offset_mad: float | None
    package_version: str
    package_commit: str
    executed_at_utc: str
    disposition: (
        InstrumentChannelCalibrationMultiSourceValidationDisposition | str
    )
    reasons: tuple[str, ...]
    all_acceptance_criteria_passed: bool
    raw_source_identifiers_included: bool = False
    private_input_distributed: bool = False
    private_detailed_report_distributed: bool = False
    catalogue_population_performed: bool = False
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_MULTISOURCE_SUMMARY_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_MULTISOURCE_SUMMARY_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported multi-source summary schema.")

        for name in (
            "summary_id",
            "summary_version",
            "protocol_id",
            "protocol_version",
            "candidate_protocol_id",
            "candidate_protocol_version",
            "rule_id",
            "package_version",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_text(getattr(self, name), name=name),
            )

        for name in (
            "protocol_sha256",
            "candidate_protocol_sha256",
            "private_input_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_sha256(getattr(self, name), name=name),
            )

        object.__setattr__(
            self,
            "selection_seed",
            _normalize_nonnegative_int(
                self.selection_seed,
                name="selection_seed",
            ),
        )
        for name in (
            "eligible_source_count",
            "successful_source_holdout_count",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_nonnegative_int(
                    getattr(self, name),
                    name=name,
                ),
            )

        selected = _normalize_sha256_tuple(
            self.selected_source_id_sha256s,
            name="selected_source_id_sha256s",
        )
        if len(selected) > self.eligible_source_count:
            raise ValueError("Selected-source count exceeds eligible count.")
        if self.successful_source_holdout_count > len(selected):
            raise ValueError(
                "Successful holdout count exceeds selected-source count."
            )
        object.__setattr__(
            self,
            "selected_source_id_sha256s",
            selected,
        )

        for name in (
            "median_source_holdout_normalized_rmse",
            "worst_source_holdout_normalized_rmse",
            "median_source_holdout_bias_normalized",
            "fitted_scale_median",
            "fitted_scale_mad",
            "fitted_offset_median",
            "fitted_offset_mad",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_optional_float(
                    getattr(self, name),
                    name=name,
                ),
            )

        if not _COMMIT_PATTERN.fullmatch(self.package_commit):
            raise ValueError(
                "package_commit must be a lowercase 40-character Git commit."
            )
        if not _UTC_PATTERN.fullmatch(self.executed_at_utc):
            raise ValueError(
                "executed_at_utc must use YYYY-MM-DDTHH:MM:SSZ."
            )

        object.__setattr__(
            self,
            "disposition",
            _normalize_enum(
                self.disposition,
                InstrumentChannelCalibrationMultiSourceValidationDisposition,
                name="disposition",
            ),
        )
        object.__setattr__(
            self,
            "reasons",
            _normalize_text_tuple(
                self.reasons,
                name="reasons",
                allow_empty=True,
            ),
        )

        passed = (
            self.disposition
            is InstrumentChannelCalibrationMultiSourceValidationDisposition.PASSED
        )
        if self.all_acceptance_criteria_passed is not passed:
            raise ValueError(
                "all_acceptance_criteria_passed must match disposition."
            )
        if passed and self.reasons:
            raise ValueError("Passed summaries must not contain reasons.")

        for name in (
            "raw_source_identifiers_included",
            "private_input_distributed",
            "private_detailed_report_distributed",
            "catalogue_population_performed",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"{name} must be false.")

    def to_dict(self) -> dict[str, Any]:
        """Return the strict redacted public representation."""

        return {
            "schema_version": self.schema_version,
            "summary_id": self.summary_id,
            "summary_version": self.summary_version,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "protocol_sha256": self.protocol_sha256,
            "candidate_protocol_id": self.candidate_protocol_id,
            "candidate_protocol_version": self.candidate_protocol_version,
            "candidate_protocol_sha256": self.candidate_protocol_sha256,
            "rule_id": self.rule_id,
            "private_input_sha256": self.private_input_sha256,
            "selection_seed": self.selection_seed,
            "eligible_source_count": self.eligible_source_count,
            "selected_source_id_sha256s": list(
                self.selected_source_id_sha256s
            ),
            "successful_source_holdout_count": (
                self.successful_source_holdout_count
            ),
            "median_source_holdout_normalized_rmse": (
                self.median_source_holdout_normalized_rmse
            ),
            "worst_source_holdout_normalized_rmse": (
                self.worst_source_holdout_normalized_rmse
            ),
            "median_source_holdout_bias_normalized": (
                self.median_source_holdout_bias_normalized
            ),
            "fitted_scale_median": self.fitted_scale_median,
            "fitted_scale_mad": self.fitted_scale_mad,
            "fitted_offset_median": self.fitted_offset_median,
            "fitted_offset_mad": self.fitted_offset_mad,
            "package_version": self.package_version,
            "package_commit": self.package_commit,
            "executed_at_utc": self.executed_at_utc,
            "disposition": self.disposition.value,
            "reasons": list(self.reasons),
            "all_acceptance_criteria_passed": (
                self.all_acceptance_criteria_passed
            ),
            "raw_source_identifiers_included": (
                self.raw_source_identifiers_included
            ),
            "private_input_distributed": self.private_input_distributed,
            "private_detailed_report_distributed": (
                self.private_detailed_report_distributed
            ),
            "catalogue_population_performed": (
                self.catalogue_population_performed
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> InstrumentChannelCalibrationMultiSourceValidationSummary:
        """Construct a strict redacted summary from JSON-safe data."""

        if not isinstance(payload, dict):
            raise TypeError("Multi-source summary must be a dictionary.")

        expected = set(
            cls(
                summary_id="summary",
                summary_version="1",
                protocol_id="protocol",
                protocol_version="1",
                protocol_sha256="0" * 64,
                candidate_protocol_id="candidate",
                candidate_protocol_version="1",
                candidate_protocol_sha256="1" * 64,
                rule_id="rule",
                private_input_sha256="2" * 64,
                selection_seed=0,
                eligible_source_count=0,
                selected_source_id_sha256s=(),
                successful_source_holdout_count=0,
                median_source_holdout_normalized_rmse=None,
                worst_source_holdout_normalized_rmse=None,
                median_source_holdout_bias_normalized=None,
                fitted_scale_median=None,
                fitted_scale_mad=None,
                fitted_offset_median=None,
                fitted_offset_mad=None,
                package_version="test",
                package_commit="3" * 40,
                executed_at_utc="2026-07-26T00:00:00Z",
                disposition=(
                    InstrumentChannelCalibrationMultiSourceValidationDisposition.
                    INCONCLUSIVE
                ),
                reasons=("insufficient_evidence",),
                all_acceptance_criteria_passed=False,
            ).to_dict()
        )
        if set(payload) != expected:
            raise ValueError(
                "Multi-source summary fields do not match the contract."
            )

        result = cls(
            **{
                **payload,
                "selected_source_id_sha256s": tuple(
                    payload["selected_source_id_sha256s"]
                ),
                "reasons": tuple(payload["reasons"]),
            }
        )
        if result.to_dict() != payload:
            raise ValueError("Multi-source summary failed strict round trip.")
        return result
