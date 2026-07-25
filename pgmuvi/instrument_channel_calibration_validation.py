"""Prospective validation contracts for instrument-channel calibration rules.

The records in this module define a pre-registered, fail-closed validation
boundary. They do not execute calibration validation, create pairing rules, or
populate a pairing-rule catalogue.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER,
    InstrumentChannelPairingMethod,
)

INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_DATASET_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-validation-dataset-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_DATASET_MANIFEST_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-validation-dataset-manifest-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_ACCEPTANCE_CRITERIA_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-validation-acceptance-criteria-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_PROTOCOL_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-validation-protocol-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_FOLD_RESULT_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-validation-fold-result-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_SOURCE_RESULT_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-validation-source-result-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-validation-result-v1"
)
INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_REPORT_SCHEMA_VERSION = (
    "pgmuvi-instrument-channel-calibration-validation-report-v1"
)

__all__ = [
    "INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_ACCEPTANCE_CRITERIA_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_DATASET_MANIFEST_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_DATASET_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_FOLD_RESULT_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_PROTOCOL_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_REPORT_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_SCHEMA_VERSION",
    "INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_SOURCE_RESULT_SCHEMA_VERSION",
    "InstrumentChannelCalibrationValidationAcceptanceCriteria",
    "InstrumentChannelCalibrationValidationDataset",
    "InstrumentChannelCalibrationValidationDatasetManifest",
    "InstrumentChannelCalibrationValidationDisposition",
    "InstrumentChannelCalibrationValidationFoldResult",
    "InstrumentChannelCalibrationValidationProtocol",
    "InstrumentChannelCalibrationValidationReport",
    "InstrumentChannelCalibrationValidationResult",
    "InstrumentChannelCalibrationValidationSourceResult",
    "assess_instrument_channel_calibration_validation_result",
]


class _StringEnum(str, Enum):
    """Enum whose members serialize as stable public strings."""

    def __str__(self) -> str:
        return self.value


class InstrumentChannelCalibrationValidationDisposition(_StringEnum):
    """Outcome of prospective calibration-rule validation."""

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


def _normalize_optional_text(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _normalize_text(value, name=name)


def _normalize_text_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings.")
    try:
        normalized = tuple(
            _normalize_text(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of strings.") from exc
    if not normalized:
        raise ValueError(f"{name} must contain at least one item.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    return normalized


def _normalize_sha256(value: Any, *, name: str) -> str:
    normalized = _normalize_text(value, name=name)
    if (
        len(normalized) != 64
        or normalized != normalized.lower()
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(
            f"{name} must be a lowercase 64-character hexadecimal SHA-256 digest."
        )
    return normalized


def _normalize_positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
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


def _normalize_finite_float(
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
        invalid = normalized <= minimum if strict_minimum else normalized < minimum
        if invalid:
            comparator = "greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} must be {comparator} {minimum}.")
    return normalized


def _normalize_optional_finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float | None:
    if value is None:
        return None
    return _normalize_finite_float(
        value,
        name=name,
        minimum=minimum,
        strict_minimum=strict_minimum,
    )


@dataclass(frozen=True)
class InstrumentChannelCalibrationValidationDataset:
    """Immutable identity and lineage for one validation dataset."""

    dataset_id: str
    astrophysical_source_id: str
    dataset_reference: str
    dataset_sha256: str
    derivation_parent_dataset_id: str | None = None
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_DATASET_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_DATASET_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported calibration-validation dataset schema version: "
                f"{self.schema_version!r}."
            )
        for name in (
            "dataset_id",
            "astrophysical_source_id",
            "dataset_reference",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_text(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "dataset_sha256",
            _normalize_sha256(self.dataset_sha256, name="dataset_sha256"),
        )
        parent = _normalize_optional_text(
            self.derivation_parent_dataset_id,
            name="derivation_parent_dataset_id",
        )
        if parent == self.dataset_id:
            raise ValueError("A validation dataset cannot derive from itself.")
        object.__setattr__(self, "derivation_parent_dataset_id", parent)

    @property
    def is_derived(self) -> bool:
        """Whether this dataset is derived from another dataset."""

        return self.derivation_parent_dataset_id is not None

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-safe dataset representation."""

        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "astrophysical_source_id": self.astrophysical_source_id,
            "dataset_reference": self.dataset_reference,
            "dataset_sha256": self.dataset_sha256,
            "derivation_parent_dataset_id": self.derivation_parent_dataset_id,
            "is_derived": self.is_derived,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> InstrumentChannelCalibrationValidationDataset:
        """Construct a dataset identity from a strict representation."""

        if not isinstance(payload, dict):
            raise TypeError("Calibration-validation dataset must be a dictionary.")
        expected_keys = {
            "schema_version",
            "dataset_id",
            "astrophysical_source_id",
            "dataset_reference",
            "dataset_sha256",
            "derivation_parent_dataset_id",
            "is_derived",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "Calibration-validation dataset payload must contain exactly "
                "the contract fields."
            )
        dataset = cls(
            schema_version=payload["schema_version"],
            dataset_id=payload["dataset_id"],
            astrophysical_source_id=payload["astrophysical_source_id"],
            dataset_reference=payload["dataset_reference"],
            dataset_sha256=payload["dataset_sha256"],
            derivation_parent_dataset_id=payload[
                "derivation_parent_dataset_id"
            ],
        )
        if dataset.to_dict() != payload:
            raise ValueError(
                "Calibration-validation dataset payload does not match the "
                "strict contract representation."
            )
        return dataset



@dataclass(frozen=True)
class InstrumentChannelCalibrationValidationDatasetManifest:
    """Strict additional-source datasets bound to one frozen protocol."""

    protocol_id: str
    protocol_version: str
    protocol_sha256: str
    datasets: tuple[InstrumentChannelCalibrationValidationDataset, ...]
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_DATASET_MANIFEST_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_DATASET_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported calibration-validation dataset-manifest schema "
                f"version: {self.schema_version!r}."
            )
        object.__setattr__(
            self,
            "protocol_id",
            _normalize_text(self.protocol_id, name="protocol_id"),
        )
        object.__setattr__(
            self,
            "protocol_version",
            _normalize_text(
                self.protocol_version,
                name="protocol_version",
            ),
        )
        object.__setattr__(
            self,
            "protocol_sha256",
            _normalize_sha256(
                self.protocol_sha256,
                name="protocol_sha256",
            ),
        )

        datasets = self.datasets
        if isinstance(datasets, (str, bytes)):
            raise TypeError("datasets must be a sequence of validation datasets.")
        normalized = tuple(datasets)
        if not normalized:
            raise ValueError(
                "datasets must contain at least one additional dataset."
            )
        if any(
            not isinstance(
                dataset,
                InstrumentChannelCalibrationValidationDataset,
            )
            for dataset in normalized
        ):
            raise TypeError(
                "datasets must contain only "
                "InstrumentChannelCalibrationValidationDataset records."
            )
        if any(dataset.is_derived for dataset in normalized):
            raise ValueError(
                "Additional validation datasets must be primary, not derived."
            )

        dataset_ids = tuple(dataset.dataset_id for dataset in normalized)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError(
                "Additional validation datasets must use unique dataset_id values."
            )

        source_ids = tuple(
            dataset.astrophysical_source_id for dataset in normalized
        )
        if len(set(source_ids)) != len(source_ids):
            raise ValueError(
                "Additional validation datasets must represent distinct "
                "astrophysical sources."
            )

        object.__setattr__(self, "datasets", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-safe manifest representation."""

        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "protocol_sha256": self.protocol_sha256,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> InstrumentChannelCalibrationValidationDatasetManifest:
        """Construct an additional-source manifest from strict JSON data."""

        if not isinstance(payload, dict):
            raise TypeError(
                "Calibration-validation dataset manifest must be a dictionary."
            )
        expected_keys = {
            "schema_version",
            "protocol_id",
            "protocol_version",
            "protocol_sha256",
            "datasets",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "Calibration-validation dataset-manifest payload must contain "
                "exactly the contract fields."
            )

        manifest = cls(
            schema_version=payload["schema_version"],
            protocol_id=payload["protocol_id"],
            protocol_version=payload["protocol_version"],
            protocol_sha256=payload["protocol_sha256"],
            datasets=tuple(
                InstrumentChannelCalibrationValidationDataset.from_dict(item)
                for item in payload["datasets"]
            ),
        )
        if manifest.to_dict() != payload:
            raise ValueError(
                "Calibration-validation dataset-manifest payload does not "
                "match the strict contract representation."
            )
        return manifest


@dataclass(frozen=True)
class InstrumentChannelCalibrationValidationAcceptanceCriteria:
    """Pre-registered quantitative and evidence-adequacy gates."""

    minimum_independent_astrophysical_sources: int
    minimum_matched_pairs_per_source: int
    temporal_fold_count: int
    minimum_holdout_pairs_per_fold: int
    minimum_holdout_normalization_amplitude_to_median_reference_error: float
    maximum_median_holdout_normalized_rmse: float
    maximum_worst_fold_holdout_normalized_rmse: float
    maximum_absolute_holdout_median_bias_normalized: float
    require_all_temporal_folds_successful: bool = True
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_ACCEPTANCE_CRITERIA_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_ACCEPTANCE_CRITERIA_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported calibration-validation acceptance-criteria schema "
                f"version: {self.schema_version!r}."
            )
        object.__setattr__(
            self,
            "minimum_independent_astrophysical_sources",
            _normalize_positive_int(
                self.minimum_independent_astrophysical_sources,
                name="minimum_independent_astrophysical_sources",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "minimum_matched_pairs_per_source",
            _normalize_positive_int(
                self.minimum_matched_pairs_per_source,
                name="minimum_matched_pairs_per_source",
                minimum=3,
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
                minimum=3,
            ),
        )
        object.__setattr__(
            self,
            "minimum_holdout_normalization_amplitude_to_median_reference_error",
            _normalize_finite_float(
                self.minimum_holdout_normalization_amplitude_to_median_reference_error,
                name=(
                    "minimum_holdout_normalization_amplitude_to_median_"
                    "reference_error"
                ),
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        median_limit = _normalize_finite_float(
            self.maximum_median_holdout_normalized_rmse,
            name="maximum_median_holdout_normalized_rmse",
            minimum=0.0,
            strict_minimum=True,
        )
        worst_limit = _normalize_finite_float(
            self.maximum_worst_fold_holdout_normalized_rmse,
            name="maximum_worst_fold_holdout_normalized_rmse",
            minimum=0.0,
            strict_minimum=True,
        )
        if worst_limit < median_limit:
            raise ValueError(
                "maximum_worst_fold_holdout_normalized_rmse must be at "
                "least the median holdout limit."
            )
        object.__setattr__(
            self,
            "maximum_median_holdout_normalized_rmse",
            median_limit,
        )
        object.__setattr__(
            self,
            "maximum_worst_fold_holdout_normalized_rmse",
            worst_limit,
        )
        object.__setattr__(
            self,
            "maximum_absolute_holdout_median_bias_normalized",
            _normalize_finite_float(
                self.maximum_absolute_holdout_median_bias_normalized,
                name="maximum_absolute_holdout_median_bias_normalized",
                minimum=0.0,
            ),
        )
        if self.require_all_temporal_folds_successful is not True:
            raise ValueError(
                "Prospective scientific validation requires every temporal "
                "fold to complete successfully."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-safe acceptance-criteria representation."""

        return {
            "schema_version": self.schema_version,
            "minimum_independent_astrophysical_sources": (
                self.minimum_independent_astrophysical_sources
            ),
            "minimum_matched_pairs_per_source": (
                self.minimum_matched_pairs_per_source
            ),
            "temporal_fold_count": self.temporal_fold_count,
            "minimum_holdout_pairs_per_fold": (
                self.minimum_holdout_pairs_per_fold
            ),
            (
                "minimum_holdout_normalization_amplitude_to_median_"
                "reference_error"
            ): (
                self.minimum_holdout_normalization_amplitude_to_median_reference_error
            ),
            "maximum_median_holdout_normalized_rmse": (
                self.maximum_median_holdout_normalized_rmse
            ),
            "maximum_worst_fold_holdout_normalized_rmse": (
                self.maximum_worst_fold_holdout_normalized_rmse
            ),
            "maximum_absolute_holdout_median_bias_normalized": (
                self.maximum_absolute_holdout_median_bias_normalized
            ),
            "require_all_temporal_folds_successful": (
                self.require_all_temporal_folds_successful
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> InstrumentChannelCalibrationValidationAcceptanceCriteria:
        """Construct acceptance criteria from a strict representation."""

        if not isinstance(payload, dict):
            raise TypeError(
                "Calibration-validation acceptance criteria must be a dictionary."
            )
        expected_keys = {
            "schema_version",
            "minimum_independent_astrophysical_sources",
            "minimum_matched_pairs_per_source",
            "temporal_fold_count",
            "minimum_holdout_pairs_per_fold",
            (
                "minimum_holdout_normalization_amplitude_to_median_"
                "reference_error"
            ),
            "maximum_median_holdout_normalized_rmse",
            "maximum_worst_fold_holdout_normalized_rmse",
            "maximum_absolute_holdout_median_bias_normalized",
            "require_all_temporal_folds_successful",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "Calibration-validation acceptance-criteria payload must "
                "contain exactly the contract fields."
            )
        criteria = cls(**payload)
        if criteria.to_dict() != payload:
            raise ValueError(
                "Calibration-validation acceptance-criteria payload does not "
                "match the strict contract representation."
            )
        return criteria


@dataclass(frozen=True)
class InstrumentChannelCalibrationValidationProtocol:
    """Immutable prospective protocol for one exact candidate pairing rule."""

    protocol_id: str
    protocol_version: str
    rule_id: str
    reference_instrument: str
    reference_channel: str
    channel_instrument: str
    channel: str
    physical_wavelength: float
    pairing_method: InstrumentChannelPairingMethod | str
    time_unit: str
    maximum_time_separation: float | None
    anchor_dataset: InstrumentChannelCalibrationValidationDataset
    reference_channel_justification: str
    pairing_method_justification: str
    time_tolerance_justification: str
    acceptance_criteria_justification: str
    calibration_family: str
    sigma_clip: float
    maximum_fit_iterations: int
    minimum_fit_pairs: int
    measurement_error_policy: str
    temporal_holdout_method: str
    temporal_holdout_ordering_field: str
    holdout_metric_normalization: str
    source_independence_unit: str
    derived_datasets_count_as_independent: bool
    acceptance_criteria: InstrumentChannelCalibrationValidationAcceptanceCriteria
    applicability_boundaries: tuple[str, ...]
    protocol_frozen_before_execution: bool = True
    protocol_execution_performed: bool = False
    populated_catalogue_created: bool = False
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_PROTOCOL_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_PROTOCOL_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported calibration-validation protocol schema version: "
                f"{self.schema_version!r}."
            )
        for name in (
            "protocol_id",
            "protocol_version",
            "rule_id",
            "reference_instrument",
            "reference_channel",
            "channel_instrument",
            "channel",
            "time_unit",
            "reference_channel_justification",
            "pairing_method_justification",
            "time_tolerance_justification",
            "acceptance_criteria_justification",
            "calibration_family",
            "measurement_error_policy",
            "temporal_holdout_method",
            "temporal_holdout_ordering_field",
            "holdout_metric_normalization",
            "source_independence_unit",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_text(getattr(self, name), name=name),
            )
        if self.reference_channel == self.channel:
            raise ValueError(
                "reference_channel and channel must identify different "
                "observational channels."
            )
        object.__setattr__(
            self,
            "physical_wavelength",
            _normalize_finite_float(
                self.physical_wavelength,
                name="physical_wavelength",
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        method = self.pairing_method
        if not isinstance(method, InstrumentChannelPairingMethod):
            try:
                method = InstrumentChannelPairingMethod(str(method))
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported validation-protocol pairing method: {method!r}."
                ) from exc
        maximum = _normalize_optional_finite_float(
            self.maximum_time_separation,
            name="maximum_time_separation",
            minimum=0.0,
        )
        if method is InstrumentChannelPairingMethod.EXACT_TIMESTAMP:
            if maximum not in (None, 0.0):
                raise ValueError(
                    "Exact-timestamp protocols permit no non-zero maximum "
                    "time separation."
                )
        elif maximum is None or maximum <= 0.0:
            raise ValueError(
                "Nearest-within-tolerance protocols require a finite positive "
                "maximum time separation."
            )
        object.__setattr__(self, "pairing_method", method)
        object.__setattr__(self, "maximum_time_separation", maximum)
        object.__setattr__(
            self,
            "sigma_clip",
            _normalize_finite_float(
                self.sigma_clip,
                name="sigma_clip",
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        object.__setattr__(
            self,
            "maximum_fit_iterations",
            _normalize_positive_int(
                self.maximum_fit_iterations,
                name="maximum_fit_iterations",
            ),
        )
        object.__setattr__(
            self,
            "minimum_fit_pairs",
            _normalize_positive_int(
                self.minimum_fit_pairs,
                name="minimum_fit_pairs",
                minimum=3,
            ),
        )
        if self.derived_datasets_count_as_independent is not False:
            raise ValueError(
                "Derived datasets cannot count as independent astrophysical "
                "sources."
            )
        if not isinstance(
            self.anchor_dataset,
            InstrumentChannelCalibrationValidationDataset,
        ):
            raise TypeError(
                "anchor_dataset must be an "
                "InstrumentChannelCalibrationValidationDataset."
            )
        if self.anchor_dataset.is_derived:
            raise ValueError("The anchor validation dataset cannot be derived.")
        if not isinstance(
            self.acceptance_criteria,
            InstrumentChannelCalibrationValidationAcceptanceCriteria,
        ):
            raise TypeError(
                "acceptance_criteria must be an "
                "InstrumentChannelCalibrationValidationAcceptanceCriteria."
            )
        object.__setattr__(
            self,
            "applicability_boundaries",
            _normalize_text_tuple(
                self.applicability_boundaries,
                name="applicability_boundaries",
            ),
        )
        if self.protocol_frozen_before_execution is not True:
            raise ValueError("Validation protocol must be frozen before execution.")
        if self.protocol_execution_performed is not False:
            raise ValueError(
                "A validation protocol record cannot claim execution was performed."
            )
        if self.populated_catalogue_created is not False:
            raise ValueError(
                "A validation protocol record cannot create a populated catalogue."
            )

    @property
    def identity_key(self) -> tuple[str, str, str, str, str, float]:
        """Return the exact candidate rule and channel identity."""

        return (
            self.rule_id,
            self.reference_instrument,
            self.reference_channel,
            self.channel_instrument,
            self.channel,
            self.physical_wavelength,
        )

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
            "rule_id": self.rule_id,
            "reference_instrument": self.reference_instrument,
            "reference_channel": self.reference_channel,
            "channel_instrument": self.channel_instrument,
            "channel": self.channel,
            "physical_wavelength": self.physical_wavelength,
            "pairing_method": self.pairing_method.value,
            "time_unit": self.time_unit,
            "maximum_time_separation": self.maximum_time_separation,
            "anchor_dataset": self.anchor_dataset.to_dict(),
            "reference_channel_justification": (
                self.reference_channel_justification
            ),
            "pairing_method_justification": self.pairing_method_justification,
            "time_tolerance_justification": self.time_tolerance_justification,
            "acceptance_criteria_justification": (
                self.acceptance_criteria_justification
            ),
            "calibration_family": self.calibration_family,
            "sigma_clip": self.sigma_clip,
            "maximum_fit_iterations": self.maximum_fit_iterations,
            "minimum_fit_pairs": self.minimum_fit_pairs,
            "measurement_error_policy": self.measurement_error_policy,
            "temporal_holdout_method": self.temporal_holdout_method,
            "temporal_holdout_ordering_field": (
                self.temporal_holdout_ordering_field
            ),
            "holdout_metric_normalization": (
                self.holdout_metric_normalization
            ),
            "source_independence_unit": self.source_independence_unit,
            "derived_datasets_count_as_independent": (
                self.derived_datasets_count_as_independent
            ),
            "acceptance_criteria": self.acceptance_criteria.to_dict(),
            "applicability_boundaries": list(self.applicability_boundaries),
            "protocol_frozen_before_execution": (
                self.protocol_frozen_before_execution
            ),
            "protocol_execution_performed": self.protocol_execution_performed,
            "populated_catalogue_created": self.populated_catalogue_created,
            "prior_exploratory_runs_eligible_as_validation_evidence": False,
            "marker": INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> InstrumentChannelCalibrationValidationProtocol:
        """Construct a prospective protocol from a strict representation."""

        if not isinstance(payload, dict):
            raise TypeError("Calibration-validation protocol must be a dictionary.")
        expected_keys = {
            "schema_version",
            "protocol_id",
            "protocol_version",
            "rule_id",
            "reference_instrument",
            "reference_channel",
            "channel_instrument",
            "channel",
            "physical_wavelength",
            "pairing_method",
            "time_unit",
            "maximum_time_separation",
            "anchor_dataset",
            "reference_channel_justification",
            "pairing_method_justification",
            "time_tolerance_justification",
            "acceptance_criteria_justification",
            "calibration_family",
            "sigma_clip",
            "maximum_fit_iterations",
            "minimum_fit_pairs",
            "measurement_error_policy",
            "temporal_holdout_method",
            "temporal_holdout_ordering_field",
            "holdout_metric_normalization",
            "source_independence_unit",
            "derived_datasets_count_as_independent",
            "acceptance_criteria",
            "applicability_boundaries",
            "protocol_frozen_before_execution",
            "protocol_execution_performed",
            "populated_catalogue_created",
            "prior_exploratory_runs_eligible_as_validation_evidence",
            "marker",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "Calibration-validation protocol payload must contain exactly "
                "the contract fields."
            )
        protocol = cls(
            schema_version=payload["schema_version"],
            protocol_id=payload["protocol_id"],
            protocol_version=payload["protocol_version"],
            rule_id=payload["rule_id"],
            reference_instrument=payload["reference_instrument"],
            reference_channel=payload["reference_channel"],
            channel_instrument=payload["channel_instrument"],
            channel=payload["channel"],
            physical_wavelength=payload["physical_wavelength"],
            pairing_method=payload["pairing_method"],
            time_unit=payload["time_unit"],
            maximum_time_separation=payload["maximum_time_separation"],
            anchor_dataset=(
                InstrumentChannelCalibrationValidationDataset.from_dict(
                    payload["anchor_dataset"]
                )
            ),
            reference_channel_justification=payload[
                "reference_channel_justification"
            ],
            pairing_method_justification=payload[
                "pairing_method_justification"
            ],
            time_tolerance_justification=payload[
                "time_tolerance_justification"
            ],
            acceptance_criteria_justification=payload[
                "acceptance_criteria_justification"
            ],
            calibration_family=payload["calibration_family"],
            sigma_clip=payload["sigma_clip"],
            maximum_fit_iterations=payload["maximum_fit_iterations"],
            minimum_fit_pairs=payload["minimum_fit_pairs"],
            measurement_error_policy=payload["measurement_error_policy"],
            temporal_holdout_method=payload["temporal_holdout_method"],
            temporal_holdout_ordering_field=payload[
                "temporal_holdout_ordering_field"
            ],
            holdout_metric_normalization=payload[
                "holdout_metric_normalization"
            ],
            source_independence_unit=payload["source_independence_unit"],
            derived_datasets_count_as_independent=payload[
                "derived_datasets_count_as_independent"
            ],
            acceptance_criteria=(
                InstrumentChannelCalibrationValidationAcceptanceCriteria.from_dict(
                    payload["acceptance_criteria"]
                )
            ),
            applicability_boundaries=tuple(payload["applicability_boundaries"]),
            protocol_frozen_before_execution=payload[
                "protocol_frozen_before_execution"
            ],
            protocol_execution_performed=payload[
                "protocol_execution_performed"
            ],
            populated_catalogue_created=payload["populated_catalogue_created"],
        )
        if (
            payload[
                "prior_exploratory_runs_eligible_as_validation_evidence"
            ]
            is not False
        ):
            raise ValueError(
                "Prior exploratory runs cannot be validation evidence for a "
                "prospectively frozen protocol."
            )
        if payload["marker"] != INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER:
            raise ValueError("Calibration-validation protocol marker is invalid.")
        if protocol.to_dict() != payload:
            raise ValueError(
                "Calibration-validation protocol payload does not match the "
                "strict contract representation."
            )
        return protocol


@dataclass(frozen=True)
class InstrumentChannelCalibrationValidationFoldResult:
    """Auditable held-out metrics for one pre-registered temporal fold."""

    fold_index: int
    n_training_pairs: int
    n_holdout_pairs: int
    successful: bool
    holdout_reference_flux_q05_q95_amplitude: float | None
    holdout_median_reference_flux_error: float | None
    holdout_normalized_rmse: float | None
    holdout_median_bias_normalized: float | None
    maximum_absolute_time_separation: float | None
    fitted_offset: float | None
    fitted_scale: float | None
    failure_reasons: tuple[str, ...] = ()
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_FOLD_RESULT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_FOLD_RESULT_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported calibration-validation fold-result schema "
                f"version: {self.schema_version!r}."
            )
        object.__setattr__(
            self,
            "fold_index",
            _normalize_nonnegative_int(
                self.fold_index,
                name="fold_index",
            ),
        )
        for name in ("n_training_pairs", "n_holdout_pairs"):
            object.__setattr__(
                self,
                name,
                _normalize_nonnegative_int(
                    getattr(self, name),
                    name=name,
                ),
            )
        if not isinstance(self.successful, bool):
            raise TypeError("successful must be boolean.")
        for name in (
            "holdout_reference_flux_q05_q95_amplitude",
            "holdout_median_reference_flux_error",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_optional_finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    strict_minimum=True,
                ),
            )
        object.__setattr__(
            self,
            "holdout_normalized_rmse",
            _normalize_optional_finite_float(
                self.holdout_normalized_rmse,
                name="holdout_normalized_rmse",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "holdout_median_bias_normalized",
            _normalize_optional_finite_float(
                self.holdout_median_bias_normalized,
                name="holdout_median_bias_normalized",
            ),
        )
        object.__setattr__(
            self,
            "maximum_absolute_time_separation",
            _normalize_optional_finite_float(
                self.maximum_absolute_time_separation,
                name="maximum_absolute_time_separation",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "fitted_offset",
            _normalize_optional_finite_float(
                self.fitted_offset,
                name="fitted_offset",
            ),
        )
        object.__setattr__(
            self,
            "fitted_scale",
            _normalize_optional_finite_float(
                self.fitted_scale,
                name="fitted_scale",
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        reasons = self.failure_reasons
        if isinstance(reasons, (str, bytes)):
            raise TypeError("failure_reasons must be a sequence of strings.")
        normalized_reasons = tuple(
            _normalize_text(reason, name=f"failure_reasons[{index}]")
            for index, reason in enumerate(reasons)
        )
        if len(set(normalized_reasons)) != len(normalized_reasons):
            raise ValueError("failure_reasons must not contain duplicates.")
        if self.successful and normalized_reasons:
            raise ValueError(
                "A successful temporal fold cannot contain failure reasons."
            )
        if not self.successful and not normalized_reasons:
            raise ValueError(
                "An unsuccessful temporal fold requires at least one failure reason."
            )
        object.__setattr__(self, "failure_reasons", normalized_reasons)
        if self.successful and not self.complete:
            raise ValueError(
                "A successful temporal fold requires every auditable metric."
            )

    @property
    def normalization_amplitude_to_median_reference_error(self) -> float | None:
        """Return the fold dynamic-range-to-error adequacy ratio."""

        amplitude = self.holdout_reference_flux_q05_q95_amplitude
        median_error = self.holdout_median_reference_flux_error
        if amplitude is None or median_error is None:
            return None
        return amplitude / median_error

    @property
    def complete(self) -> bool:
        """Whether all metrics required for one fold are available."""

        return (
            self.successful
            and not self.failure_reasons
            and self.n_training_pairs > 0
            and self.n_holdout_pairs > 0
            and self.holdout_reference_flux_q05_q95_amplitude is not None
            and self.holdout_median_reference_flux_error is not None
            and self.holdout_normalized_rmse is not None
            and self.holdout_median_bias_normalized is not None
            and self.maximum_absolute_time_separation is not None
            and self.fitted_offset is not None
            and self.fitted_scale is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-safe fold-result representation."""

        return {
            "schema_version": self.schema_version,
            "fold_index": self.fold_index,
            "n_training_pairs": self.n_training_pairs,
            "n_holdout_pairs": self.n_holdout_pairs,
            "successful": self.successful,
            "holdout_reference_flux_q05_q95_amplitude": (
                self.holdout_reference_flux_q05_q95_amplitude
            ),
            "holdout_median_reference_flux_error": (
                self.holdout_median_reference_flux_error
            ),
            "normalization_amplitude_to_median_reference_error": (
                self.normalization_amplitude_to_median_reference_error
            ),
            "holdout_normalized_rmse": self.holdout_normalized_rmse,
            "holdout_median_bias_normalized": (
                self.holdout_median_bias_normalized
            ),
            "maximum_absolute_time_separation": (
                self.maximum_absolute_time_separation
            ),
            "fitted_offset": self.fitted_offset,
            "fitted_scale": self.fitted_scale,
            "failure_reasons": list(self.failure_reasons),
            "complete": self.complete,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> InstrumentChannelCalibrationValidationFoldResult:
        """Construct one fold result from a strict representation."""

        if not isinstance(payload, dict):
            raise TypeError(
                "Calibration-validation fold result must be a dictionary."
            )
        expected_keys = {
            "schema_version",
            "fold_index",
            "n_training_pairs",
            "n_holdout_pairs",
            "successful",
            "holdout_reference_flux_q05_q95_amplitude",
            "holdout_median_reference_flux_error",
            "normalization_amplitude_to_median_reference_error",
            "holdout_normalized_rmse",
            "holdout_median_bias_normalized",
            "maximum_absolute_time_separation",
            "fitted_offset",
            "fitted_scale",
            "failure_reasons",
            "complete",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "Calibration-validation fold-result payload must contain "
                "exactly the contract fields."
            )
        result = cls(
            schema_version=payload["schema_version"],
            fold_index=payload["fold_index"],
            n_training_pairs=payload["n_training_pairs"],
            n_holdout_pairs=payload["n_holdout_pairs"],
            successful=payload["successful"],
            holdout_reference_flux_q05_q95_amplitude=payload[
                "holdout_reference_flux_q05_q95_amplitude"
            ],
            holdout_median_reference_flux_error=payload[
                "holdout_median_reference_flux_error"
            ],
            holdout_normalized_rmse=payload["holdout_normalized_rmse"],
            holdout_median_bias_normalized=payload[
                "holdout_median_bias_normalized"
            ],
            maximum_absolute_time_separation=payload[
                "maximum_absolute_time_separation"
            ],
            fitted_offset=payload["fitted_offset"],
            fitted_scale=payload["fitted_scale"],
            failure_reasons=tuple(payload["failure_reasons"]),
        )
        if result.to_dict() != payload:
            raise ValueError(
                "Calibration-validation fold-result payload does not match "
                "the strict contract representation."
            )
        return result


@dataclass(frozen=True)
class InstrumentChannelCalibrationValidationSourceResult:
    """Observed fold-level validation evidence for one exact dataset."""

    dataset: InstrumentChannelCalibrationValidationDataset
    n_matched_pairs: int
    fold_results: tuple[InstrumentChannelCalibrationValidationFoldResult, ...]
    failure_reasons: tuple[str, ...] = ()
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_SOURCE_RESULT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_SOURCE_RESULT_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported calibration-validation source-result schema "
                f"version: {self.schema_version!r}."
            )
        if not isinstance(
            self.dataset,
            InstrumentChannelCalibrationValidationDataset,
        ):
            raise TypeError(
                "dataset must be an InstrumentChannelCalibrationValidationDataset."
            )
        object.__setattr__(
            self,
            "n_matched_pairs",
            _normalize_nonnegative_int(
                self.n_matched_pairs,
                name="n_matched_pairs",
            ),
        )
        fold_results = self.fold_results
        if isinstance(fold_results, (str, bytes)):
            raise TypeError("fold_results must be a sequence of fold results.")
        normalized_folds = tuple(fold_results)
        if any(
            not isinstance(
                fold_result,
                InstrumentChannelCalibrationValidationFoldResult,
            )
            for fold_result in normalized_folds
        ):
            raise TypeError(
                "fold_results must contain only "
                "InstrumentChannelCalibrationValidationFoldResult records."
            )
        fold_indices = tuple(
            fold_result.fold_index for fold_result in normalized_folds
        )
        if len(set(fold_indices)) != len(fold_indices):
            raise ValueError("fold_results must use unique fold_index values.")
        if fold_indices != tuple(range(len(fold_indices))):
            raise ValueError(
                "fold_results must be ordered contiguously from fold_index zero."
            )
        object.__setattr__(self, "fold_results", normalized_folds)
        reasons = self.failure_reasons
        if isinstance(reasons, (str, bytes)):
            raise TypeError("failure_reasons must be a sequence of strings.")
        normalized_reasons = tuple(
            _normalize_text(reason, name=f"failure_reasons[{index}]")
            for index, reason in enumerate(reasons)
        )
        if len(set(normalized_reasons)) != len(normalized_reasons):
            raise ValueError("failure_reasons must not contain duplicates.")
        object.__setattr__(self, "failure_reasons", normalized_reasons)

    @property
    def n_temporal_folds(self) -> int:
        """Return the number of explicitly recorded fold results."""

        return len(self.fold_results)

    @property
    def n_successful_temporal_folds(self) -> int:
        """Return the number of successful fold results."""

        return sum(fold_result.successful for fold_result in self.fold_results)

    @property
    def complete(self) -> bool:
        """Whether all pre-registered fold-level evidence is available."""

        return (
            not self.failure_reasons
            and bool(self.fold_results)
            and all(fold_result.complete for fold_result in self.fold_results)
        )

    def _complete_fold_values(self, name: str) -> tuple[float, ...] | None:
        if not self.complete:
            return None
        return tuple(
            float(getattr(fold_result, name))
            for fold_result in self.fold_results
        )

    @property
    def median_holdout_normalized_rmse(self) -> float | None:
        values = self._complete_fold_values("holdout_normalized_rmse")
        return None if values is None else float(median(values))

    @property
    def worst_fold_holdout_normalized_rmse(self) -> float | None:
        values = self._complete_fold_values("holdout_normalized_rmse")
        return None if values is None else max(values)

    @property
    def maximum_absolute_holdout_median_bias_normalized(self) -> float | None:
        values = self._complete_fold_values(
            "holdout_median_bias_normalized"
        )
        return None if values is None else max(abs(value) for value in values)

    @property
    def maximum_absolute_time_separation(self) -> float | None:
        values = self._complete_fold_values(
            "maximum_absolute_time_separation"
        )
        return None if values is None else max(values)

    @property
    def fitted_scale_minimum(self) -> float | None:
        values = self._complete_fold_values("fitted_scale")
        return None if values is None else min(values)

    @property
    def fitted_scale_maximum(self) -> float | None:
        values = self._complete_fold_values("fitted_scale")
        return None if values is None else max(values)

    @property
    def minimum_holdout_normalization_amplitude_to_median_reference_error(
        self,
    ) -> float | None:
        values = self._complete_fold_values(
            "normalization_amplitude_to_median_reference_error"
        )
        return None if values is None else min(values)

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-safe source-result representation."""

        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset.to_dict(),
            "n_matched_pairs": self.n_matched_pairs,
            "fold_results": [
                fold_result.to_dict()
                for fold_result in self.fold_results
            ],
            "n_temporal_folds": self.n_temporal_folds,
            "n_successful_temporal_folds": (
                self.n_successful_temporal_folds
            ),
            "median_holdout_normalized_rmse": (
                self.median_holdout_normalized_rmse
            ),
            "worst_fold_holdout_normalized_rmse": (
                self.worst_fold_holdout_normalized_rmse
            ),
            "maximum_absolute_holdout_median_bias_normalized": (
                self.maximum_absolute_holdout_median_bias_normalized
            ),
            "maximum_absolute_time_separation": (
                self.maximum_absolute_time_separation
            ),
            "fitted_scale_minimum": self.fitted_scale_minimum,
            "fitted_scale_maximum": self.fitted_scale_maximum,
            (
                "minimum_holdout_normalization_amplitude_to_median_"
                "reference_error"
            ): (
                self.minimum_holdout_normalization_amplitude_to_median_reference_error
            ),
            "failure_reasons": list(self.failure_reasons),
            "complete": self.complete,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> InstrumentChannelCalibrationValidationSourceResult:
        """Construct a source result from a strict representation."""

        if not isinstance(payload, dict):
            raise TypeError(
                "Calibration-validation source result must be a dictionary."
            )
        expected_keys = {
            "schema_version",
            "dataset",
            "n_matched_pairs",
            "fold_results",
            "n_temporal_folds",
            "n_successful_temporal_folds",
            "median_holdout_normalized_rmse",
            "worst_fold_holdout_normalized_rmse",
            "maximum_absolute_holdout_median_bias_normalized",
            "maximum_absolute_time_separation",
            "fitted_scale_minimum",
            "fitted_scale_maximum",
            (
                "minimum_holdout_normalization_amplitude_to_median_"
                "reference_error"
            ),
            "failure_reasons",
            "complete",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "Calibration-validation source-result payload must contain "
                "exactly the contract fields."
            )
        result = cls(
            schema_version=payload["schema_version"],
            dataset=InstrumentChannelCalibrationValidationDataset.from_dict(
                payload["dataset"]
            ),
            n_matched_pairs=payload["n_matched_pairs"],
            fold_results=tuple(
                InstrumentChannelCalibrationValidationFoldResult.from_dict(
                    item
                )
                for item in payload["fold_results"]
            ),
            failure_reasons=tuple(payload["failure_reasons"]),
        )
        if result.to_dict() != payload:
            raise ValueError(
                "Calibration-validation source-result payload does not match "
                "the strict contract representation."
            )
        return result


@dataclass(frozen=True)
class InstrumentChannelCalibrationValidationResult:
    """Immutable execution result bound to one exact protocol digest."""

    result_id: str
    result_version: str
    protocol_id: str
    protocol_version: str
    protocol_sha256: str
    rule_id: str
    execution_reference: str
    package_version: str
    package_commit: str
    executed_at_utc: str
    source_results: tuple[InstrumentChannelCalibrationValidationSourceResult, ...]
    execution_completed: bool
    execution_failure_reasons: tuple[str, ...] = ()
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_RESULT_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported calibration-validation result schema version: "
                f"{self.schema_version!r}."
            )
        for name in (
            "result_id",
            "result_version",
            "protocol_id",
            "protocol_version",
            "rule_id",
            "execution_reference",
            "package_version",
            "package_commit",
            "executed_at_utc",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_text(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "protocol_sha256",
            _normalize_sha256(self.protocol_sha256, name="protocol_sha256"),
        )
        if isinstance(self.source_results, (str, bytes)):
            raise TypeError("source_results must be a sequence of source results.")
        source_results = tuple(self.source_results)
        if any(
            not isinstance(
                result,
                InstrumentChannelCalibrationValidationSourceResult,
            )
            for result in source_results
        ):
            raise TypeError(
                "source_results must contain only "
                "InstrumentChannelCalibrationValidationSourceResult records."
            )
        dataset_ids = tuple(result.dataset.dataset_id for result in source_results)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("source_results must use unique dataset_id values.")
        object.__setattr__(self, "source_results", source_results)
        if not isinstance(self.execution_completed, bool):
            raise TypeError("execution_completed must be boolean.")
        reasons = self.execution_failure_reasons
        if isinstance(reasons, (str, bytes)):
            raise TypeError(
                "execution_failure_reasons must be a sequence of strings."
            )
        normalized_reasons = tuple(
            _normalize_text(
                reason,
                name=f"execution_failure_reasons[{index}]",
            )
            for index, reason in enumerate(reasons)
        )
        if len(set(normalized_reasons)) != len(normalized_reasons):
            raise ValueError(
                "execution_failure_reasons must not contain duplicates."
            )
        if self.execution_completed and normalized_reasons:
            raise ValueError(
                "A completed validation execution cannot record execution "
                "failure reasons."
            )
        if not self.execution_completed and not normalized_reasons:
            raise ValueError(
                "An incomplete validation execution requires at least one "
                "failure reason."
            )
        object.__setattr__(
            self,
            "execution_failure_reasons",
            normalized_reasons,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-safe result representation."""

        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "result_version": self.result_version,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "protocol_sha256": self.protocol_sha256,
            "rule_id": self.rule_id,
            "execution_reference": self.execution_reference,
            "package_version": self.package_version,
            "package_commit": self.package_commit,
            "executed_at_utc": self.executed_at_utc,
            "source_results": [result.to_dict() for result in self.source_results],
            "execution_completed": self.execution_completed,
            "execution_failure_reasons": list(self.execution_failure_reasons),
            "scientific_validation_claim_embedded": False,
            "catalogue_population_performed": False,
            "marker": INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> InstrumentChannelCalibrationValidationResult:
        """Construct an execution result from a strict representation."""

        if not isinstance(payload, dict):
            raise TypeError("Calibration-validation result must be a dictionary.")
        expected_keys = {
            "schema_version",
            "result_id",
            "result_version",
            "protocol_id",
            "protocol_version",
            "protocol_sha256",
            "rule_id",
            "execution_reference",
            "package_version",
            "package_commit",
            "executed_at_utc",
            "source_results",
            "execution_completed",
            "execution_failure_reasons",
            "scientific_validation_claim_embedded",
            "catalogue_population_performed",
            "marker",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "Calibration-validation result payload must contain exactly "
                "the contract fields."
            )
        result = cls(
            schema_version=payload["schema_version"],
            result_id=payload["result_id"],
            result_version=payload["result_version"],
            protocol_id=payload["protocol_id"],
            protocol_version=payload["protocol_version"],
            protocol_sha256=payload["protocol_sha256"],
            rule_id=payload["rule_id"],
            execution_reference=payload["execution_reference"],
            package_version=payload["package_version"],
            package_commit=payload["package_commit"],
            executed_at_utc=payload["executed_at_utc"],
            source_results=tuple(
                InstrumentChannelCalibrationValidationSourceResult.from_dict(item)
                for item in payload["source_results"]
            ),
            execution_completed=payload["execution_completed"],
            execution_failure_reasons=tuple(
                payload["execution_failure_reasons"]
            ),
        )
        if payload["scientific_validation_claim_embedded"] is not False:
            raise ValueError(
                "Validation result records cannot embed a scientific-validation "
                "claim."
            )
        if payload["catalogue_population_performed"] is not False:
            raise ValueError(
                "Validation result records cannot populate a pairing-rule catalogue."
            )
        if payload["marker"] != INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER:
            raise ValueError("Calibration-validation result marker is invalid.")
        if result.to_dict() != payload:
            raise ValueError(
                "Calibration-validation result payload does not match the "
                "strict contract representation."
            )
        return result


@dataclass(frozen=True)
class InstrumentChannelCalibrationValidationReport:
    """Deterministic fail-closed assessment of one protocol/result pair."""

    result_id: str
    disposition: InstrumentChannelCalibrationValidationDisposition | str
    protocol_identity_matches: bool
    protocol_digest_matches: bool
    anchor_dataset_present: bool
    independent_astrophysical_source_count: int
    evaluated_astrophysical_source_ids: tuple[str, ...]
    passed_astrophysical_source_ids: tuple[str, ...]
    failed_astrophysical_source_ids: tuple[str, ...]
    inconclusive_astrophysical_source_ids: tuple[str, ...]
    all_acceptance_criteria_passed: bool
    reasons: tuple[str, ...]
    schema_version: str = (
        INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_REPORT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            INSTRUMENT_CHANNEL_CALIBRATION_VALIDATION_REPORT_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported calibration-validation report schema version: "
                f"{self.schema_version!r}."
            )
        object.__setattr__(
            self,
            "result_id",
            _normalize_text(self.result_id, name="result_id"),
        )
        disposition = self.disposition
        if not isinstance(
            disposition,
            InstrumentChannelCalibrationValidationDisposition,
        ):
            try:
                disposition = InstrumentChannelCalibrationValidationDisposition(
                    str(disposition)
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported calibration-validation disposition: {disposition!r}."
                ) from exc
        object.__setattr__(self, "disposition", disposition)
        for name in (
            "protocol_identity_matches",
            "protocol_digest_matches",
            "anchor_dataset_present",
            "all_acceptance_criteria_passed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        object.__setattr__(
            self,
            "independent_astrophysical_source_count",
            _normalize_nonnegative_int(
                self.independent_astrophysical_source_count,
                name="independent_astrophysical_source_count",
            ),
        )
        for name in (
            "evaluated_astrophysical_source_ids",
            "passed_astrophysical_source_ids",
            "failed_astrophysical_source_ids",
            "inconclusive_astrophysical_source_ids",
        ):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)):
                raise TypeError(f"{name} must be a sequence of strings.")
            normalized = tuple(
                _normalize_text(item, name=f"{name}[{index}]")
                for index, item in enumerate(value)
            )
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{name} must not contain duplicates.")
            object.__setattr__(self, name, normalized)
        reasons = self.reasons
        if isinstance(reasons, (str, bytes)):
            raise TypeError("reasons must be a sequence of strings.")
        normalized_reasons = tuple(
            _normalize_text(reason, name=f"reasons[{index}]")
            for index, reason in enumerate(reasons)
        )
        if len(set(normalized_reasons)) != len(normalized_reasons):
            raise ValueError("reasons must not contain duplicates.")
        object.__setattr__(self, "reasons", normalized_reasons)
        if disposition is InstrumentChannelCalibrationValidationDisposition.PASSED:
            if normalized_reasons:
                raise ValueError("A passed validation report cannot contain reasons.")
            if not self.all_acceptance_criteria_passed:
                raise ValueError(
                    "A passed validation report requires all criteria to pass."
                )
            if (
                self.failed_astrophysical_source_ids
                or self.inconclusive_astrophysical_source_ids
            ):
                raise ValueError(
                    "A passed validation report cannot contain failed or "
                    "inconclusive sources."
                )
        elif not normalized_reasons:
            raise ValueError(
                "Failed and inconclusive validation reports require reasons."
            )
        if (
            disposition is InstrumentChannelCalibrationValidationDisposition.FAILED
            and not self.failed_astrophysical_source_ids
        ):
            raise ValueError(
                "A failed validation report requires at least one failed source."
            )

    @property
    def passed(self) -> bool:
        """Whether the complete prospective validation passed."""

        return self.disposition is (
            InstrumentChannelCalibrationValidationDisposition.PASSED
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-safe report representation."""

        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "disposition": self.disposition.value,
            "protocol_identity_matches": self.protocol_identity_matches,
            "protocol_digest_matches": self.protocol_digest_matches,
            "anchor_dataset_present": self.anchor_dataset_present,
            "independent_astrophysical_source_count": (
                self.independent_astrophysical_source_count
            ),
            "evaluated_astrophysical_source_ids": list(
                self.evaluated_astrophysical_source_ids
            ),
            "passed_astrophysical_source_ids": list(
                self.passed_astrophysical_source_ids
            ),
            "failed_astrophysical_source_ids": list(
                self.failed_astrophysical_source_ids
            ),
            "inconclusive_astrophysical_source_ids": list(
                self.inconclusive_astrophysical_source_ids
            ),
            "all_acceptance_criteria_passed": (
                self.all_acceptance_criteria_passed
            ),
            "reasons": list(self.reasons),
            "passed": self.passed,
            "scientific_validation_executed_by_assessor": False,
            "eligible_for_caller_review_as_validation_evidence": self.passed,
            "catalogue_population_performed": False,
            "marker": INSTRUMENT_CHANNEL_CALIBRATION_TBD_MARKER,
        }


def _source_result_is_structurally_complete(
    source_result: InstrumentChannelCalibrationValidationSourceResult,
    protocol: InstrumentChannelCalibrationValidationProtocol,
) -> bool:
    """Return whether one source supplies complete protocol-conforming evidence."""

    criteria = protocol.acceptance_criteria
    if not source_result.complete:
        return False
    if source_result.n_matched_pairs < criteria.minimum_matched_pairs_per_source:
        return False
    if source_result.n_temporal_folds != criteria.temporal_fold_count:
        return False
    if (
        source_result.n_successful_temporal_folds
        != source_result.n_temporal_folds
    ):
        return False
    if any(
        fold_result.n_training_pairs < protocol.minimum_fit_pairs
        or fold_result.n_holdout_pairs
        < criteria.minimum_holdout_pairs_per_fold
        for fold_result in source_result.fold_results
    ):
        return False
    adequacy = (
        source_result
        .minimum_holdout_normalization_amplitude_to_median_reference_error
    )
    if (
        adequacy is None
        or adequacy
        < (
            criteria
            .minimum_holdout_normalization_amplitude_to_median_reference_error
        )
    ):
        return False
    maximum_separation = protocol.maximum_time_separation
    if (
        maximum_separation is not None
        and source_result.maximum_absolute_time_separation
        > maximum_separation + 1.0e-12
    ):
        return False
    return True


def _source_result_passes_quantitative_gates(
    source_result: InstrumentChannelCalibrationValidationSourceResult,
    protocol: InstrumentChannelCalibrationValidationProtocol,
) -> bool:
    """Return whether complete source evidence passes the metric thresholds."""

    criteria = protocol.acceptance_criteria
    return (
        source_result.median_holdout_normalized_rmse
        <= criteria.maximum_median_holdout_normalized_rmse
        and source_result.worst_fold_holdout_normalized_rmse
        <= criteria.maximum_worst_fold_holdout_normalized_rmse
        and source_result.maximum_absolute_holdout_median_bias_normalized
        <= criteria.maximum_absolute_holdout_median_bias_normalized
    )


def assess_instrument_channel_calibration_validation_result(
    protocol: InstrumentChannelCalibrationValidationProtocol,
    result: InstrumentChannelCalibrationValidationResult,
) -> InstrumentChannelCalibrationValidationReport:
    """Assess one result against a prospectively frozen protocol.

    The assessment is structural and deterministic. It does not run fitting,
    alter the protocol, construct a pairing rule, or populate a catalogue.
    Derived datasets never increase the independent astrophysical-source count.
    Incomplete folds, insufficient held-out evidence, and protocol violations
    are inconclusive rather than calibration failures. Only complete evidence
    that exceeds a pre-registered quantitative gate is classified as failed.
    """

    if not isinstance(protocol, InstrumentChannelCalibrationValidationProtocol):
        raise TypeError(
            "protocol must be an InstrumentChannelCalibrationValidationProtocol."
        )
    if not isinstance(result, InstrumentChannelCalibrationValidationResult):
        raise TypeError(
            "result must be an InstrumentChannelCalibrationValidationResult."
        )

    identity_matches = (
        result.protocol_id == protocol.protocol_id
        and result.protocol_version == protocol.protocol_version
        and result.rule_id == protocol.rule_id
    )
    digest_matches = result.protocol_sha256 == protocol.canonical_sha256
    anchor_present = any(
        source_result.dataset == protocol.anchor_dataset
        for source_result in result.source_results
    )

    primary_by_source: dict[
        str,
        list[InstrumentChannelCalibrationValidationSourceResult],
    ] = {}
    for source_result in result.source_results:
        if source_result.dataset.is_derived:
            continue
        primary_by_source.setdefault(
            source_result.dataset.astrophysical_source_id,
            [],
        ).append(source_result)

    evaluated_ids = tuple(sorted(primary_by_source))
    passed_ids: list[str] = []
    failed_ids: list[str] = []
    inconclusive_ids: list[str] = []
    duplicate_primary = False

    for source_id in evaluated_ids:
        source_results = primary_by_source[source_id]
        if len(source_results) != 1:
            duplicate_primary = True
            inconclusive_ids.append(source_id)
            continue
        source_result = source_results[0]
        if not _source_result_is_structurally_complete(
            source_result,
            protocol,
        ):
            inconclusive_ids.append(source_id)
        elif _source_result_passes_quantitative_gates(
            source_result,
            protocol,
        ):
            passed_ids.append(source_id)
        else:
            failed_ids.append(source_id)

    independent_count = len(evaluated_ids)
    reasons: list[str] = []
    if not result.execution_completed:
        reasons.append("validation_execution_incomplete")
    if not identity_matches:
        reasons.append("result_protocol_identity_mismatch")
    if not digest_matches:
        reasons.append("result_protocol_digest_mismatch")
    if not anchor_present:
        reasons.append("anchor_dataset_missing_or_mismatched")
    if duplicate_primary:
        reasons.append("multiple_primary_datasets_for_astrophysical_source")
    if inconclusive_ids:
        reasons.append("source_results_inconclusive")
    if (
        independent_count
        < protocol.acceptance_criteria.minimum_independent_astrophysical_sources
    ):
        reasons.append("insufficient_independent_astrophysical_sources")
    if failed_ids:
        reasons.append("source_acceptance_criteria_failed")

    structural_inconclusive = any(
        reason
        in {
            "validation_execution_incomplete",
            "result_protocol_identity_mismatch",
            "result_protocol_digest_mismatch",
            "anchor_dataset_missing_or_mismatched",
            "multiple_primary_datasets_for_astrophysical_source",
            "source_results_inconclusive",
            "insufficient_independent_astrophysical_sources",
        }
        for reason in reasons
    )

    if structural_inconclusive:
        disposition = InstrumentChannelCalibrationValidationDisposition.INCONCLUSIVE
    elif failed_ids:
        disposition = InstrumentChannelCalibrationValidationDisposition.FAILED
    else:
        disposition = InstrumentChannelCalibrationValidationDisposition.PASSED

    all_passed = (
        disposition
        is InstrumentChannelCalibrationValidationDisposition.PASSED
    )
    return InstrumentChannelCalibrationValidationReport(
        result_id=result.result_id,
        disposition=disposition,
        protocol_identity_matches=identity_matches,
        protocol_digest_matches=digest_matches,
        anchor_dataset_present=anchor_present,
        independent_astrophysical_source_count=independent_count,
        evaluated_astrophysical_source_ids=evaluated_ids,
        passed_astrophysical_source_ids=tuple(passed_ids),
        failed_astrophysical_source_ids=tuple(failed_ids),
        inconclusive_astrophysical_source_ids=tuple(inconclusive_ids),
        all_acceptance_criteria_passed=all_passed,
        reasons=tuple(reasons),
    )
