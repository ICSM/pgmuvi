import contextlib
import csv
import copy
import datetime
import enum
import random
import subprocess
import sys
from pathlib import Path
import time
from typing import ClassVar
import numpy as np
import torch
import gpytorch
from .gps import (
    SpectralMixtureLinearMeanGPModel,
    SpectralMixtureLinearMeanKISSGPModel,
    TwoDSpectralMixtureLinearMeanGPModel,
    TwoDSpectralMixtureLinearMeanKISSGPModel,
    SpectralMixtureGPModel,
    SpectralMixtureKISSGPModel,
    TwoDSpectralMixtureGPModel,
    TwoDSpectralMixtureKISSGPModel,
    QuasiPeriodicGPModel,
    MaternGPModel,
    PeriodicPlusStochasticGPModel,
    SeparableGPModel,
    AchromaticGPModel,
    WavelengthDependentGPModel,
    LinearMeanQuasiPeriodicGPModel,
    TwoDSpectralMixturePowerLawMeanGPModel,
    TwoDSpectralMixturePowerLawMeanKISSGPModel,
    TwoDSpectralMixtureDustMeanGPModel,
    TwoDSpectralMixtureDustMeanKISSGPModel,
    DustMeanGPModel,
    PowerLawMeanGPModel,
)
import matplotlib.pyplot as plt
from .trainers import train
from .initialization import _to_numpy
from gpytorch.constraints import Interval, GreaterThan, LessThan, Positive  # noqa: F401
from .constraints import get_constraint_set
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior  # noqa: F401
from .priors import (
    LogNormalFrequencyPrior,
    LogNormalPeriodPrior,
    NormalFrequencyPrior,
    NormalPeriodPrior,
    get_prior_set,
)
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from inspect import isclass
import xarray as xr
import arviz as az
import warnings
import dataclasses
import json
import math
from types import MappingProxyType
from pgmuvi.parameter_context import (
    LightcurveDiagnostics,
    ParameterEstimationContext,
)
from pgmuvi.parameter_workflow import (
    build_and_apply_parameter_estimates,
    model_supports_parameter_workflow,
)
from pgmuvi.constraint_utils import register_constraint_preserving_value

try:
    from scipy.signal import find_peaks as _scipy_find_peaks
except ImportError:
    _scipy_find_peaks = None


class ConsensusFitError(RuntimeError):
    """Raised when the consensus-fit pipeline cannot produce a valid result.

    This exception is raised instead of a bare ``RuntimeError`` whenever the
    consensus-fit algorithm determines that the data do not support a coherent
    shared period.  It is **not** raised for unrelated optimisation or GP
    errors; those continue to raise standard exceptions.

    Attributes
    ----------
    failure_diagnostics : dict
        A lightweight, JSON-safe structured description of the failure.
        The dict always contains the key ``"status": "failed"`` and a
        machine-readable ``"reason"`` string.  Additional keys vary by
        failure mode and are described in the individual ``reason`` values
        below.

        Common ``reason`` values:

        ``"no_accepted_bands"``
            Every band was rejected by the pre-LS sampling quality gate
            before any frequency could be extracted.  Extra keys:
            ``rejection_reasons`` (dict).

        ``"insufficient_consensus_inliers"``
            Accepted bands carry mutually inconsistent frequencies; after
            sigma-clipping fewer than ``min_consensus_inliers`` bands remain
            in the inlier cluster.  Extra keys: ``n_inlier_bands`` (int),
            ``required_inliers`` (int), ``n_candidate_bands`` (int),
            ``candidate_periods`` (list[float | None]).

        ``"frequency_aggregation_error"``
            An unexpected error occurred during robust frequency aggregation.
            Extra keys: ``detail`` (str).

        ``"invalid_consensus_frequency"``
            The aggregated consensus frequency is not finite or not strictly
            positive.  Extra keys: ``frequency_value`` (float | None).

    Parameters
    ----------
    message : str
        Human-readable description of the failure.  Must be scientifically
        informative and must not imply a software bug when the cause is a
        data-quality issue.
    failure_diagnostics : dict, optional
        Structured diagnostics dict (see ``failure_diagnostics`` attribute).
        If omitted, an empty ``{"status": "failed"}`` dict is attached.

    Examples
    --------
    >>> raise ConsensusFitError(
    ...     "Consensus fit failed: only 1 inlier band remained after period"
    ...     " consistency filtering (minimum required: 2).",
    ...     failure_diagnostics={
    ...         "status": "failed",
    ...         "reason": "insufficient_consensus_inliers",
    ...         "n_inlier_bands": 1,
    ...         "required_inliers": 2,
    ...         "n_candidate_bands": 4,
    ...         "candidate_periods": [18.0, 31.0, 47.0, 73.0],
    ...     },
    ... )
    """

    def __init__(self, message, *, failure_diagnostics=None):
        super().__init__(message)
        self.failure_diagnostics = (
            dict(failure_diagnostics)
            if failure_diagnostics is not None
            else {"status": "failed"}
        )
        self.failure_summary = None


_CONSENSUS_MIN_FREQUENCY_BOUND = 1.0e-12
_CONSENSUS_MIN_SCALE_BOUND = 1.0e-6
_CONSENSUS_MULTICOMP_DRIFT_WARNING_FRACTION = 0.10
# Extra LS peaks requested beyond max_components_per_band to ensure
# above-Nyquist alias peaks (which often dominate by raw power) are
# absorbed before the plausibility filter selects physical candidates.
_CONSENSUS_MULTICOMP_ALIAS_PEAK_BUFFER = 8
_ACF_STATUS_AGREEMENT = "agreement"
_ACF_STATUS_HARMONIC = "harmonic"
_ACF_STATUS_DISAGREEMENT = "disagreement"
_ACF_STATUS_UNAVAILABLE = "unavailable"
_CONSENSUS_BAND_STATUS_PENDING = "pending"
_CONSENSUS_BAND_STATUS_ACCEPTED = "accepted"
_CONSENSUS_BAND_STATUS_REJECTED = "rejected"
_CONSENSUS_GP_VALIDATION_STATUS_NOT_REQUESTED = "not_requested"
_CONSENSUS_GP_VALIDATION_STATUS_SKIPPED = "skipped"
_CONSENSUS_GP_VALIDATION_STATUS_FAILED = "failed"
_CONSENSUS_GP_VALIDATION_STATUS_SUCCESS = "success"
_CONSENSUS_GP_VALIDATION_STATUS_REJECTED = "rejected"
_CONSENSUS_GP_VALIDATION_REASON_BAND_NOT_ACCEPTED = "band_not_accepted"
_CONSENSUS_GP_VALIDATION_REASON_DIAGNOSTICS_FAILED = "diagnostics_failed"
_CONSENSUS_GP_VALIDATION_REASON_EXCEPTION = "exception"
_CONSENSUS_REJECTION_REASON_SAMPLING_METRICS_UNAVAILABLE = (
    "sampling metrics unavailable"
)
_CONSENSUS_REJECTION_REASON_NO_LS_PEAKS = "no_ls_peaks"
_CONSENSUS_REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK = (
    "no_physically_plausible_ls_peak"
)
_CONSENSUS_REJECTION_REASON_CANDIDATE_FREQUENCY_TOO_LOW = (
    "candidate_frequency_too_low"
)
_CONSENSUS_REJECTION_REASON_LS_ACF_DISAGREEMENT = "ls_acf_disagreement"
_CONSENSUS_REJECTION_REASON_GP_LS_DISAGREEMENT = "gp_ls_frequency_disagreement"
_CONSENSUS_REJECTION_REASON_GP_VALIDATION_FAILED = "gp_validation_failed"
_CONSENSUS_REJECTION_REASON_PREFIX_TOO_FEW_POINTS = "too_few_points ("
_CONSENSUS_REJECTION_REASON_PREFIX_MAX_GAP_FRACTION = "max_gap_fraction ("
_CONSENSUS_REJECTION_REASON_PREFIX_DUTY_CYCLE = "duty_cycle ("

# Set to True to enable lightweight structural validation at key consensus
# checkpoints inside _consensus_standard_fit.  Off by default to avoid
# performance overhead in production.  Can be toggled at runtime by setting
# pgmuvi.lightcurve._CONSENSUS_DEBUG_VALIDATE = True.
_CONSENSUS_DEBUG_VALIDATE = False

_CONSENSUS_ALLOWED_ACF_COMPARISON_STATUSES = frozenset(
    {
        _ACF_STATUS_AGREEMENT,
        _ACF_STATUS_HARMONIC,
        _ACF_STATUS_DISAGREEMENT,
        _ACF_STATUS_UNAVAILABLE,
    }
)
_CONSENSUS_ALLOWED_BAND_STATUSES = frozenset(
    {
        _CONSENSUS_BAND_STATUS_PENDING,
        _CONSENSUS_BAND_STATUS_ACCEPTED,
        _CONSENSUS_BAND_STATUS_REJECTED,
    }
)
_CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES = frozenset(
    {
        _CONSENSUS_GP_VALIDATION_STATUS_NOT_REQUESTED,
        _CONSENSUS_GP_VALIDATION_STATUS_SKIPPED,
        _CONSENSUS_GP_VALIDATION_STATUS_FAILED,
        _CONSENSUS_GP_VALIDATION_STATUS_SUCCESS,
        _CONSENSUS_GP_VALIDATION_STATUS_REJECTED,
    }
)
_CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES_SORTED = tuple(
    sorted(_CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES)
)
_CONSENSUS_GP_VALIDATION_STATUSES_CLEAR_REASON = frozenset(
    {
        _CONSENSUS_GP_VALIDATION_STATUS_NOT_REQUESTED,
        _CONSENSUS_GP_VALIDATION_STATUS_SUCCESS,
    }
)
_CONSENSUS_ALLOWED_GP_VALIDATION_REASONS = frozenset(
    {
        _CONSENSUS_GP_VALIDATION_REASON_BAND_NOT_ACCEPTED,
        _CONSENSUS_GP_VALIDATION_REASON_DIAGNOSTICS_FAILED,
        _CONSENSUS_GP_VALIDATION_REASON_EXCEPTION,
    }
)
_CONSENSUS_ALLOWED_REJECTION_REASONS = frozenset(
    {
        _CONSENSUS_REJECTION_REASON_SAMPLING_METRICS_UNAVAILABLE,
        _CONSENSUS_REJECTION_REASON_NO_LS_PEAKS,
        _CONSENSUS_REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK,
        _CONSENSUS_REJECTION_REASON_CANDIDATE_FREQUENCY_TOO_LOW,
        _CONSENSUS_REJECTION_REASON_LS_ACF_DISAGREEMENT,
        _CONSENSUS_REJECTION_REASON_GP_LS_DISAGREEMENT,
        _CONSENSUS_REJECTION_REASON_GP_VALIDATION_FAILED,
    }
)
_CONSENSUS_ALLOWED_REJECTION_REASON_PREFIXES = (
    _CONSENSUS_REJECTION_REASON_PREFIX_TOO_FEW_POINTS,
    _CONSENSUS_REJECTION_REASON_PREFIX_MAX_GAP_FRACTION,
    _CONSENSUS_REJECTION_REASON_PREFIX_DUTY_CYCLE,
)

# Fit-history schema registry.
_FIT_HISTORY_SCHEMA_VERSION = 2
_FIT_HISTORY_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})

# Allowed gp_validation_reason values for each gp_validation_status.  Used by
# _consensus_validate_result_structure to enforce the reason/status invariant.
_CONSENSUS_ALLOWED_GP_VALIDATION_REASONS_BY_STATUS = {
    _CONSENSUS_GP_VALIDATION_STATUS_NOT_REQUESTED: (None,),
    _CONSENSUS_GP_VALIDATION_STATUS_SUCCESS: (None,),
    _CONSENSUS_GP_VALIDATION_STATUS_SKIPPED: (
        None,
        _CONSENSUS_GP_VALIDATION_REASON_BAND_NOT_ACCEPTED,
    ),
    _CONSENSUS_GP_VALIDATION_STATUS_REJECTED: (
        None,
        _CONSENSUS_GP_VALIDATION_REASON_DIAGNOSTICS_FAILED,
    ),
    _CONSENSUS_GP_VALIDATION_STATUS_FAILED: (
        None,
        _CONSENSUS_GP_VALIDATION_REASON_EXCEPTION,
    ),
}
_CONSENSUS_ALLOWED_GP_VALIDATION_REASONS_BY_STATUS_SORTED = {
    status: tuple(
        sorted(val for val in allowed_reasons if val is not None)
    )
    for (
        status,
        allowed_reasons,
    ) in _CONSENSUS_ALLOWED_GP_VALIDATION_REASONS_BY_STATUS.items()
}
_CONSENSUS_ALLOWED_ACF_COMPARISON_STATUSES_SORTED = tuple(
    sorted(_CONSENSUS_ALLOWED_ACF_COMPARISON_STATUSES)
)
_CONSENSUS_ALLOWED_BAND_STATUSES_SORTED = tuple(
    sorted(_CONSENSUS_ALLOWED_BAND_STATUSES)
)
_CONSENSUS_ALLOWED_REJECTION_REASONS_SORTED = tuple(
    sorted(_CONSENSUS_ALLOWED_REJECTION_REASONS)
)

def _consensus_schema_field(
    *,
    default=None,
    default_factory=None,
    nullable=True,
    container_type=None,
    allowed_values=None,
    deprecated_alias=False,
    canonical_alias_for=None,
):
    """Build immutable schema metadata for one diagnostics field.

    Parameters
    ----------
    default : object, optional
        Scalar default value used when ``default_factory`` is not set.
    default_factory : {"list", "dict"} or None, optional
        Factory identifier for container defaults. When set, a fresh container
        is created per-record at initialization.
    nullable : bool, optional
        Whether ``None`` is considered a valid value for the field.
    container_type : {"list", "dict"} or None, optional
        Expected container type for validator type checks.
    allowed_values : iterable or None, optional
        Optional categorical domain for validator membership checks.
    deprecated_alias : bool, optional
        Whether this field is a deprecated alias retained for compatibility.
    canonical_alias_for : str or None, optional
        Canonical field name referenced by a deprecated alias.

    Returns
    -------
    MappingProxyType
        Immutable field metadata mapping.
    """
    if default is not None and default_factory is not None:
        raise ValueError(
            "Consensus schema field cannot define both default and "
            "default_factory."
        )
    if default_factory is not None and default_factory not in {"list", "dict"}:
        raise ValueError(
            "Consensus schema field default_factory must be one of "
            "{'list', 'dict'}."
        )
    if container_type is not None and container_type not in {"list", "dict"}:
        raise ValueError(
            "Consensus schema field container_type must be one of "
            "{'list', 'dict'} or None."
        )
    if allowed_values is None:
        allowed_values_tuple = None
    else:
        allowed_values_tuple = tuple(allowed_values)
    return MappingProxyType(
        {
            "default": default,
            "default_factory": default_factory,
            "nullable": bool(nullable),
            "container_type": container_type,
            "allowed_values": allowed_values_tuple,
            "deprecated_alias": bool(deprecated_alias),
            "canonical_alias_for": canonical_alias_for,
        }
    )


def _consensus_schema_default_record(schema_fields):
    """Instantiate a mutable diagnostics record from immutable schema metadata.

    Parameters
    ----------
    schema_fields : Mapping[str, Mapping]
        Canonical schema field definitions where each value is a metadata
        mapping produced by :func:`_consensus_schema_field`.

    Returns
    -------
    dict
        Mutable diagnostics record containing one initialized value per schema
        field key.
    """
    record = {}
    for key, metadata in schema_fields.items():
        default_factory = metadata.get("default_factory")
        if default_factory == "list":
            record[key] = []
        elif default_factory == "dict":
            record[key] = {}
        else:
            record[key] = metadata.get("default")
    return record


_CONSENSUS_TOP_LEVEL_SCHEMA_FIELDS = MappingProxyType(
    {
        "fit_strategy": _consensus_schema_field(default="consensus", nullable=False),
        "consensus_success": _consensus_schema_field(default=False, nullable=False),
        "consensus_frequency": _consensus_schema_field(default=None, nullable=True),
        "consensus_period": _consensus_schema_field(default=None, nullable=True),
        "consensus_frequency_width": _consensus_schema_field(
            default=None, nullable=True
        ),
        "consensus_frequency_scatter": _consensus_schema_field(
            default=None, nullable=True
        ),
        "accepted_bands": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "rejected_bands": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "rejection_summary": _consensus_schema_field(
            default_factory="dict",
            nullable=False,
            container_type="dict",
            deprecated_alias=True,
            canonical_alias_for="rejection_reasons",
        ),
        "per_band_diagnostics": _consensus_schema_field(
            default_factory="dict", nullable=False, container_type="dict"
        ),
        "n_total_bands": _consensus_schema_field(default=0, nullable=False),
        "n_accepted_bands": _consensus_schema_field(default=0, nullable=False),
        "n_rejected_bands": _consensus_schema_field(default=0, nullable=False),
        "use_acf_validation": _consensus_schema_field(default=False, nullable=False),
        "use_gp_validation": _consensus_schema_field(default=False, nullable=False),
        "gp_validation_requested": _consensus_schema_field(
            default=False, nullable=False
        ),
        "gp_validation_performed": _consensus_schema_field(
            default=False, nullable=False
        ),
        "trusted_candidate_count": _consensus_schema_field(
            default=None, nullable=True
        ),
        "candidate_count": _consensus_schema_field(default=None, nullable=True),
        "consensus_generation_method": _consensus_schema_field(
            default=None, nullable=True
        ),
        "rejection_reasons": _consensus_schema_field(
            default_factory="dict", nullable=False, container_type="dict"
        ),
        "per_band_dominant_periods": _consensus_schema_field(
            default_factory="dict", nullable=False, container_type="dict"
        ),
        "per_band_dominant_frequencies": _consensus_schema_field(
            default_factory="dict", nullable=False, container_type="dict"
        ),
        "median_frequency": _consensus_schema_field(default=None, nullable=True),
        "mad_frequency_scatter": _consensus_schema_field(default=None, nullable=True),
        "consensus_inlier_bands": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "consensus_outlier_bands": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "final_consensus_frequency": _consensus_schema_field(
            default=None, nullable=True
        ),
        "final_consensus_period": _consensus_schema_field(default=None, nullable=True),
        "robust_frequency_width": _consensus_schema_field(default=None, nullable=True),
        "final_constraint_bounds": _consensus_schema_field(
            default=None, nullable=True
        ),
        "controls": _consensus_schema_field(
            default_factory="dict", nullable=False, container_type="dict"
        ),
        "mode": _consensus_schema_field(default=None, nullable=True),
        "consensus_constraints_applied": _consensus_schema_field(
            default=False, nullable=False
        ),
        "consensus_constraint_bounds": _consensus_schema_field(
            default=None, nullable=True
        ),
        "consensus_constraint_target_key": _consensus_schema_field(
            default=None, nullable=True
        ),
        "consensus_scale_constraint_bounds": _consensus_schema_field(
            default=None, nullable=True
        ),
        "consensus_scale_constraint_target_key": _consensus_schema_field(
            default=None, nullable=True
        ),
        "requested_consensus_frequencies": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "requested_consensus_scales": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "consensus_component_strengths": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "consensus_mixture_init_scales": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "initialized_mixture_means": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "initialized_mixture_scales": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "fitted_mixture_frequencies": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "fitted_mixture_periods": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "fitted_mixture_scales": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "fitted_mixture_period_widths": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "fitted_frequency_shift_from_initialization": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "fitted_period_shift_from_initialization": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        (
            "fitted_fractional_frequency_shift_from_initialization"
        ): _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        (
            "fitted_fractional_period_shift_from_initialization"
        ): _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "max_abs_fractional_period_shift_from_initialization": _consensus_schema_field(
            default=None, nullable=True
        ),
        "max_abs_fractional_frequency_shift_from_initialization": _consensus_schema_field(
            default=None, nullable=True
        ),
        "component_fit_drift_flags": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "components_with_large_period_drift": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "components_with_large_frequency_drift": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "nearest_initialized_component_index": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "nearest_initialized_component_fractional_period_distance": (
            _consensus_schema_field(
                default_factory="list", nullable=False, container_type="list"
            )
        ),
        "nearest_initialized_component_fractional_frequency_distance": (
            _consensus_schema_field(
                default_factory="list", nullable=False, container_type="list"
            )
        ),
        "component_identity_preserved": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "all_component_identities_preserved": _consensus_schema_field(
            default=True, nullable=False
        ),
        "possible_component_swaps": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "drift_warning_fraction": _consensus_schema_field(
            default=float(_CONSENSUS_MULTICOMP_DRIFT_WARNING_FRACTION),
            nullable=False,
        ),
        "initialization_strategy": _consensus_schema_field(
            default=None, nullable=True
        ),
        "default_constraints_applied_before_consensus": _consensus_schema_field(
            default=False, nullable=False
        ),
        "constraints_marked_set_after_consensus": _consensus_schema_field(
            default=False, nullable=False
        ),
    }
)
_CONSENSUS_TOP_LEVEL_SCHEMA = MappingProxyType(
    {
        "fields": _CONSENSUS_TOP_LEVEL_SCHEMA_FIELDS,
        "required_keys": tuple(_CONSENSUS_TOP_LEVEL_SCHEMA_FIELDS.keys()),
        "deprecated_alias_fields": tuple(
            key
            for key, metadata in _CONSENSUS_TOP_LEVEL_SCHEMA_FIELDS.items()
            if metadata["deprecated_alias"]
        ),
    }
)

_CONSENSUS_BAND_SCHEMA_FIELDS = MappingProxyType(
    {
        "band": _consensus_schema_field(default="", nullable=False),
        "status": _consensus_schema_field(
            default=_CONSENSUS_BAND_STATUS_PENDING,
            nullable=False,
            allowed_values=_CONSENSUS_ALLOWED_BAND_STATUSES_SORTED,
        ),
        "rejection_reason": _consensus_schema_field(default=None, nullable=True),
        "rejection_reasons": _consensus_schema_field(
            default_factory="list", nullable=False, container_type="list"
        ),
        "metrics": _consensus_schema_field(default=None, nullable=True),
        "dominant_frequency": _consensus_schema_field(default=None, nullable=True),
        "dominant_period": _consensus_schema_field(default=None, nullable=True),
        "ls_significant": _consensus_schema_field(default=None, nullable=True),
        "ls_peak_power": _consensus_schema_field(default=None, nullable=True),
        "ls_peak_prominence": _consensus_schema_field(default=None, nullable=True),
        "ls_peak_area_fraction": _consensus_schema_field(default=None, nullable=True),
        "acf_frequency": _consensus_schema_field(default=None, nullable=True),
        "acf_period": _consensus_schema_field(default=None, nullable=True),
        "acf_supported": _consensus_schema_field(default=None, nullable=True),
        "acf_comparison_status": _consensus_schema_field(
            default=None,
            nullable=True,
            allowed_values=_CONSENSUS_ALLOWED_ACF_COMPARISON_STATUSES_SORTED,
        ),
        "acf_period_ratio": _consensus_schema_field(default=None, nullable=True),
        "acf_harmonic_order": _consensus_schema_field(default=None, nullable=True),
        "acf_error": _consensus_schema_field(default=None, nullable=True),
        "selected_from": _consensus_schema_field(default=None, nullable=True),
        "gp_validation_used": _consensus_schema_field(default=False, nullable=False),
        "gp_dominant_frequency": _consensus_schema_field(default=None, nullable=True),
        "gp_dominant_period": _consensus_schema_field(default=None, nullable=True),
        "gp_frequency_difference": _consensus_schema_field(default=None, nullable=True),
        "gp_fractional_frequency_difference": _consensus_schema_field(
            default=None, nullable=True
        ),
        "gp_frequency_tolerance": _consensus_schema_field(default=None, nullable=True),
        "gp_validation_error": _consensus_schema_field(default=None, nullable=True),
        "gp_validation_status": _consensus_schema_field(
            default=_CONSENSUS_GP_VALIDATION_STATUS_NOT_REQUESTED,
            nullable=False,
            allowed_values=_CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES_SORTED,
        ),
        "gp_validation_reason": _consensus_schema_field(
            default=None,
            nullable=True,
            allowed_values=tuple(sorted(_CONSENSUS_ALLOWED_GP_VALIDATION_REASONS)),
        ),
    }
)
_CONSENSUS_BAND_SCHEMA = MappingProxyType(
    {
        "fields": _CONSENSUS_BAND_SCHEMA_FIELDS,
        "required_keys": tuple(_CONSENSUS_BAND_SCHEMA_FIELDS.keys()),
        "deprecated_alias_fields": tuple(),
    }
)

# Required-key sets are derived from the canonical schema definitions.
_CONSENSUS_REQUIRED_RESULT_KEYS = frozenset(
    _CONSENSUS_TOP_LEVEL_SCHEMA["required_keys"]
)
_CONSENSUS_REQUIRED_BAND_KEYS = frozenset(_CONSENSUS_BAND_SCHEMA["required_keys"])


def _reraise_with_note(e, note):
    """Reraise an exception with a note added to the message

    This function is to provide a way to add a note to an exception, without
    losing the traceback, and without requiring python 3.11, which has
    added notes. It is based on this answer on stackoverflow:
    https://stackoverflow.com/a/75549200/16164384

    Parameters
    ----------
    e : Exception
        The exception to reraise
    note : str
        The note to add to the exception message
    """
    try:
        e.add_note(note)
    except AttributeError:
        args = e.args
        arg0 = f"{args[0]}\n{note}" if args else note
        e.args = (arg0, *args[1:])
    raise e


# Function to walk through nested dict and yield all values
# Taken from https://stackoverflow.com/a/12507546/16164384
def dict_walk_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                yield from dict_walk_generator(value, [*pre, key])
            elif isinstance(value, list | tuple):
                for v in value:
                    yield from dict_walk_generator(v, [*pre, key])
            else:
                yield [*pre, key, value]
    else:
        yield [*pre, indict]


def _convert_time_to_days(xdata, time_units):
    """Convert the time axis of xdata to days.

    Parameters
    ----------
    xdata : torch.Tensor, numpy.ndarray, or array-like
        The independent variable data.  For 1-D light curves this is a
        1-D (or single-column) array of time values.  For 2-D (multi-band)
        light curves this is a 2-D array of shape ``(N, 2)`` where column 0
        is time and column 1 is wavelength/band; only the time column is
        converted.  Non-tensor inputs are coerced to a ``torch.float32``
        tensor automatically.
    time_units : str, astropy.units.UnitBase, or None
        Units of the time values.  Any string accepted by
        ``astropy.units.Unit`` (e.g. ``'s'``, ``'hr'``, ``'yr'``,
        ``'days'``) and any ``astropy.units`` unit object are supported.
        If *None* the data are assumed to already be in days and are
        returned unchanged.

    Returns
    -------
    torch.Tensor
        ``xdata`` with the time axis expressed in days.

    Raises
    ------
    ValueError
        If *time_units* cannot be converted to days (e.g. it is a unit of
        length rather than time).
    """
    if time_units is None:
        return xdata

    import astropy.units as u

    if isinstance(time_units, str):
        unit = u.Unit(time_units)
    else:
        unit = time_units

    try:
        conversion_factor = float(unit.to(u.day))
    except u.UnitConversionError as e:
        raise ValueError(
            f"Cannot convert time_units '{time_units}' to days: {e}"
        ) from e

    # Coerce to tensor so .dim() / .shape are always available, regardless of
    # whether the caller passed a list, NumPy array, or torch.Tensor.
    if not isinstance(xdata, torch.Tensor):
        xdata = torch.as_tensor(xdata, dtype=torch.float32)

    if xdata.dim() <= 1 or xdata.shape[1] == 1:
        # 1-D light curve: all values are time
        return xdata * conversion_factor
    else:
        # 2-D light curve: column 0 is time, column 1 is wavelength
        xdata = xdata.clone()
        xdata[:, 0] = xdata[:, 0] * conversion_factor
        return xdata


class Transformer(torch.nn.Module):
    def __init__(self):
        """Baseclass for data transformers

        This is a baseclass for data transformers, which are used to transform
        data before it is passed to the GP.

        Parameters
        ----------

        Examples
        --------

        Notes
        -----
        The baseclass has no implementation, and should not be used directly.
        `__init__` is only implemented to allow the class to be subclassed and
        ensure that `nn.Module` stuff is setup correctly. Subclasses should
        implement the `transform` and `inverse` methods."""
        super().__init__()

    def transform(self, data, **kwargs):
        """Transform some data and return it, storing the parameters required
        to repeat or reverse the transform

        This is a baseclass with no implementations, your subclass should
        implement the transform itself
        """
        raise NotImplementedError

    def inverse(self, data, shift=True, **kwargs):
        """Invert a transform based on saved parameters

        This is a baseclass with no implementation, your subclass should
        implement the inverse transform itself
        """
        raise NotImplementedError


class MinMax(Transformer):
    def transform(self, data, dim=0, apply_to=None, recalc=False, shift=True, **kwargs):
        """Perform a MinMax transformation

        Transform the data such that each dimension is rescaled to the [0,1]
        interval. It stores the min and range of the data for the inverse
        transformation.

        Parameters
        ----------
        data : Tensor of floats
            The data to be transformed
        apply_to : tensor of ints or slice objects, optional
            Which dimensions to apply the transform to. If None, apply to all
        recalc : bool, default False
            Should the min and range of the transform be recalculated, or
            reused from previously?
        shift : bool, default True
            Should the data be shifted such that the minimum value is 0?
            This is mainly included so that data or parameters can be
            transformed when they apply to a single period - in this case,
            only the range needs to be applied.
        """
        if recalc or not hasattr(self, "min"):
            self.register_buffer("min", torch.min(data, dim=dim, keepdim=True)[0])
            self.register_buffer(
                "range", torch.max(data, dim=dim, keepdim=True)[0] - self.min
            )
            shift = True  # if we're recalculating, we need to shift
        if apply_to is not None:
            return (data - (shift * self.min[apply_to])) / self.range[apply_to]
        return (data - (shift * self.min)) / self.range

    def inverse(self, data, shift=True, **kwargs):
        """Invert a MinMax transformation based on saved state

        Invert the transformation of the data from  the [0,1] interval.
        It used the stored min and range of the data for the inverse
        transformation.

        Parameters
        ----------
        data : Tensor of floats
            The data to be reverse-transformed
        """
        return (data * self.range) + (shift * self.min)


class ZScore(Transformer):
    def transform(self, data, dim=0, apply_to=None, recalc=False, shift=True, **kwargs):
        """Perform a z-score transformation

        Transform the data such that each dimension is rescaled such that
        its mean is 0 and its standard deviation is 1.

        Parameters
        ----------
        data : Tensor of floats
            The data to be transformed
        apply_to : int or tensor of ints, optional
            Which dimensions to apply the transform to. If None, apply to all
        recalc : bool, default False
            Should the parameters of the transform be recalculated, or reused
            from previously?
        shift : bool, default True
            Should the data be shifted such that the mean value is 0?
            This is mainly included so that data or parameters can be
            transformed when they apply to a single period - in this case,
            only the standard deviation needs to be applied.
        """
        if recalc or not hasattr(self, "mean"):
            mean = torch.mean(data, dim=dim, keepdim=True)[0]
            self.register_buffer("mean", mean)
            sd = torch.std(data, dim=dim, keepdim=True)[0]
            self.register_buffer("sd", sd)
            shift = True  # if we're recalculating, we need to shift
        if apply_to is not None:
            return (data - (shift * self.mean[apply_to])) / self.sd[apply_to]
        return (data - shift * self.mean) / self.sd

    def inverse(self, data, shift=True, **kwargs):
        """Invert a z-score transformation based on saved state

        Invert the z-scoring of the data based on the saved mean and standard
        deviation

        Parameters
        ----------
        data : Tensor of floats
            The data to be reverse-transformed
        """
        return (data * self.sd) + (self.mean * shift)


class RobustZScore(Transformer):
    def transform(self, data, dim=0, apply_to=None, recalc=False, shift=True, **kwargs):
        """Perform a robust z-score transformation

        Transform the data such that each dimension is rescaled such that
        its median is 0 and its median absolute deviation is 1.

        Parameters
        ----------
        data : Tensor of floats
            The data to be transformed
        apply_to : int or tensor of ints, optional
            Which dimensions to apply the transform to. If None, apply to all
        recalc : bool, default False
            Should the parameters of the transform be recalculated, or reused
            from previously?
        shift : bool, default True
            Should the data be shifted such that the median value is 0?
            This is mainly included so that data or parameters can be
            transformed when they apply to a single period - in this case,
            only the median absolute deviation needs to be applied.
        """
        if recalc or not hasattr(self, "mad"):
            median = torch.median(data, dim=dim, keepdim=True)[0]
            self.register_buffer("median", median)
            mad = torch.median(torch.abs(data - median), dim=dim, keepdim=True)[0]
            self.register_buffer("mad", mad)
            shift = True  # if we're recalculating, we need to shift
        if apply_to is not None:
            return (data - shift * self.median[apply_to]) / self.mad[apply_to]
        return (data - shift * self.median) / self.mad

    def inverse(self, data, shift=True, **kwargs):
        """Invert a robust z-score transformation based on saved state

        Invert the robust z-scoring of the data based on the saved median and
        median absolute deviation.

        Parameters
        ----------
        data : Tensor of floats
            The data to be reverse-transformed
        """
        return (data * self.mad) + (self.median * shift)


def minmax(data, dim=0):
    m = torch.min(data, dim=dim, keepdim=True)
    r = torch.max(data, dim=dim, keepdim=True) - m
    return (data - m) / r, m, r


class InputHelpers:
    """Mixin class providing helper methods for reading data from various input formats.

    This class provides classmethods for instantiating a :class:`Lightcurve`
    from different input formats, with flexible column name detection.
    :class:`Lightcurve` inherits from this class so all methods are available
    directly on :class:`Lightcurve`.

    Attributes
    ----------
    _X_COLUMN_NAMES : list of str
        Candidate column names used for auto-detecting the time (independent
        variable) column, checked case-insensitively in order.
    _Y_COLUMN_NAMES : list of str
        Candidate column names used for auto-detecting the dependent variable
        (y) column, checked case-insensitively in order.
    _YERR_COLUMN_NAMES : list of str
        Candidate column names used for auto-detecting the uncertainty column,
        checked case-insensitively in order.
    _WAVELENGTH_COLUMN_NAMES : list of str
        Candidate column names used for auto-detecting a *numeric* wavelength
        column, checked case-insensitively in order.  When such a column is
        found and contains more than one unique value, the data are loaded as
        a 2-D lightcurve whose ``xdata`` has shape ``(N, 2)`` with the time
        values in column 0 and the wavelength values in column 1.
    _WAVELENGTH_ID_COLUMN_NAMES : list of str
        Candidate column names used for auto-detecting a *string* band
        identifier column (e.g. ``"V"``, ``"R"``, ``"W1"``), checked
        case-insensitively in order.  When such a column is found it is
        ingested as the ``band`` attribute of the resulting
        :class:`Lightcurve`.
    """

    _X_COLUMN_NAMES: ClassVar[list[str]] = [
        "x", "time", "t", "jd", "mjd", "date", "hjd", "bjd", "epoch"
    ]
    _Y_COLUMN_NAMES: ClassVar[list[str]] = [
        "y", "magnitude", "mag", "flux", "value", "data"
    ]
    _YERR_COLUMN_NAMES: ClassVar[list[str]] = [
        "yerr",
        "uncertainty",
        "error",
        "err",
        "unc",
        "sigma",
        "e_magnitude",
        "e_mag",
        "e_flux",
        "flux_error",
        "mag_error",
        "magnitude_error",
        "value_error",
        "data_error",
        "y_error",
    ]
    _WAVELENGTH_COLUMN_NAMES: ClassVar[list[str]] = [
        "wavelength",
        "wave",
        "wl",
        "lambda",
        "freq",
        "frequency",
        "channel",
    ]
    _WAVELENGTH_ID_COLUMN_NAMES: ClassVar[list[str]] = [
        "band",
        "filter",
        "filtername",
        "filter_name",
    ]

    @classmethod
    def _find_column(
        cls, columns: list[str], candidates: list[str]
    ) -> str | None:
        """Find the first matching column name from a list of candidates.

        Matching is case-insensitive.

        Parameters
        ----------
        columns : list of str
            The available column names.
        candidates : list of str
            Candidate column names to search for, in priority order.

        Returns
        -------
        str or None
            The matched column name (preserving the original capitalisation
            from *columns*), or ``None`` if no candidate was found.
        """
        columns_lower = {c.lower(): c for c in columns}
        for candidate in candidates:
            if candidate.lower() in columns_lower:
                return columns_lower[candidate.lower()]
        return None

    @staticmethod
    def _drop_nonfinite_rows(x, y, yerr):
        """Drop rows containing non-finite (NaN or Inf) values from data arrays.

        Parameters
        ----------
        x : torch.Tensor
            Independent variable tensor of shape ``(N,)`` or ``(N, D)``.
        y : torch.Tensor
            Dependent variable tensor of shape ``(N,)``.
        yerr : torch.Tensor or None
            Uncertainty tensor of shape ``(N,)``, or ``None``.

        Returns
        -------
        x : torch.Tensor
            Filtered independent variable tensor.
        y : torch.Tensor
            Filtered dependent variable tensor.
        yerr : torch.Tensor or None
            Filtered uncertainty tensor, or ``None`` if it was ``None`` on
            input.

        Notes
        -----
        A ``UserWarning`` is emitted when one or more rows are dropped.
        A ``ValueError`` is raised when no valid rows remain after filtering.
        """
        valid_mask = torch.isfinite(y)
        if x.dim() > 1:
            valid_mask &= torch.isfinite(x).all(dim=1)
        else:
            valid_mask &= torch.isfinite(x)
        if yerr is not None:
            valid_mask &= torch.isfinite(yerr)
        n_dropped = int((~valid_mask).sum().item())
        if n_dropped > 0:
            warnings.warn(
                f"Dropped {n_dropped} row(s) containing non-finite "
                "(NaN or Inf) values.",
                stacklevel=3,
            )
            x = x[valid_mask]
            y = y[valid_mask]
            if yerr is not None:
                yerr = yerr[valid_mask]
        if y.numel() == 0:
            raise ValueError(
                "No valid data rows remain after dropping non-finite rows."
            )
        elif y.numel() < 10 and n_dropped > 0:
            warnings.warn(
                f"Fewer than 10 elements remain after dropping {n_dropped} rows,"
                " take care interpreting results!",
                stacklevel=3,
            )
        return x, y, yerr

    @staticmethod
    def _drop_nan_rows(x, y, yerr):
        """Drop rows that contain NaN in any of the data arrays.

        .. deprecated:: 0.3.0
            Use :meth:`_drop_nonfinite_rows` instead, which also handles
            infinite values.
        """
        return Lightcurve._drop_nonfinite_rows(x, y, yerr)

    @classmethod
    def from_csv(
        cls,
        filepath: str | Path,
        xcol: str | list[str] | None = None,
        ycol: str | None = None,
        yerrcol: str | None = None,
        wavelcol: str | None = None,
        **kwargs,
    ) -> "Lightcurve":
        """Instantiate a Lightcurve from a CSV file.

        The file must have a header line whose entries are used to identify
        the relevant data columns.  Column names are matched
        case-insensitively.

        **1-D lightcurves** (single time series)
            When only a time column and a flux/magnitude column are present,
            or when all observations share the same wavelength/band, the
            resulting ``xdata`` is a 1-D tensor of shape ``(N,)``.

        **2-D (multiband) lightcurves**
            When the CSV contains a *numeric* wavelength column with more than
            one unique value, the resulting ``xdata`` has shape ``(N, 2)``
            where column 0 holds the time values and column 1 holds the
            numeric wavelength values.  The ``ydata`` (and optional ``yerr``)
            remain 1-D tensors of shape ``(N,)``.

            The numeric wavelength column is selected in one of three ways:

            1. *Explicit ``xcol`` list*: pass ``xcol`` as a list of two
               column names, e.g. ``xcol=["time", "wavelength"]``.  The first
               element is the time column and the second is the wavelength
               column.  All subsequent x-axis columns are stacked in the
               order given.
            2. *Explicit ``wavelcol``*: pass the column name as a separate
               ``wavelcol`` keyword argument.
            3. *Auto-detection*: if neither an iterable ``xcol`` nor a
               ``wavelcol`` is supplied, the method searches for a column
               whose name matches one of the entries in
               :attr:`_WAVELENGTH_COLUMN_NAMES` (e.g. ``"wavelength"``,
               ``"wl"``).  If such a column is found and contains more than
               one unique value, a 2-D lightcurve is returned automatically.

        **Band labels**
            String band-identifier columns (e.g. one named ``"band"`` or
            ``"filter"`` containing values like ``"V"``, ``"R"``) are
            resolved *independently* of the numeric wavelength column.  When
            the CSV contains a string-typed column whose name matches one of
            the entries in :attr:`_WAVELENGTH_ID_COLUMN_NAMES`, those labels
            are stored automatically in :attr:`Lightcurve.band` when
            ``band`` is not supplied explicitly in ``**kwargs``.  For 1-D
            lightcurves, auto-population occurs only when the band-ID column
            contains exactly one distinct non-empty label (matching the 1-D
            constructor contract, which expects a single band label). If
            multiple distinct labels are present for 1-D input, ``band`` is
            left unset and a warning is emitted. Numeric wavelength columns are
            used directly and
            :attr:`Lightcurve.band` is left as ``None`` unless the caller
            provides ``band=`` explicitly in ``**kwargs``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the CSV file.
        xcol : str or list of str or None, optional
            Name of the column containing the time (independent variable)
            data, or a list of column names to stack as the x-axis (first
            element is time, subsequent elements are additional dimensions
            such as wavelength).  If not provided, auto-detection is
            attempted using :attr:`_X_COLUMN_NAMES` for the time column.
        ycol : str or None, optional
            Name of the column containing the dependent variable (y) data.
            If not provided, auto-detection is attempted using
            :attr:`_Y_COLUMN_NAMES`.
        yerrcol : str or None, optional
            Name of the column containing the uncertainties on the dependent
            variable.  If not provided, auto-detection is attempted using
            :attr:`_YERR_COLUMN_NAMES`.  If no matching column is found,
            ``yerr`` is set to ``None``.
        wavelcol : str or None, optional
            Name of the column containing wavelength or band values.  When
            provided, the time and wavelength columns are stacked to form a
            2-D ``xdata``.  Ignored when ``xcol`` is a list.
        **kwargs
            Additional keyword arguments passed to the Lightcurve constructor.
            If ``band`` is not provided and a string band-ID column exists,
            it is populated automatically.

        Returns
        -------
        Lightcurve
            A 1-D lightcurve when a single time column is used (or when the
            wavelength/band column has only one unique value), or a 2-D
            lightcurve when multiple wavelengths/bands are present.

        Raises
        ------
        ValueError
            If a required column cannot be auto-detected and was not specified
            explicitly, or if an explicitly specified column name is not
            present in the file.
        """
        filepath = Path(filepath)
        # Use dtype=None so that NumPy auto-detects each column's type.
        # This allows string/bytes band columns (e.g. "V", "R") to be read
        # as-is rather than being silently coerced to NaN.
        data = np.genfromtxt(
            filepath, delimiter=",", names=True, dtype=None, encoding=None
        )
        columns = list(data.dtype.names)

        # ------------------------------------------------------------------
        # Helper: is a structured-array column string-typed?
        # ------------------------------------------------------------------
        def _is_str_col(col_name):
            dt = data.dtype[col_name]
            return np.issubdtype(dt, np.str_) or np.issubdtype(dt, np.bytes_)

        # ------------------------------------------------------------------
        # Resolve all column names (no tensor building yet)
        # ------------------------------------------------------------------

        # Resolve x / time column
        if not isinstance(xcol, list):
            if xcol is None:
                xcol = cls._find_column(columns, cls._X_COLUMN_NAMES)
                if xcol is None:
                    raise ValueError(
                        f"Could not auto-detect x column. "
                        f"Available columns: {columns}. "
                        "Please specify xcol explicitly."
                    )
            elif xcol not in columns:
                raise ValueError(
                    f"Column '{xcol}' not found in CSV. "
                    f"Available columns: {columns}"
                )

        # Resolve y column
        if ycol is None:
            ycol = cls._find_column(columns, cls._Y_COLUMN_NAMES)
            if ycol is None:
                raise ValueError(
                    f"Could not auto-detect y column. "
                    f"Available columns: {columns}. "
                    "Please specify ycol explicitly."
                )
        elif ycol not in columns:
            raise ValueError(
                f"Column '{ycol}' not found in CSV. Available columns: {columns}"
            )

        # Resolve yerr column
        if yerrcol is None:
            yerrcol = cls._find_column(columns, cls._YERR_COLUMN_NAMES)
        elif yerrcol not in columns:
            raise ValueError(
                f"Column '{yerrcol}' not found in CSV. Available columns: {columns}"
            )

        # ------------------------------------------------------------------
        # Resolve the x (time + optional numeric wavelength) columns.
        # The string band-ID column is resolved independently below.
        # ------------------------------------------------------------------
        band_id_col = None  # set in the else branch when applicable
        if isinstance(xcol, list):
            # Explicit multi-column x specification
            for col in xcol:
                if col not in columns:
                    raise ValueError(
                        f"Column '{col}' not found in CSV. "
                        f"Available columns: {columns}"
                    )
            # Build NaN mask across all columns before stacking
            xcol_names = xcol
        else:
            # Resolve numeric wavelength column for xdata[:, 1].
            # Only _WAVELENGTH_COLUMN_NAMES is consulted; string band-ID
            # columns are handled separately and independently.
            if wavelcol is not None:
                # Explicit: validate it exists before proceeding.
                if wavelcol not in columns:
                    raise ValueError(
                        f"Column '{wavelcol}' not found in CSV. "
                        f"Available columns: {columns}"
                    )
            else:
                wavelcol = cls._find_column(columns, cls._WAVELENGTH_COLUMN_NAMES)

            # Independently resolve string band-ID column for lc.band.
            # This is always attempted, regardless of whether a numeric
            # wavelength column was found.
            band_id_col = cls._find_column(columns, cls._WAVELENGTH_ID_COLUMN_NAMES)

            xcol_names = [xcol] + ([wavelcol] if wavelcol is not None else [])

        # ------------------------------------------------------------------
        # Build NaN / validity mask from ALL relevant columns.
        # With dtype=None, integer and string columns cannot be NaN, so only
        # check floating-point columns for non-finite values.
        # ------------------------------------------------------------------
        relevant_cols = xcol_names + [ycol] + ([yerrcol] if yerrcol else [])
        valid_mask = np.ones(len(data), dtype=bool)
        for col in relevant_cols:
            col_dtype = data.dtype[col]
            if np.issubdtype(col_dtype, np.floating):
                valid_mask &= ~np.isnan(data[col])
            elif _is_str_col(col):
                # Treat empty strings as missing; convert once outside the
                # per-element comparison.
                valid_mask &= np.asarray(data[col], dtype=np.str_) != ""
            # Integer columns cannot contain NaN; no filtering needed.

        n_dropped = int((~valid_mask).sum())
        if n_dropped > 0:
            warnings.warn(
                f"Dropped {n_dropped} row(s) containing NaN values.",
                stacklevel=2,
            )
        if valid_mask.sum() == 0:
            raise ValueError(
                "No valid data rows remain after dropping NaN-containing rows."
            )

        # Apply mask to get clean data
        clean = data[valid_mask]

        # ------------------------------------------------------------------
        # Helper: convert a structured-array column to a float32 tensor.
        # Boolean-indexing a structured array with dtype=None can produce
        # non-contiguous strides; np.array() (not np.asarray) forces a copy.
        # ------------------------------------------------------------------
        def _to_float_tensor(arr):
            return torch.as_tensor(
                np.array(arr, dtype=np.float64), dtype=torch.float32
            )

        # ------------------------------------------------------------------
        # Helper: map string band labels to float indices and record them.
        # Returns (wave_tensor, unique_labels_array).
        # ------------------------------------------------------------------
        def _str_col_to_wave(arr):
            str_vals = np.asarray(arr, dtype=np.str_)
            # Preserve first-appearance order via dict.fromkeys.
            unique_labels = list(dict.fromkeys(str_vals.tolist()))
            label_to_idx = {lbl: float(i) for i, lbl in enumerate(unique_labels)}
            indices = np.array([label_to_idx[v] for v in str_vals], dtype=np.float64)
            return (
                torch.as_tensor(indices, dtype=torch.float32),
                np.array(unique_labels, dtype=np.str_),
            )

        # ------------------------------------------------------------------
        # Build tensors from clean data
        # ------------------------------------------------------------------
        if isinstance(xcol, list):
            x_tensors = []
            for col in xcol:
                if _is_str_col(col):
                    wave_t, _unique = _str_col_to_wave(clean[col])
                    x_tensors.append(wave_t)
                    if "band" not in kwargs:
                        kwargs["band"] = np.asarray(clean[col], dtype=np.str_)
                else:
                    x_tensors.append(_to_float_tensor(clean[col]))
            x = torch.stack(x_tensors, dim=1) if len(x_tensors) > 1 else x_tensors[0]
        else:
            time_tensor = _to_float_tensor(clean[xcol])
            if wavelcol is not None:
                if _is_str_col(wavelcol):
                    # Explicitly-provided string wavelcol: map labels → indices.
                    wave_tensor, _unique = _str_col_to_wave(clean[wavelcol])
                    if "band" not in kwargs:
                        kwargs["band"] = np.asarray(clean[wavelcol], dtype=np.str_)
                else:
                    wave_tensor = _to_float_tensor(clean[wavelcol])
                if wave_tensor.unique().numel() > 1:
                    # Multiple wavelengths → 2-D lightcurve
                    x = torch.stack([time_tensor, wave_tensor], dim=1)
                else:
                    # Single wavelength → treat as 1-D
                    x = time_tensor
            else:
                x = time_tensor

            # Independently populate band from the string band-ID column.
            if "band" not in kwargs and band_id_col is not None:
                if _is_str_col(band_id_col):
                    band_vals = np.asarray(clean[band_id_col], dtype=np.str_)
                    if x.dim() == 2:
                        kwargs["band"] = band_vals
                    elif band_vals.size > 0:
                        stripped_band_vals = np.char.strip(band_vals)
                        unique_band_labels = [
                            lbl
                            for lbl in dict.fromkeys(stripped_band_vals.tolist())
                            if lbl
                        ]
                        if len(unique_band_labels) == 1:
                            kwargs["band"] = np.array(
                                [unique_band_labels[0]], dtype=np.str_
                            )
                        elif len(unique_band_labels) > 1:
                            _msg = (
                                f"Column '{band_id_col}' contains multiple "
                                "distinct non-empty labels for 1-D input; "
                                "leaving 'band' unset. Provide wavelcol or "
                                "2-D input for mixed-band data."
                            )
                            warnings.warn(_msg, UserWarning, stacklevel=2)

        y = _to_float_tensor(clean[ycol])
        yerr = _to_float_tensor(clean[yerrcol]) if yerrcol else None

        return cls(xdata=x, ydata=y, yerr=yerr, **kwargs)


# Spectral-mixture model names that support MLS-based initialisation in fit().
_SM_MODELS: frozenset[str] = frozenset(
    {
        "2D",
        "1D",
        "1DLinear",
        "2DLinear",
        "2DPowerLaw",
        "2DDust",
        "1DSKI",
        "2DSKI",
        "1DLinearSKI",
        "2DLinearSKI",
        "2DPowerLawSKI",
        "2DDustSKI",
    }
)


@dataclasses.dataclass(frozen=True)
class PeriodPeakResult:
    """A single PSD peak from :meth:`Lightcurve.get_period_summary`."""

    rank: int = 1
    frequency: float = float("nan")
    period: float = float("nan")
    height: float = float("nan")
    prominence: float = float("nan")
    area_fraction: float = float("nan")
    interval_frequency: tuple = (float("nan"), float("nan"))
    interval_period: tuple = (float("nan"), float("nan"))
    period_ratio_to_primary: float = 1.0
    is_candidate_lsp: bool = False
    notes: str = ""
    coherence_proxy: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "frequency": self.frequency,
            "period": self.period,
            "height": self.height,
            "prominence": self.prominence,
            "area_fraction": self.area_fraction,
            "interval_frequency": list(self.interval_frequency),
            "interval_period": list(self.interval_period),
            "period_ratio_to_primary": self.period_ratio_to_primary,
            "is_candidate_lsp": self.is_candidate_lsp,
            "notes": self.notes,
            "coherence_proxy": self.coherence_proxy,
        }


@dataclasses.dataclass
class ACFResult:
    """Result container for :meth:`Lightcurve.acf`.

    Attributes
    ----------
    lag : torch.Tensor
        Lag values (same units as the time axis).
    acf : torch.Tensor
        Autocorrelation values at each lag.
    method : str
        The method used to compute the ACF (``"data"`` or ``"gp"``).
    counts : torch.Tensor or None
        Number of data pairs contributing to each lag bin (data method only).
    normalized : bool
        Whether the ACF has been normalised so that ``acf(0) == 1``.
    band : str, float, or None
        Band label or wavelength used when computing the ACF, if applicable.
    """

    lag: torch.Tensor
    acf: torch.Tensor
    method: str
    counts: torch.Tensor | None = None
    normalized: bool = True
    band: str | float | None = None


class ComponentDiagnosticsResult:
    """Kernel-component diagnostic information for a spectral-mixture GP.

    These values are extracted directly from GP hyperparameters and are
    provided for diagnostic purposes only.  They must **not** be interpreted
    as independent physical periods.  The literature-comparable period
    estimates are the summed-PSD peaks stored in
    :attr:`PeriodSummaryResult.peaks`.

    Attributes
    ----------
    component_periods : numpy.ndarray
        Centre period of each mixture component (1/frequency).
    component_frequencies : numpy.ndarray
        Centre frequency of each mixture component.
    component_weights : numpy.ndarray
        Relative amplitude weight of each mixture component.
    component_period_scales : numpy.ndarray
        Width (sigma) of each Gaussian component in period units.
    component_frequency_scales : numpy.ndarray
        Width (sigma) of each Gaussian component in frequency units.
    n_components : int
        Number of mixture components.
    kernel_family : str
        Name of the spectral-mixture kernel family.
    notes : str
        Diagnostic notes for this component set.
    component_labels : list of str
        Human-readable label for each component,
        e.g. ``["SM component 1", "SM component 2"]``.
    """

    def __init__(
        self,
        component_periods=None,
        component_frequencies=None,
        component_weights=None,
        component_period_scales=None,
        component_frequency_scales=None,
        n_components=0,
        kernel_family="",
        notes="",
        component_labels=None,
    ):
        self.component_periods = (
            component_periods
            if component_periods is not None
            else np.array([])
        )
        self.component_frequencies = (
            component_frequencies
            if component_frequencies is not None
            else np.array([])
        )
        self.component_weights = (
            component_weights
            if component_weights is not None
            else np.array([])
        )
        self.component_period_scales = (
            component_period_scales
            if component_period_scales is not None
            else np.array([])
        )
        self.component_frequency_scales = (
            component_frequency_scales
            if component_frequency_scales is not None
            else np.array([])
        )
        self.n_components = n_components
        self.kernel_family = kernel_family
        self.notes = notes
        self.component_labels = component_labels or [
            f"SM component {i + 1}" for i in range(n_components)
        ]

    def as_dict(self) -> dict:
        """Return a plain-dict representation of this diagnostics object."""
        return {
            "n_components": self.n_components,
            "kernel_family": self.kernel_family,
            "notes": self.notes,
            "component_labels": self.component_labels,
            "component_periods": self.component_periods,
            "component_frequencies": self.component_frequencies,
            "component_weights": self.component_weights,
            "component_period_scales": self.component_period_scales,
            "component_frequency_scales": self.component_frequency_scales,
        }


class PeriodSummaryResult:
    """Structured result from :meth:`Lightcurve.get_period_summary`."""

    def __init__(
        self,
        method="",
        model_name="",
        n_peaks_detected=0,
        n_peaks_analyzed=0,
        n_peaks_requested=None,
        dominant_period=None,
        dominant_frequency=None,
        peaks=None,
        freq_grid=None,
        psd=None,
        notes="",
        component_diagnostics=None,
        interval_definition="peak_centered_68pct_mass_interval",
        backend="",
        kernel_family="",
        time_kernel_family="",
        has_stochastic_background=False,
        q_factor=None,
        is_multicomponent=False,
        component_periods=None,
        component_period_widths=None,
        component_fitted_periods=None,
        component_initialized_periods=None,
        component_strengths=None,
        component_source_cluster_ids=None,
        component_member_bands=None,
        component_summaries=None,
        drift_warning_fraction=None,
    ):
        self.method = method
        self.model_name = model_name
        self.backend = backend
        self.kernel_family = kernel_family
        self.time_kernel_family = time_kernel_family
        self.has_stochastic_background = has_stochastic_background
        self.is_multicomponent = bool(is_multicomponent)
        self.component_periods = (
            np.asarray(component_periods, dtype=float).ravel().tolist()
            if component_periods is not None
            else []
        )
        self.component_period_widths = (
            np.asarray(component_period_widths, dtype=float).ravel().tolist()
            if component_period_widths is not None
            else []
        )
        self.component_fitted_periods = (
            np.asarray(component_fitted_periods, dtype=float).ravel().tolist()
            if component_fitted_periods is not None
            else []
        )
        self.component_initialized_periods = (
            np.asarray(component_initialized_periods, dtype=float).ravel().tolist()
            if component_initialized_periods is not None
            else []
        )
        self.component_strengths = (
            np.asarray(component_strengths, dtype=float).ravel().tolist()
            if component_strengths is not None
            else []
        )
        self.component_source_cluster_ids = (
            list(component_source_cluster_ids)
            if component_source_cluster_ids is not None
            else []
        )
        self.component_member_bands = (
            [list(bands or []) for bands in component_member_bands]
            if component_member_bands is not None
            else []
        )
        self.component_summaries = (
            [dict(entry) for entry in component_summaries if isinstance(entry, dict)]
            if component_summaries is not None
            else []
        )
        self.drift_warning_fraction = drift_warning_fraction
        if self.is_multicomponent and not self.component_summaries:
            n_components = len(self.component_periods)
            for idx in range(n_components):
                self.component_summaries.append(
                    {
                        "component_index": idx,
                        "consensus_period": (
                            self.component_periods[idx]
                            if idx < len(self.component_periods)
                            else None
                        ),
                        "consensus_period_width": (
                            self.component_period_widths[idx]
                            if idx < len(self.component_period_widths)
                            else None
                        ),
                        "fitted_mixture_period": (
                            self.component_fitted_periods[idx]
                            if idx < len(self.component_fitted_periods)
                            else None
                        ),
                        "initialized_mixture_period": (
                            self.component_initialized_periods[idx]
                            if idx < len(self.component_initialized_periods)
                            else None
                        ),
                        "consensus_component_strength": (
                            self.component_strengths[idx]
                            if idx < len(self.component_strengths)
                            else None
                        ),
                        "source_cluster_id": (
                            self.component_source_cluster_ids[idx]
                            if idx < len(self.component_source_cluster_ids)
                            else None
                        ),
                        "member_bands": (
                            self.component_member_bands[idx]
                            if idx < len(self.component_member_bands)
                            else []
                        ),
                    }
                )
        self.n_peaks_detected = n_peaks_detected
        self.n_peaks_analyzed = n_peaks_analyzed
        self.n_peaks_requested = n_peaks_requested
        self.dominant_period = dominant_period
        self.dominant_frequency = dominant_frequency
        self.q_factor = q_factor
        # Sort peaks by physical ranking so that peaks[0] is the primary
        # pulsation candidate, not the largest-area feature.
        #
        # Sort key (ascending tuple, NaN treated as worst):
        #   1. descending prominence  — most distinct peak first
        #   2. descending coherence_proxy — narrower/more coherent peak first
        #   3. descending area_fraction  — larger integrated power next
        #   4. descending height         — absolute amplitude tie-breaker
        #   5. ascending original rank   — deterministic final tie-breaker
        #
        # After sorting, ranks are reassigned sequentially (1, 2, 3 …) so
        # that peak.rank reliably reflects position in the sorted list.
        _raw_peaks = peaks if peaks is not None else []

        def _physical_rank_key(p):
            prom = (
                p.prominence if np.isfinite(p.prominence) else -np.inf
            )
            coh = (
                p.coherence_proxy
                if np.isfinite(p.coherence_proxy)
                else -np.inf
            )
            af = p.area_fraction if np.isfinite(p.area_fraction) else -np.inf
            h = p.height if np.isfinite(p.height) else -np.inf
            return (-prom, -coh, -af, -h, p.rank)

        _sorted = sorted(_raw_peaks, key=_physical_rank_key)
        # Reassign ranks sequentially and update period_ratio_to_primary so
        # that the new rank-1 peak always has ratio=1.0 and the other peaks
        # are relative to it.
        _primary_period = _sorted[0].period if _sorted else 1.0
        self.peaks = [
            dataclasses.replace(
                p,
                rank=i + 1,
                period_ratio_to_primary=(
                    p.period / _primary_period
                    if _primary_period > 0 and np.isfinite(p.period)
                    else float("nan")
                ),
            )
            for i, p in enumerate(_sorted)
        ]
        # Track which peak in the sorted list carries the largest area_fraction.
        # This is the "largest integrated-power feature", which may differ from
        # the primary pulsation candidate (peaks[0]).
        if self.peaks:
            self.largest_area_peak_index = max(
                range(len(self.peaks)),
                key=lambda i: (
                    self.peaks[i].area_fraction
                    if np.isfinite(self.peaks[i].area_fraction)
                    else -np.inf
                ),
            )
            self.primary_peak_index = 0
        else:
            self.largest_area_peak_index = None
            self.primary_peak_index = None
        # Keep dominant_period / dominant_frequency in sync with the
        # post-sort primary peak so that direct attribute access is also
        # consistent (not just as_dict() which already prefers peaks[0]).
        if self.peaks:
            _primary = self.peaks[0]
            self.dominant_period = _primary.period
            self.dominant_frequency = _primary.frequency
            # Compute q_factor from the post-sort primary peak's
            # interval_frequency so that summary.q_factor is always correct
            # at the object level, not just inside as_dict().
            # Formula: q_factor = frequency / (f_hi - f_lo)
            _f_lo, _f_hi = _primary.interval_frequency
            _width = _f_hi - _f_lo
            if (
                np.isfinite(_width)
                and _width > 0
                and np.isfinite(_primary.frequency)
            ):
                self.q_factor = _primary.frequency / _width
            else:
                # Invalid interval (e.g. explicit-period backend with no RBF
                # lengthscale): set to None so that self.q_factor always
                # describes the post-sort primary peak, never a stale
                # upstream value.
                self.q_factor = None
        else:
            # No peaks — pass the constructor-provided value through
            # unchanged, since there is no primary peak to override it.
            self.q_factor = q_factor
        # Validate internal consistency: dominant attributes must agree with
        # peaks[0] whenever peaks exist.
        if self.peaks:
            assert self.dominant_period == self.peaks[0].period, (
                f"PeriodSummaryResult internal error: dominant_period "
                f"({self.dominant_period!r}) != peaks[0].period "
                f"({self.peaks[0].period!r})"
            )
            assert self.dominant_frequency == self.peaks[0].frequency, (
                f"PeriodSummaryResult internal error: dominant_frequency "
                f"({self.dominant_frequency!r}) != peaks[0].frequency "
                f"({self.peaks[0].frequency!r})"
            )
        self.freq_grid = freq_grid
        self.psd = psd
        self.notes = notes
        self.interval_definition = interval_definition
        self.component_diagnostics = component_diagnostics

    def as_dict(self) -> dict:
        # Derive primary-peak quantities once so they can be reused for
        # the backward-compatible alias keys without repeating the logic.
        primary = self.get_primary_peak()
        primary_interval = primary.interval_period if primary is not None else None
        primary_area = (
            primary.area_fraction if primary is not None else float("nan")
        )
        _sig_peaks = self.get_significant_peaks()
        significant_periods = np.array([p.period for p in _sig_peaks])

        # Largest-area-fraction peak (may differ from the primary).
        _la_peak = (
            self.peaks[self.largest_area_peak_index]
            if self.peaks
            else None
        )
        _la_rank = _la_peak.rank if _la_peak is not None else None
        _la_period = _la_peak.period if _la_peak is not None else float("nan")
        _la_freq = (
            _la_peak.frequency if _la_peak is not None else float("nan")
        )
        _la_frac = (
            _la_peak.area_fraction
            if _la_peak is not None
            else float("nan")
        )

        payload = {
            "component_diagnostics": (
                self.component_diagnostics.as_dict()
                if self.component_diagnostics is not None
                else None
            ),
            "freq_grid": self.freq_grid,
            "psd": self.psd,
            # self.dominant_frequency/dominant_period/q_factor are set in
            # __init__() from peaks[0] and are already authoritative.
            "dominant_frequency": self.dominant_frequency,
            "dominant_period": self.dominant_period,
            # Backward-compatible interval keys (both alias the same value).
            "period_interval_fwhm_like": primary_interval,
            "period_interval": primary_interval,
            "interval_definition": self.interval_definition,
            "q_factor": self.q_factor,
            "peak_fraction": primary_area,
            "n_peaks": len(self.peaks),
            "n_peaks_detected": self.n_peaks_detected,
            "n_significant_peaks": len(_sig_peaks),
            "significant_periods": significant_periods,
            "peaks": [p.as_dict() for p in self.peaks],
            "method": self.method,
            "notes": self.notes,
            # Kernel-dispatch metadata
            "backend": self.backend,
            "model_name": self.model_name,
            "kernel_family": self.kernel_family,
            "time_kernel_family": self.time_kernel_family,
            "has_stochastic_background": self.has_stochastic_background,
            # Physical-ranking indices
            "primary_peak_rank": primary.rank if primary is not None else None,
            "largest_area_peak_rank": _la_rank,
            # Largest-area-fraction feature (diagnostic)
            "largest_area_period": _la_period,
            "largest_area_frequency": _la_freq,
            "largest_area_fraction": _la_frac,
        }
        if self.is_multicomponent:
            payload.update(
                {
                    "is_multicomponent": self.is_multicomponent,
                    "component_periods": self.component_periods,
                    "component_period_widths": self.component_period_widths,
                    "component_fitted_periods": self.component_fitted_periods,
                    "component_initialized_periods": (
                        self.component_initialized_periods
                    ),
                    "component_strengths": self.component_strengths,
                    "component_source_cluster_ids": (
                        self.component_source_cluster_ids
                    ),
                    "component_member_bands": self.component_member_bands,
                    "component_summaries": self.component_summaries,
                    "drift_warning_fraction": self.drift_warning_fraction,
                }
            )
        return payload

    def __getitem__(self, key):
        return self.as_dict()[key]

    def __contains__(self, key):
        return key in self.as_dict()

    def get(self, key, default=None):
        return self.as_dict().get(key, default)

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()

    def values(self):
        return self.as_dict().values()

    # ------------------------------------------------------------------
    # Multi-peak accessors
    # ------------------------------------------------------------------

    def get_primary_peak(self):
        """Return the primary (rank-1) peak, or ``None`` if none exist.

        Returns
        -------
        PeriodPeakResult or None
            The first entry in :attr:`peaks` (sorted by ascending rank,
            so rank 1 is always first), or ``None`` when :attr:`peaks`
            is empty.
        """
        return self.peaks[0] if self.peaks else None

    def get_top_n_peaks(self, n):
        """Return up to *n* peaks in ascending rank order.

        Parameters
        ----------
        n : int
            Maximum number of peaks to return.

        Returns
        -------
        list of PeriodPeakResult
            A slice of :attr:`peaks` of length ``min(n, len(peaks))``.
        """
        return self.peaks[:n]

    def get_significant_peaks(self, threshold=0.68):
        """Return peaks whose area fraction meets or exceeds *threshold*.

        Parameters
        ----------
        threshold : float, optional
            Minimum ``area_fraction`` to qualify as significant.
            Default is ``0.68`` (~1 sigma).

        Returns
        -------
        list of PeriodPeakResult
            Peaks from :attr:`peaks` (in rank order) for which
            ``peak.area_fraction >= threshold``.  Peaks with NaN area
            fraction are excluded.
        """
        return [
            p for p in self.peaks
            if np.isfinite(p.area_fraction) and p.area_fraction >= threshold
        ]

    def to_table(self) -> list:
        return [
            {
                "peak_rank": p.rank,
                "period": p.period,
                "frequency": p.frequency,
                "height": p.height,
                "prominence": p.prominence,
                "area_fraction": p.area_fraction,
                "period_interval_lo": p.interval_period[0],
                "period_interval_hi": p.interval_period[1],
                "period_ratio_to_primary": p.period_ratio_to_primary,
                "is_candidate_lsp": p.is_candidate_lsp,
                "notes": p.notes,
            }
            for p in self.peaks
        ]

    def to_text(
        self,
        include_components=True,
        include_peaks=True,
        include_psd_info=False,
        max_peaks_to_show=3,
    ) -> str:
        """Return a human-readable text summary of this period result.

        The text is plain UTF-8 text, suitable for writing to a ``.txt``
        file, reading in a terminal, or storing alongside analysis outputs.
        It clearly separates **analyzed peak results** (the
        literature-comparable outputs) from **kernel component diagnostics**
        (internal quantities derived directly from GP hyperparameters).

        Parameters
        ----------
        include_components : bool, optional
            If ``True`` (default), include a section listing the kernel
            component periods, frequencies, and weights.  These are
            **diagnostic quantities** and should not be cited as final
            period determinations.
        include_peaks : bool, optional
            If ``True`` (default), include one block per analyzed peak.
        include_psd_info : bool, optional
            If ``True``, include a short summary of the PSD grid
            (existence, length, frequency range, PSD range).  The full
            arrays are never dumped.  Default is ``False``.
        max_peaks_to_show : int, optional
            Maximum number of peaks to show in detail.  The primary peak
            is always shown first; up to ``max_peaks_to_show - 1``
            additional peaks follow.  If more peaks exist, a count line
            is appended.  Default is ``3``.

        Returns
        -------
        str
            Formatted text summary.
        """

        def _fmt(v, precision=6):
            """Format a scalar value for display."""
            if v is None:
                return "N/A"
            try:
                if np.isnan(v):
                    return "nan"
                if np.isinf(v):
                    return "inf" if v > 0 else "-inf"
            except (TypeError, ValueError):
                pass
            try:
                return f"{v:.{precision}g}"
            except (TypeError, ValueError):
                return str(v)

        def _fmt_interval(pair, precision=6):
            """Format a (lo, hi) interval pair."""
            if pair is None:
                return "N/A"
            lo, hi = pair
            return f"[{_fmt(lo, precision)}, {_fmt(hi, precision)}]"

        def _arr_summary(arr, label, precision=6):
            """One-line summary of a 1-D array."""
            if arr is None or len(arr) == 0:
                return f"  {label}: (none)"
            vals = ", ".join(_fmt(v, precision) for v in arr)
            return f"  {label}: {vals}"

        lines = []

        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        # Use the same dominant-period/frequency logic as as_dict(): prefer
        # the primary peak's values so that to_text() and as_dict() always
        # describe the same dominant peak.
        _primary = self.get_primary_peak()
        _display_period = (
            _primary.period if _primary is not None else self.dominant_period
        )
        _display_frequency = (
            _primary.frequency
            if _primary is not None
            else self.dominant_frequency
        )

        lines.append("PERIOD SUMMARY")
        lines.append("==============")
        lines.append(f"  Model name          : {self.model_name or 'N/A'}")
        lines.append(f"  Method              : {self.method or 'N/A'}")
        lines.append(f"  Backend             : {self.backend or 'N/A'}")
        lines.append(
            f"  Kernel family       : {self.kernel_family or 'N/A'}"
        )
        _tkf = self.time_kernel_family or "N/A"
        lines.append(f"  Time-kernel family  : {_tkf}")
        _hsb = str(self.has_stochastic_background)
        lines.append(f"  Stochastic bg       : {_hsb}")
        lines.append(
            f"  Interval definition : {self.interval_definition or 'N/A'}"
        )
        lines.append(f"  Dominant period     : {_fmt(_display_period)}")
        lines.append(
            f"  Dominant frequency  : {_fmt(_display_frequency)}"
        )
        lines.append(f"  Peaks detected      : {self.n_peaks_detected}")
        lines.append(f"  Peaks analyzed      : {self.n_peaks_analyzed}")
        _req = (
            str(self.n_peaks_requested)
            if self.n_peaks_requested is not None
            else "N/A"
        )
        lines.append(f"  Peaks requested     : {_req}")
        if self.notes:
            lines.append(f"  Notes               : {self.notes}")
        lines.append("")

        if self.is_multicomponent:
            drift_threshold = self.drift_warning_fraction
            if drift_threshold is None:
                drift_threshold = _CONSENSUS_MULTICOMP_DRIFT_WARNING_FRACTION
            large_period_drift_count = 0
            possible_identity_swap_count = 0
            lines.append("MULTI-COMPONENT PERIOD SUMMARY")
            lines.append("==============================")
            for idx, component in enumerate(self.component_summaries):
                if not isinstance(component, dict):
                    continue
                component_idx = component.get("component_index", idx)
                member_bands = list(component.get("member_bands") or [])
                lines.append(f"  Component {component_idx}")
                lines.append(
                    f"      Consensus period: "
                    f"{_fmt(component.get('consensus_period'))}"
                )
                lines.append(
                    f"      Fitted period: "
                    f"{_fmt(component.get('fitted_mixture_period'))}"
                )
                lines.append(
                    f"      Initialized period: "
                    f"{_fmt(component.get('initialized_mixture_period'))}"
                )
                lines.append(
                    f"      Period width: "
                    f"{_fmt(component.get('consensus_period_width'))}"
                )
                lines.append(
                    f"      Supported by: "
                    f"{', '.join(member_bands) if member_bands else 'N/A'}"
                )
                lines.append(
                    f"      Strength: "
                    f"{_fmt(component.get('consensus_component_strength'))}"
                )
                lines.append(
                    f"      Consensus frequency: "
                    f"{_fmt(component.get('consensus_frequency'))}"
                )
                fractional_period_shift = component.get(
                    "fitted_fractional_period_shift_from_initialization"
                )
                drift_flag = bool(component.get("fitted_period_drift_flag"))
                drift_line = "      Drift from initialization: N/A"
                if fractional_period_shift is not None:
                    try:
                        fractional_period_shift = float(fractional_period_shift)
                    except (TypeError, ValueError):
                        fractional_period_shift = None
                if (
                    fractional_period_shift is not None
                    and np.isfinite(fractional_period_shift)
                ):
                    drift_line = (
                        "      Drift from initialization: "
                        f"{fractional_period_shift * 100.0:+.3g}%"
                    )
                    if drift_flag:
                        drift_line += "  [warning: large drift]"
                lines.append(drift_line)
                if drift_flag:
                    large_period_drift_count += 1
                identity_preserved = component.get("component_identity_preserved")
                if isinstance(identity_preserved, (bool, np.bool_)):
                    if bool(identity_preserved):
                        lines.append("      Identity preserved: yes")
                    else:
                        nearest_idx = component.get(
                            "nearest_initialized_component_index"
                        )
                        lines.append(
                            "      Identity preserved: no; nearest initialized "
                            f"component: {_fmt(nearest_idx)}"
                        )
                        possible_identity_swap_count += 1
                else:
                    lines.append("      Identity preserved: N/A")
                lines.append("")
            if large_period_drift_count > 0:
                lines.append(
                    "Warning: "
                    f"{large_period_drift_count} "
                    f"{'component' if large_period_drift_count == 1 else 'components'} "
                    "shifted by "
                    f"more than {drift_threshold * 100.0:g}% from consensus "
                    "initialization."
                )
                lines.append("")
            if possible_identity_swap_count > 0:
                lines.append(
                    "Warning: fitted component identity may have changed during "
                    "optimization."
                )
                lines.append("")

        # ------------------------------------------------------------------
        # Analyzed peaks (literature-comparable outputs)
        # ------------------------------------------------------------------
        if include_peaks and self.peaks:
            primary = self.peaks[0]

            # ---- Primary pulsation candidate (full detail) ---------------
            lines.append(
                "PRIMARY PEAK  (primary pulsation candidate)"
            )
            lines.append("=" * 44)
            lines.append(
                f"    Period                     : {_fmt(primary.period)}"
            )
            lines.append(
                f"    Frequency                  : {_fmt(primary.frequency)}"
            )
            lines.append(
                f"    Height                     : {_fmt(primary.height)}"
            )
            lines.append(
                f"    Prominence                 : {_fmt(primary.prominence)}"
            )
            lines.append(
                f"    Coherence proxy            : "
                f"{_fmt(primary.coherence_proxy)}"
            )
            lines.append(
                f"    Area fraction              : "
                f"{_fmt(primary.area_fraction)}"
            )
            lines.append(
                f"    Interval (frequency)       : "
                f"{_fmt_interval(primary.interval_frequency)}"
            )
            lines.append(
                f"    Interval (period)          : "
                f"{_fmt_interval(primary.interval_period)}"
            )
            lines.append(
                f"    LSP candidate              : {primary.is_candidate_lsp}"
            )
            if primary.notes:
                lines.append(
                    f"    Notes                      : {primary.notes}"
                )
            lines.append("")

            # ---- Largest integrated-power feature (when different) -------
            _la_idx = self.largest_area_peak_index
            if _la_idx != 0 and _la_idx < len(self.peaks):
                la_peak = self.peaks[_la_idx]
                lines.append(
                    "LARGEST INTEGRATED-POWER FEATURE  "
                    "(diagnostic — differs from primary)"
                )
                lines.append("=" * 51)
                lines.append(
                    f"    Rank                       : {la_peak.rank}"
                )
                lines.append(
                    f"    Period                     : "
                    f"{_fmt(la_peak.period)}"
                )
                lines.append(
                    f"    Frequency                  : "
                    f"{_fmt(la_peak.frequency)}"
                )
                lines.append(
                    f"    Area fraction              : "
                    f"{_fmt(la_peak.area_fraction)}"
                )
                lines.append(
                    f"    Prominence                 : "
                    f"{_fmt(la_peak.prominence)}"
                )
                lines.append(
                    f"    Coherence proxy            : "
                    f"{_fmt(la_peak.coherence_proxy)}"
                )
                lines.append("")
            elif self.peaks:
                # Primary peak is also the largest-area feature
                lines.append(
                    "  (Primary peak also has the largest area fraction.)"
                )
                lines.append("")

            # ---- Additional peaks (compact) ------------------------------
            extra_peaks = self.peaks[1:]
            if extra_peaks:
                n_to_show = max(0, max_peaks_to_show - 1)
                shown = extra_peaks[:n_to_show]
                n_hidden = len(extra_peaks) - len(shown)

                if shown:
                    lines.append("ADDITIONAL PEAKS")
                    lines.append("=" * 16)
                    for pk in shown:
                        _int_str = _fmt_interval(pk.interval_period)
                        _la_tag = (
                            " [largest-area]"
                            if pk.rank - 1 == _la_idx
                            else ""
                        )
                        lines.append(
                            f"  #{pk.rank}  period={_fmt(pk.period)}"
                            f"  freq={_fmt(pk.frequency)}"
                            f"  area={_fmt(pk.area_fraction)}"
                            f"  prom={_fmt(pk.prominence)}"
                            f"  interval={_int_str}"
                            f"{_la_tag}"
                        )
                    if n_hidden > 0:
                        lines.append(
                            f"  (+{n_hidden} additional peak"
                            f"{'s' if n_hidden != 1 else ''} not shown)"
                        )
                    lines.append("")

        # ------------------------------------------------------------------
        # Kernel component diagnostics (NOT final periods)
        # ------------------------------------------------------------------
        if include_components and self.component_diagnostics is not None:
            diag = self.component_diagnostics
            lines.append(
                "KERNEL COMPONENT DIAGNOSTICS  "
                "(internal quantities -- not final periods)"
            )
            lines.append("=" * 60)
            lines.append(
                "  These values are derived directly from GP kernel"
                " hyperparameters."
            )
            lines.append(
                "  They are provided for diagnostics only and should not"
                " be cited"
            )
            lines.append(
                "  as literature-comparable period determinations."
            )
            lines.append("")
            lines.append(
                _arr_summary(diag.component_periods, "Component periods")
            )
            lines.append(
                _arr_summary(
                    diag.component_frequencies,
                    "Component frequencies",
                )
            )
            lines.append(
                _arr_summary(diag.component_weights, "Component weights")
            )
            lines.append(
                _arr_summary(
                    diag.component_period_scales,
                    "Component period scales",
                )
            )
            lines.append(
                _arr_summary(
                    diag.component_frequency_scales,
                    "Component frequency scales",
                )
            )
            lines.append("")

        # ------------------------------------------------------------------
        # Optional PSD grid summary (never dumps full arrays)
        # ------------------------------------------------------------------
        if include_psd_info:
            lines.append("PSD GRID INFORMATION")
            lines.append("====================")
            has_freq = self.freq_grid is not None
            has_psd = self.psd is not None
            lines.append(
                f"  Frequency grid present : {has_freq}"
            )
            lines.append(f"  PSD array present      : {has_psd}")
            if has_freq:
                try:
                    lines.append(
                        f"  Grid length            : {len(self.freq_grid)}"
                    )
                    lines.append(
                        f"  Frequency min          : "
                        f"{_fmt(float(self.freq_grid[0]))}"
                    )
                    lines.append(
                        f"  Frequency max          : "
                        f"{_fmt(float(self.freq_grid[-1]))}"
                    )
                except Exception:
                    pass
            if has_psd:
                try:
                    _psd_min = float(np.min(self.psd))
                    _psd_max = float(np.max(self.psd))
                    lines.append(f"  PSD min                : {_fmt(_psd_min)}")
                    lines.append(f"  PSD max                : {_fmt(_psd_max)}")
                except Exception:
                    pass
            lines.append("")

        return "\n".join(lines)

    def write_text(
        self,
        filename,
        include_components=True,
        include_peaks=True,
        include_psd_info=False,
    ):
        """Write a human-readable text summary to *filename*.

        Calls :meth:`to_text` and writes the result to disk.

        Parameters
        ----------
        filename : str or Path-like
            Destination file path.  The file is created or overwritten.
        include_components : bool, optional
            Forwarded to :meth:`to_text`.  Default is ``True``.
        include_peaks : bool, optional
            Forwarded to :meth:`to_text`.  Default is ``True``.
        include_psd_info : bool, optional
            Forwarded to :meth:`to_text`.  Default is ``False``.

        Returns
        -------
        pathlib.Path
            The path to the file that was written, constructed from
            *filename* via :class:`pathlib.Path`.  If *filename* is a
            relative path, the returned value is also relative.
        """
        from pathlib import Path

        path = Path(filename)
        text = self.to_text(
            include_components=include_components,
            include_peaks=include_peaks,
            include_psd_info=include_psd_info,
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def _json_serialize(self, obj):
        """Recursively convert *obj* to a JSON-serializable Python object.

        Handles nested dicts, lists/tuples, numpy arrays and scalars, and
        the standard JSON primitives.  Raises ``TypeError`` for any
        unrecognised type so that serialization bugs are caught immediately
        rather than silently corrupted via ``str()``.
        """
        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, int):
            return obj
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        if isinstance(obj, dict):
            return {k: self._json_serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_serialize(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return self._json_serialize(obj.tolist())
        if isinstance(obj, np.floating):
            scalar = obj.item()
            return None if not math.isfinite(scalar) else scalar
        if isinstance(obj, np.integer):
            return obj.item()
        raise TypeError(
            f"Cannot JSON-serialize object of type {type(obj).__name__}"
        )

    def write_json(
        self,
        filename,
        include_psd=False,
        include_fit_history=False,
        fit_history=None,
    ):
        """Write a JSON period summary with optional PSD and fit provenance."""
        d = self.as_dict()
        # Handle freq_grid/psd before general serialization: omit them
        # unless the caller explicitly requests PSD data.
        if not include_psd or d.get("freq_grid") is None:
            d = {**d, "freq_grid": None, "psd": None}
        if include_fit_history:
            d["fit_history"] = Lightcurve._sanitize_fit_history_value(
                [] if fit_history is None else fit_history
            )
        data = self._json_serialize(d)
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, allow_nan=False)


class FitFailureSummary:
    """Lightweight structured summary for failed fit attempts."""

    def __init__(self, status="failed", reason=None, message="", diagnostics=None):
        self.status = status
        self.reason = reason
        self.message = message
        self.diagnostics = dict(diagnostics or {})

    def _json_serialize(self, obj):
        if obj is None or isinstance(obj, (bool, str, int)):
            return obj
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        if isinstance(obj, dict):
            return {str(k): self._json_serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_serialize(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return self._json_serialize(obj.tolist())
        if isinstance(obj, np.floating):
            scalar = obj.item()
            return None if not math.isfinite(scalar) else scalar
        if isinstance(obj, np.integer):
            return obj.item()
        raise TypeError(
            f"Cannot JSON-serialize object of type {type(obj).__name__}"
        )

    def to_dict(self, include_fit_history=False, fit_history=None):
        """Return a JSON-safe failure-summary dict with optional fit history."""
        payload = {
            "status": self.status,
            "reason": self.reason,
            "message": self.message,
            "diagnostics": self.diagnostics,
        }
        if include_fit_history:
            payload["fit_history"] = Lightcurve._sanitize_fit_history_value(
                [] if fit_history is None else fit_history
            )
        return self._json_serialize(payload)

    def to_text(self):
        lines = [
            "FIT FAILURE SUMMARY",
            "===================",
            f"Status : {self.status}",
            f"Reason : {self.reason or 'N/A'}",
            f"Message: {self.message or 'N/A'}",
        ]
        if self.diagnostics:
            lines.append("Diagnostics:")
            for key in sorted(self.diagnostics):
                lines.append(f"  - {key}: {self.to_dict()['diagnostics'].get(key)}")
        return "\n".join(lines)

    def write_json(
        self,
        filename,
        include_fit_history=False,
        fit_history=None,
    ):
        """Write failure summary JSON with optional fit-history provenance."""
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(
                self.to_dict(
                    include_fit_history=include_fit_history,
                    fit_history=fit_history,
                ),
                fh,
                indent=2,
                allow_nan=False,
            )

class Lightcurve(InputHelpers, gpytorch.Module):
    """A class for storing, manipulating and fitting light curves

    This class is designed to be a convenient way to store and manipulate
    light curve data, and to fit Gaussian Processes to that data. It is
    designed to be used with the GPyTorch library, and in future will be
    compatible with the Pyro library for MCMC fitting.

    Parameters
    ----------
    xdata : Tensor of floats
        The independent variable data
    ydata : Tensor of floats
        The dependent variable data
    yerr : Tensor of floats, optional
        The uncertainties on the dependent variable data, by default None
    xtransform : str, optional
        The transform to apply to the x data, by default None
    ytransform : str, optional
        The transform to apply to the y data, by default None
    time_units : str, astropy.units.UnitBase, or None, optional
        Units of the time axis.  Time values are converted to days
        internally.  If *None* (default) the data are assumed to already
        be in days.
    band : array-like of str or None, optional
        Optional per-row band labels for a 2-D light curve.  Each element
        must be a string identifier (e.g. ``"V"``, ``"R"``, ``"W1"``), and
        there must be exactly one label per observation row — i.e.
        ``len(band) == len(xdata)`` for 2-D data.  For 1-D light curves a
        single-element array ``["V"]`` is accepted.  ``None`` (default)
        means no band labels are stored.

    Attributes
    ----------
    band : numpy.ndarray of str or None
        Per-row string labels aligned with ``xdata``, or ``None``
        if no labels were provided.


    Examples
    --------


    Notes
    -----
    """

    def __init__(
        self,
        xdata,
        ydata,
        yerr=None,
        xtransform=None,
        ytransform=None,
        name=None,
        time_units=None,
        max_samples: int | None = 1000,
        max_samples_per_band: int | None = None,
        subsample_seed: int | None = None,
        check_sampling: bool = False,
        sampling_kwargs: dict | None = None,
        check_variability: bool = False,
        variability_kwargs: dict | None = None,
        band=None,
        **kwargs,
    ):
        """Initialize a Lightcurve.

        Parameters
        ----------
        xdata : torch.Tensor
            The independent variable data (time, or time + wavelength for 2-D
            light curves).
        ydata : torch.Tensor
            The dependent variable data
        yerr : torch.Tensor, optional
            The uncertainties on the dependent variable data, by default None
        xtransform : str or Transformer, optional
            The transform to apply to the x data, by default None
        ytransform : str or Transformer, optional
            The transform to apply to the y data, by default None
        name : str, optional
            A name for this light curve, by default 'Lightcurve'
        time_units : str, astropy.units.UnitBase, or None, optional
            Units of the time axis in *xdata*.  The time values will be
            converted to days internally.  Accepts any string recognised by
            ``astropy.units`` (e.g. ``'s'``, ``'hr'``, ``'yr'``, ``'days'``)
            or an ``astropy.units`` unit object.  If *None* (default) the
            data are assumed to already be in days and no conversion is
            performed.
        max_samples_per_band : int or None, optional
            Maximum number of observations to retain per band for 2-D
            (multiband) lightcurves.  Each band is checked independently:
            only bands that exceed `max_samples_per_band` are subsampled;
            bands already at or below the limit are left untouched.  For
            1-D lightcurves this parameter has no effect.  Set to ``None``
            (default) to disable per-band subsampling entirely.  A
            :class:`UserWarning` is issued whenever subsampling occurs
            (see :func:`~pgmuvi.preprocess.subsample_lightcurve`).
        max_samples : int or None, optional
            Maximum number of observations to retain.  For 1-D lightcurves,
            when the total number of points exceeds `max_samples`, a
            gap-preserving random subsample of `max_samples` points is
            drawn and stored permanently.  For 2-D lightcurves, this
            parameter does **not** trigger subsampling; use
            `max_samples_per_band` for that.  Instead, a
            :class:`UserWarning` is issued if the total point count exceeds
            `max_samples`, as a compute-budget advisory.  Default is
            ``1000``.  Set to ``None`` to disable all automatic subsampling
            (1-D) or to suppress the advisory warning (2-D).
        subsample_seed : int or None, optional
            Random seed for the subsampler.  Provide an integer for
            reproducible results; ``None`` (default) gives a non-deterministic
            subsample.  Only used when *max_samples* is set.
        check_sampling : bool, optional
            If ``True``, assess temporal sampling quality after storing the
            data.  For 1-D lightcurves a :class:`ValueError` is raised if
            sampling is poor.  For 2-D (multiband) lightcurves each band is
            checked independently: bands that fail are removed from the stored
            data with a :class:`UserWarning`, and a :class:`ValueError` is
            raised only if no bands pass.  Default is ``False``.
        sampling_kwargs : dict or None, optional
            Keyword arguments forwarded to the sampling quality gates
            (``min_points``, ``max_gap_fraction``, ``min_baseline_factor``,
            ``min_snr``, ``min_fraction_good_snr``).  Only used when
            *check_sampling* is ``True``.
        check_variability : bool, optional
            If ``True``, verify that the lightcurve shows significant
            variability after storing the data.  Raises :class:`ValueError`
            if not variable.  Only supported for 1-D lightcurves.  Default
            is ``False``.
        variability_kwargs : dict or None, optional
            Keyword arguments forwarded to the variability tests (``alpha``,
            ``fvar_min``, ``stetson_k_min``; diagnostic reference). Only used
            when
            *check_variability* is ``True``.
        band : array-like of str or None, optional
            Optional per-row labels for a 2-D light curve.  Each element
            should be a string identifier (e.g. ``"V"``, ``"R"``,
            ``"W1"``).  The length must match the number of observation rows
            (``len(band) == len(xdata)`` for 2-D data, or 1 for 1-D data).
            ``None`` (default) means no band labels are stored.
        """
        super().__init__()

        transform_dic = {
            "minmax": MinMax,
            "zscore": ZScore,
            "robust_score": RobustZScore,
        }

        if xtransform is None or isinstance(xtransform, Transformer):
            self.xtransform = xtransform
        else:
            self.xtransform = transform_dic[xtransform]()

        if ytransform is None or isinstance(ytransform, Transformer):
            self.ytransform = ytransform
        else:
            self.ytransform = transform_dic[ytransform]()

        # Convert time units and coerce to tensors before non-finite filtering.
        xdata = _convert_time_to_days(xdata, time_units)
        xdata = self._ensure_tensor(xdata)
        ydata = self._ensure_tensor(ydata)
        if yerr is not None:
            yerr = self._ensure_tensor(yerr)

        _valid_rows_mask = None
        if ydata.dim() == 1 and band is not None and xdata.dim() > 1:
            _valid_rows_mask = torch.isfinite(ydata)
            _valid_rows_mask &= torch.isfinite(xdata).all(dim=1)
            if yerr is not None:
                _valid_rows_mask &= torch.isfinite(yerr)

        # Drop rows that contain NaN or Inf in any of the data arrays so that
        # all subsequent operations (transforms, GP training, LS) see only
        # finite values.  Only applied when ydata is 1-D (the standard case
        # for all supported GP models).  Non-standard multi-dimensional ydata
        # (e.g. legacy test fixtures with shape (D, N)) bypass this step;
        # those cases rely on the existing per-setter NaN validation.
        if ydata.dim() == 1:
            xdata, ydata, yerr = self._drop_nonfinite_rows(xdata, ydata, yerr)

        self.xdata = xdata
        self.ydata = ydata
        if yerr is not None:
            self.yerr = yerr

        self.name = "Lightcurve" if name is None else name

        # ------------------------------------------------------------------
        # Band labels
        # ------------------------------------------------------------------
        if band is None:
            self.band = None
        else:
            band_arr = np.asarray(band, dtype=np.str_)
            if band_arr.ndim != 1:
                raise ValueError(
                    f"'band' must be a 1-D array-like of strings (shape (n,)); "
                    f"got shape {band_arr.shape}."
                )
            if (
                _valid_rows_mask is not None
                # Keep this as a defensive guard: if caller supplied a
                # mis-sized band array, length validation below should still
                # raise the existing ValueError with the expected message.
                and len(band_arr) == len(_valid_rows_mask)
            ):
                band_arr = band_arr[_valid_rows_mask.detach().cpu().numpy()]
            # Determine the expected length: one label per observation row for
            # 2-D data, or 1 for 1-D data (single-band lightcurve).
            if self.ndim > 1:
                n_rows = len(self._xdata_raw)
            else:
                n_rows = 1
            if len(band_arr) != n_rows:
                raise ValueError(
                    f"Length of 'band' ({len(band_arr)}) does not match the "
                    f"expected number of rows ({n_rows})."
                )
            self.band = band_arr

        self.__SET_LIKELIHOOD_CALLED = False
        self.__SET_MODEL_CALLED = False
        self.__CONTRAINTS_SET = False
        self.__PRIORS_SET = False
        self.__FITTED_MAP = False
        self.__FITTED_MCMC = False
        self.is_fitted = False
        self.fit_failed = False
        self.failure_reason = None
        self.failure_diagnostics = None
        self.failure_summary = None
        self.fit_history = []
        self.parameter_workflow_result = None

        # ------------------------------------------------------------------
        # Sampling quality check
        # ------------------------------------------------------------------
        if check_sampling:
            sk = sampling_kwargs or {}
            if self.ndim > 1:
                xdata_raw = self._xdata_raw
                if xdata_raw.dim() != 2 or xdata_raw.shape[1] != 2:
                    raise ValueError(
                        "For 2D/multiband light curves, xdata must have shape "
                        "(N, 2) with wavelength values in column 1. Received "
                        f"shape {tuple(xdata_raw.shape)}. Please ensure "
                        "that your input is not transposed or otherwise "
                        "malformed."
                    )
                results = self.assess_sampling_quality_per_band(
                    verbose=False, **sk
                )
                failing = results["summary"]["failing_wavelengths"]
                passing = results["summary"]["passing_wavelengths"]

                for wl in failing:
                    diag = results[float(wl)]
                    warnings_str = ", ".join(diag["warnings"])
                    warnings.warn(
                        f"Skipping band \u03bb={wl} due to poor "
                        f"temporal sampling: {warnings_str}",
                        UserWarning,
                        stacklevel=2,
                    )

                if not passing:
                    raise ValueError(
                        "No wavelength bands passed sampling quality checks. "
                        "GP fitting is not recommended.\n"
                        "To force fitting anyway, use: "
                        "Lightcurve(..., check_sampling=False)"
                    )

                if failing:
                    n_pass = len(passing)
                    n_total = results["summary"]["n_bands"]
                    skipped = [round(w, 4) for w in failing]
                    _msg = (
                        f"Retaining {n_pass}/{n_total} wavelength bands after "
                        f"sampling-quality filtering (skipping \u03bb = "
                        f"{skipped})."
                    )
                    warnings.warn(_msg, UserWarning, stacklevel=2)
                    keep_mask = torch.isin(
                        xdata_raw[:, 1],
                        torch.tensor(
                            passing,
                            dtype=xdata_raw.dtype,
                            device=xdata_raw.device,
                        ),
                    )
                    self.xdata = xdata_raw[keep_mask].clone()
                    self.ydata = self._ydata_raw[keep_mask].clone()
                    if hasattr(self, "_yerr_raw"):
                        self.yerr = self._yerr_raw[keep_mask].clone()
                    if self.band is not None:
                        self.band = self.band[keep_mask.detach().cpu().numpy()]
            else:
                from pgmuvi.preprocess.quality import assess_sampling_quality

                t = self._xdata_raw.detach().cpu().numpy()
                if t.ndim > 1:
                    t = t[:, 0]
                y_np = (
                    self._ydata_raw.detach().cpu().numpy()
                    if hasattr(self, "_ydata_raw")
                    else None
                )
                yerr_np = (
                    self._yerr_raw.detach().cpu().numpy()
                    if hasattr(self, "_yerr_raw")
                    else None
                )
                passes, diag = assess_sampling_quality(
                    t, y_np, yerr_np, verbose=False, **sk
                )
                if not passes:
                    warnings_str = "\n".join(
                        f"  \u2022 {w}" for w in diag["warnings"]
                    )
                    raise ValueError(
                        f"Lightcurve has poor temporal sampling:\n"
                        f"{warnings_str}\n\n"
                        f"Recommendation: {diag['recommendation']}\n"
                        "GP fitting not recommended for poorly sampled data.\n"
                        "To force fitting anyway, use: "
                        "Lightcurve(..., check_sampling=False)"
                    )

        # ------------------------------------------------------------------
        # Variability check
        # ------------------------------------------------------------------
        if check_variability:
            from pgmuvi.preprocess.variability import is_variable

            if self.ndim > 1:
                raise ValueError(
                    "check_variability=True is not supported for multiband "
                    "(ndim > 1) lightcurves, because pooling bands may produce "
                    "misleading variability results. Use "
                    "check_variability_per_band() or filter_variable_bands() "
                    "to assess each band independently."
                )

            vkwargs = variability_kwargs or {}
            y_v, yerr_v = self._get_variability_arrays()
            is_var, diag = is_variable(y_v, yerr_v, **vkwargs)

            if not is_var:
                raise ValueError(
                    f"Lightcurve shows NO significant variability:\n"
                    f"  p-value: {diag['p_value']:.4f} "
                    f"[{'PASS' if diag['tests_passed']['chi2_test'] else 'FAIL'}]\n"
                    f"  F_var: {diag['fvar']:.4f} "
                    f"[{'PASS' if diag['tests_passed']['fvar_test'] else 'FAIL'}]\n"
                    f"  Stetson K: {diag['stetson_k']:.3f} "
                    f"[{'PASS' if diag['tests_passed']['stetson_test'] else 'FAIL'}]\n"
                    f"Decision: {diag['decision']}\n\n"
                    "GP fitting not recommended for non-variable sources.\n"
                    "To force fitting anyway, use: "
                    "Lightcurve(..., check_variability=False)"
                )

        # ------------------------------------------------------------------
        # Subsampling: permanently reduce the stored data while preserving
        # the temporal baseline and the max-gap constraint.
        #
        # 1D: subsample the whole array when it exceeds max_samples.
        # 2D: subsample each band independently when it exceeds
        #     max_samples_per_band (max_samples is used only as an advisory
        #     threshold to warn about total compute budget).
        # ------------------------------------------------------------------
        _do_2d = self.ndim > 1 and max_samples_per_band is not None
        _do_1d = self.ndim == 1 and max_samples is not None
        if _do_1d or _do_2d:
            from pgmuvi.preprocess import subsample_lightcurve

            # Subsampling is only valid for the standard (N,) or (N,2) shapes
            # where observations lie along dimension 0.  Non-standard
            # multi-dimensional ydata (e.g. shape (D, N)) would subsample the
            # wrong axis; raise a clear error rather than silently misbehaving.
            if self._ydata_raw.dim() != 1:
                raise ValueError(
                    "max_samples is only supported for standard 1-D "
                    "ydata (shape (N,)). The supplied ydata has shape "
                    f"{tuple(self._ydata_raw.shape)}."
                )

            mgf = (sampling_kwargs or {}).get("max_gap_fraction", 0.3)
            _buffer_names = (
                "_xdata_raw",
                "_xdata_transformed",
                "_ydata_raw",
                "_ydata_transformed",
                "_yerr_raw",
                "_yerr_transformed",
            )

            if _do_2d:
                # 2D (multiband): subsample each band independently using
                # max_samples_per_band.  Bands already at or below the limit
                # are left untouched.  The second column of xdata holds the
                # numeric wavelength/band identifier for standard 2-D
                # lightcurves.
                if (
                    self._xdata_raw.dim() != 2
                    or self._xdata_raw.shape[1] != 2
                ):
                    raise ValueError(
                        "Per-band subsampling requires xdata of shape "
                        "(N, 2) with time in column 0 and wavelength in "
                        "column 1. Received shape "
                        f"{tuple(self._xdata_raw.shape)}. Please ensure "
                        "that your input is not transposed or otherwise "
                        "malformed."
                    )
                xdata_np = self._xdata_raw.detach().cpu().numpy()
                band_ids = xdata_np[:, 1]
                unique_bands = np.unique(band_ids)
                global_keep = []
                subsampled_bands = []
                for bval in unique_bands:
                    band_mask = np.where(band_ids == bval)[0]
                    n_band = len(band_mask)
                    if n_band > max_samples_per_band:
                        t_band = xdata_np[band_mask, 0]
                        local_idx = subsample_lightcurve(
                            t_band,
                            max_samples=max_samples_per_band,
                            max_gap_fraction=mgf,
                            random_seed=subsample_seed,
                        )
                        global_keep.append(band_mask[local_idx])
                        subsampled_bands.append(bval)
                    else:
                        global_keep.append(band_mask)
                if subsampled_bands:
                    _band_str = ", ".join(
                        f"\u03bb={b}" for b in subsampled_bands
                    )
                    _struct_lines = "\n".join(
                        f"    \u03bb={bval}: {len(keep)} points"
                        for bval, keep in zip(
                            unique_bands, global_keep, strict=True
                        )
                    )
                    _msg = (
                        "The following bands exceed "
                        f"max_samples_per_band={max_samples_per_band}"
                        f" and were randomly subsampled: {_band_str}. "
                        "Set max_samples_per_band=None to disable "
                        "subsampling.\nThe subsampled 2D light curve has "
                        f"the following structure:\n{_struct_lines}"
                    )
                    warnings.warn(_msg, UserWarning, stacklevel=2)
                    idx = np.concatenate(global_keep)
                    # Sort by time column to preserve temporal ordering.
                    idx = idx[
                        np.argsort(xdata_np[idx, 0], kind="stable")
                    ]
                    idx_t = torch.as_tensor(
                        idx,
                        dtype=torch.long,
                        device=self._xdata_raw.device,
                    )
                    for bname in _buffer_names:
                        if (
                            hasattr(self, bname)
                            and getattr(self, bname) is not None
                        ):
                            self.register_buffer(
                                bname, getattr(self, bname)[idx_t]
                            )
                    if self.band is not None:
                        self.band = self.band[idx]
            else:
                # 1D light curve: subsample the whole array if it exceeds the
                # limit.
                n_total = self._xdata_raw.shape[0]
                if n_total > max_samples:
                    t_np = self._xdata_raw.detach().cpu().numpy()
                    idx = subsample_lightcurve(
                        t_np,
                        max_samples=max_samples,
                        max_gap_fraction=mgf,
                        random_seed=subsample_seed,
                    )
                    warnings.warn(
                        f"Lightcurve has {n_total} points, which exceeds "
                        f"max_samples={max_samples}. Retaining a random "
                        f"subsample of {len(idx)} points. "
                        "Set max_samples=None to disable subsampling.",
                        UserWarning,
                        stacklevel=2,
                    )
                    idx_t = torch.as_tensor(
                        idx,
                        dtype=torch.long,
                        device=self._xdata_raw.device,
                    )
                    for bname in _buffer_names:
                        if (
                            hasattr(self, bname)
                            and getattr(self, bname) is not None
                        ):
                            self.register_buffer(
                                bname, getattr(self, bname)[idx_t]
                            )

        # Advisory warning for 2D lightcurves: notify if the total point
        # count exceeds max_samples, regardless of whether per-band
        # subsampling was performed.
        if (
            self.ndim > 1
            and max_samples is not None
            and self._xdata_raw.shape[0] > max_samples
        ):
            _msg = (
                f"Lightcurve has {self._xdata_raw.shape[0]} points, "
                f"which exceeds max_samples={max_samples}. "
                "Execution may be slow. Consider setting "
                "max_samples_per_band to reduce the total size of "
                "the lightcurve."
            )
            warnings.warn(_msg, UserWarning, stacklevel=2)

    @classmethod
    def from_table(
        cls,
        tab,
        file_format="votable",
        xcol="x",
        ycol="y",
        yerrcol="yerr",
        bandcol=None,
        **kwargs,
    ):
        """Instantiate a Lightcurve object with
        data read in from a VOTable.

        Parameters
        ----------
        tab: astropy.table.Table object or str or pathlib.Path instance
            Table containing the input data. If str, name (with extension) of
            file containing the input data. In this case, the file_format
            keyword must be set accordingly.
        file_format: str
            Format of file containing input data. Must be a format supported
            by Table.read. Only required if type(tab) is str.
        xcol: str
            Name of column in table that contains the x data
        ycol: str
            Name of column in table that contains the y data
        yerrcol: str
            Name of column in table that contains the yerr data
        bandcol: str or None, optional
            Name of the column containing string band labels (e.g. ``"V"``,
            ``"R"``, ``"W1"``).  When provided, the per-row labels are read
            from this column and stored in :attr:`Lightcurve.band`.  If
            ``None`` (default), the method attempts to auto-detect a
            string-typed column whose name matches one of the entries in
            :attr:`_WAVELENGTH_ID_COLUMN_NAMES` (e.g. ``"band"``,
            ``"filter"``); if found it is used as the band-label column.
            The ``band`` kwarg in ``kwargs`` always takes precedence.
        kwargs:
            Arguments to be passed to the Lightcurve constructor, including
            ``time_units`` (str or ``astropy.units`` unit, default *None*).
            If ``time_units`` is provided, the time axis read from *xcol* will
            be converted to days before being stored.

        Returns
        ----------
        Lightcurve object
        """
        from pathlib import Path
        from astropy.table import Table

        if isinstance(tab, str) or isinstance(tab, Path):
            data = Table.read(tab, format=file_format)
        elif isinstance(tab, Table):
            data = tab
        else:
            raise ValueError(
                "Input tab must be an instance of str, pathlib.Path, "
                "or astropy.table.Table!"
            )
        c = data.colnames
        if xcol not in c:
            raise ValueError(f"Table does not have column '{xcol}'")
        if ycol not in c:
            raise ValueError(f"Table does not have column '{ycol}'")

        ndim = len(data[xcol].squeeze().shape)
        x = torch.Tensor(data[xcol]).squeeze()
        y = torch.Tensor(data[ycol]).squeeze()
        if (ndim == 1) or (ndim == 2):
            if yerrcol not in c:
                yerr = None
            else:
                yerr = torch.Tensor(data[yerrcol]).squeeze()
        else:
            mesg = f"Column '{xcol}' must have shape (1, nsamples) or (1, 2, nsamples)"
            raise ValueError(mesg)

        # Ensure x is shaped (N, D) rather than (D, N) before NaN filtering,
        # since some table column shapes can squeeze to (D, N).
        if x.dim() == 2 and y.dim() >= 1:
            nsamples = y.shape[0]
            if x.shape[0] != nsamples and x.shape[1] == nsamples:
                x = x.transpose(0, 1)
        if yerr is not None and yerr.dim() == 2:
            nsamples = y.shape[0]
            if yerr.shape[0] != nsamples and yerr.shape[1] == nsamples:
                yerr = yerr.transpose(0, 1)

        # Compute the finite-row mask before calling _drop_nonfinite_rows so
        # that we can apply the same filter to the ancillary band column below.
        _valid = torch.isfinite(y)
        if x.dim() > 1:
            _valid &= torch.isfinite(x).all(dim=1)
        else:
            _valid &= torch.isfinite(x)
        if yerr is not None:
            _valid &= torch.isfinite(yerr)
        _valid_np = _valid.numpy().astype(bool)

        x, y, yerr = cls._drop_nonfinite_rows(x, y, yerr)

        # ------------------------------------------------------------------
        # Band labels: only relevant when xdata is already 2-D (multiband).
        # For 1-D lightcurves there is no wavelength axis, so band labels
        # from an ancillary string column would be meaningless.
        # One label per row is stored (same length as xdata).
        # ------------------------------------------------------------------
        if "band" not in kwargs and x.dim() == 2:
            # Prefer the explicit bandcol; fall back to auto-detection.
            if bandcol is None:
                bandcol = cls._find_column(c, cls._WAVELENGTH_ID_COLUMN_NAMES)
            if bandcol is not None and bandcol in c:
                col_data = np.asarray(data[bandcol])
                col_dtype = col_data.dtype
                if (
                    np.issubdtype(col_dtype, np.str_)
                    or np.issubdtype(col_dtype, np.bytes_)
                    or col_dtype.kind == "O"
                ):
                    # String column found — store per-row labels (filtered to
                    # the same valid rows as x/y/yerr).
                    kwargs["band"] = np.array(
                        col_data[_valid_np].astype(str), dtype=np.str_
                    )

        return cls(x, y, yerr, **kwargs)

    @property
    def ndim(self):
        return self.xdata.shape[-1] if self.xdata.dim() > 1 else 1

    @property
    def magnitudes(self):
        pass

    @magnitudes.setter
    def magnitudes(self, value):
        pass

    @property
    def xdata(self):
        """The independent variable data

        :getter: Returns the independent variable data in its raw
        (untransformed) state
        :setter: Takes the input data and transforms it as requested by the
        user
        :type: torch.Tensor
        """
        return self._xdata_raw

    @xdata.setter
    def xdata(self, values):
        # first, check that the input is a tensor
        # and modifiy it if necessary
        values = self._ensure_tensor(values)
        # check that the input has more than one element
        # and raise an exception if not
        values = self._ensure_dim(values)
        # check if there are any NaNs in the inputs
        if torch.isnan(values).any():
            errmsg = f"The x values contain {torch.isnan(values).sum()} NaNs."
            raise ValueError(errmsg)
        # then, store the raw data internally
        self.register_buffer("_xdata_raw", values)
        # then, apply the transformation to the values, so it can be used to
        # train the GP
        if self.xtransform is None:
            self.register_buffer("_xdata_transformed", values)
        elif isinstance(self.xtransform, Transformer):
            self.register_buffer(
                "_xdata_transformed", self.xtransform.transform(values)
            )

    @property
    def ydata(self):
        """The dependent variable data

        :getter: Returns the dependent variable data in its raw
        (untransformed) state
        :setter: Takes the input data and transforms it as requested by the
        user
        :type: torch.Tensor"""
        return self._ydata_raw

    @ydata.setter
    def ydata(self, values):
        # first, check that the input is a tensor
        # and modifiy it if necessary
        values = self._ensure_tensor(values)
        # then, store the raw data internally
        # check if there are any NaNs in the inputs
        if torch.isnan(values).any():
            errmsg = f"The y values contain {torch.isnan(values).sum()} NaNs."
            raise ValueError(errmsg)
        self.register_buffer("_ydata_raw", values)
        # then, apply the transformation to the values
        if self.ytransform is None:
            self.register_buffer("_ydata_transformed", values)
        elif isinstance(self.ytransform, Transformer):
            self.register_buffer(
                "_ydata_transformed", self.ytransform.transform(values)
            )

    @property
    def yerr(self):
        """The uncertainties on the dependent variable data

        :getter: Returns the uncertainties on the dependent variable data in
        its raw (untransformed) state
        :setter: Takes the input data and transforms it as requested by the
        user
        :type: torch.Tensor
        """
        return self._yerr_raw

    @yerr.setter
    def yerr(self, values):
        # first, check that the input is a tensor
        # and modifiy it if necessary
        values = self._ensure_tensor(values)
        # check if there are any NaNs in the inputs
        if torch.isnan(values).any():
            errmsg = f"The y uncertainties contain {torch.isnan(values).sum()} NaNs."
            raise ValueError(errmsg)
        # then, store the raw data internally
        self.register_buffer("_yerr_raw", values)
        # now apply the same transformation that was applied to the ydata
        if self.ytransform is None:
            self.register_buffer("_yerr_transformed", values)
        elif isinstance(self.ytransform, Transformer):
            self.register_buffer("_yerr_transformed", self.ytransform.transform(values))

    def _ensure_tensor(self, values):
        # Ensures that the input data has type torch.Tensor
        # Transforms the data if necessary
        if not isinstance(values, torch.Tensor):
            warnings.warn(
                (
                    "The function expects a torch.Tensor as input."
                    "Your data will be converted to a tensor."
                ),
                stacklevel=2,
            )
            values = torch.as_tensor(values, dtype=torch.float32)
        return values

    def _ensure_dim(self, values):
        # Ensures that the input data has more than one element
        # Returns an exception if not
        if values.numel() == 1:
            raise ValueError("The input data must have more than one element.")
        # elif values.numel() < threshold:
        #    warnings.warn(('The input data has less than threshold elements.'
        #        'This may lead to poor performance.'),
        #        stacklevel=2)
        return values

    def append_data(self, new_values_x, new_values_y):
        pass

    def select_bands(
        self, bands: list | tuple | np.ndarray
    ) -> "Lightcurve":
        """Return a new Lightcurve containing only the requested bands.

        Bands are identified exclusively through the :attr:`band` attribute.
        Wavelength values play no role in selection.

        Parameters
        ----------
        bands : list, tuple, or numpy.ndarray
            A sequence of string band labels to select.  Each element must
            be a ``str`` or ``numpy.str_``; the value is coerced to ``str``
            before comparison so numpy string scalars are handled naturally.
            ``bytes``, numeric types, and ``None`` are rejected.

        Returns
        -------
        Lightcurve
            A new :class:`Lightcurve` object built from the subset of rows
            whose :attr:`band` matches at least one of the requested labels.
            The :attr:`name`, :attr:`xtransform`, and :attr:`ytransform`
            attributes are inherited from the original light curve, and the
            subsetted :attr:`band` array is preserved.

        Raises
        ------
        TypeError
            If *bands* is not a ``list``, ``tuple``, or ``numpy.ndarray``.
        TypeError
            If *bands* is a bare string (use ``["label"]`` instead).
        TypeError
            If any element of *bands* is not a ``str`` or ``numpy.str_``
            (e.g. ``bytes``, ``int``, ``float``, ``None``, or a nested list).
        ValueError
            If :attr:`band` is ``None``.
        ValueError
            If none of the requested labels are present in :attr:`band`.
        """
        if isinstance(bands, str):
            raise TypeError(
                "'bands' must be a sequence of band labels (list, tuple, or "
                "numpy.ndarray), not a bare string. "
                "To select a single band wrap it in a list: "
                f"select_bands([{bands!r}])"
            )

        if not isinstance(bands, (list, tuple, np.ndarray)):
            raise TypeError(
                f"'bands' must be a list, tuple, or numpy.ndarray; "
                f"got {type(bands).__name__!r}."
            )

        if self.band is None:
            raise ValueError(
                "select_bands requires the 'band' attribute to be set, "
                "but this Lightcurve has band=None."
            )

        str_labels = []
        for b in bands:
            if b is None:
                raise TypeError(
                    "None is not a valid band selector in 'bands'."
                )
            if isinstance(b, (float, int, np.floating, np.integer)):
                raise TypeError(
                    f"Numeric selectors are not supported by select_bands; "
                    f"got {type(b).__name__!r} ({b!r}). "
                    "Use a string band label instead."
                )
            if not isinstance(b, (str, np.str_)):
                raise TypeError(
                    f"Each element of 'bands' must be a string band label; "
                    f"got {type(b).__name__!r}."
                )
            str_labels.append(str(b))

        xdata_raw = self._xdata_raw
        n = xdata_raw.shape[0]

        # 1-D lightcurves store a single band label (len(self.band) == 1)
        # that applies to all observations.  Build the mask differently to
        # avoid a length-1 vs length-N boolean-index mismatch.
        if len(self.band) == 1:
            single_label = str(self.band[0])
            if single_label not in str_labels:
                raise ValueError(
                    f"None of the requested band labels {str_labels!r} were "
                    "found in this Lightcurve's 'band' attribute."
                )
            # The whole lightcurve belongs to this band — return it unchanged.
            return Lightcurve(
                xdata_raw,
                self._ydata_raw,
                yerr=(
                    self._yerr_raw
                    if hasattr(self, "_yerr_raw")
                    else None
                ),
                xtransform=self.xtransform,
                ytransform=self.ytransform,
                name=self.name,
                band=self.band,
            )

        band_str = self.band.astype(str)
        mask = np.zeros(n, dtype=bool)
        for label in str_labels:
            mask |= band_str == label

        if not mask.any():
            raise ValueError(
                f"None of the requested band labels {str_labels!r} were "
                "found in this Lightcurve's 'band' attribute."
            )

        mask_tensor = torch.as_tensor(
            mask, dtype=torch.bool, device=xdata_raw.device
        )
        new_x = xdata_raw[mask_tensor]
        new_y = self._ydata_raw[mask_tensor]
        new_yerr = (
            self._yerr_raw[mask_tensor] if hasattr(self, "_yerr_raw") else None
        )
        new_band = self.band[mask]

        return Lightcurve(
            new_x,
            new_y,
            yerr=new_yerr,
            xtransform=self.xtransform,
            ytransform=self.ytransform,
            name=self.name,
            band=new_band,
        )

    def drop_bands(
        self, bands: list | tuple | np.ndarray
    ) -> "Lightcurve":
        """Return a new Lightcurve with the specified bands removed.

        This complements the string band-label behavior of
        :meth:`select_bands`: every row whose :attr:`band` label appears
        in *bands* is excluded from the returned object.

        Parameters
        ----------
        bands : list, tuple, or numpy.ndarray
            Band labels to remove.  Each element must be a string
            (``str`` or ``numpy.str_``). Numeric selectors, ``bytes``,
            ``None``, and nested containers are not accepted.

        Returns
        -------
        Lightcurve
            A new :class:`Lightcurve` built from the rows whose
            :attr:`band` label is **not** in *bands*.  The
            :attr:`name`, :attr:`xtransform`, and :attr:`ytransform`
            attributes are inherited from the original light curve.
            If none of the requested labels are present in the data
            the returned object is a copy of the original (no-op).

        Raises
        ------
        TypeError
            If *bands* is a bare string rather than a sequence.
        TypeError
            If any element of *bands* is not a string.
        ValueError
            If :attr:`band` is ``None``.
        ValueError
            If all rows are removed (no data would remain).
        """
        if isinstance(bands, str):
            raise TypeError(
                "'bands' must be a sequence of labels (list, tuple, or "
                "numpy.ndarray), not a bare string. "
                "To drop a single band wrap it in a list: "
                f"drop_bands([{bands!r}])"
            )

        if not isinstance(bands, (list, tuple, np.ndarray)):
            raise TypeError(
                "'bands' must be a list, tuple, or numpy.ndarray; "
                f"got {type(bands).__name__!r}."
            )

        for b in bands:
            if not isinstance(b, (str, np.str_)):
                raise TypeError(
                    "Each element of 'bands' must be a string; "
                    f"got {type(b).__name__!r}."
                )

        if self.band is None:
            raise ValueError(
                "drop_bands requires the 'band' attribute to be set, "
                "but this Lightcurve has band=None."
            )

        xdata_raw = self._xdata_raw
        band_arr = self.band.astype(str)
        requested = {str(b) for b in bands}

        if len(band_arr) == len(xdata_raw):
            mask = ~np.isin(band_arr, list(requested))
            if not mask.any():
                raise ValueError(
                    "All rows were removed by drop_bands; no data remains."
                )
            new_band = self.band[mask]
        elif len(band_arr) == 1:
            if band_arr[0] in requested:
                raise ValueError(
                    "All rows were removed by drop_bands; no data remains."
                )
            mask = np.ones(len(xdata_raw), dtype=bool)
            new_band = self.band.copy()
        else:
            raise ValueError(
                "drop_bands requires 'band' to have either one label "
                "for the whole lightcurve or one label per observation row."
            )

        tensor_mask = torch.as_tensor(
            mask, dtype=torch.bool, device=xdata_raw.device
        )
        new_x = xdata_raw[tensor_mask]
        new_y = self._ydata_raw[tensor_mask]
        new_yerr = (
            self._yerr_raw[tensor_mask] if hasattr(self, "_yerr_raw") else None
        )

        return Lightcurve(
            new_x,
            new_y,
            yerr=new_yerr,
            xtransform=self.xtransform,
            ytransform=self.ytransform,
            name=self.name,
            band=new_band,
        )

    @staticmethod
    def _sanitize_fit_history_value(value):
        """Return a JSON-safe value for fit-history bookkeeping."""
        return Lightcurve._consensus_make_json_safe(value)

    @staticmethod
    def _fit_configuration_safe_repr(value, *, max_length=240):
        """Return a bounded repr/str fallback for fit-configuration display."""
        try:
            _text = repr(value)
        except Exception:
            try:
                _text = str(value)
            except Exception:
                _type = type(value)
                _text = f"<{_type.__module__}.{_type.__name__}>"
        if not isinstance(_text, str):
            try:
                _text = str(_text)
            except Exception:
                _text = "<unrepresentable>"
        if len(_text) > max_length:
            return _text[: max_length - 3] + "..."
        return _text

    @staticmethod
    def _fit_configuration_make_unserializable_placeholder(
        value,
        *,
        type_name=None,
        module_name=None,
        repr_text=None,
        extra_fields=None,
    ):
        """Return an explicit placeholder for unsupported fit-config objects."""
        _type = type(value)
        _placeholder = {
            "__unserializable__": True,
            "type": type_name or _type.__name__,
            "module": module_name or _type.__module__,
            "repr": repr_text
            if repr_text is not None
            else Lightcurve._fit_configuration_safe_repr(value),
        }
        if isinstance(extra_fields, dict):
            for _key, _val in extra_fields.items():
                if _val is not None:
                    _placeholder[_key] = _val
        return _placeholder

    @staticmethod
    def _fit_configuration_make_truncation_marker(type_name, **metadata):
        """Return a structured truncation marker for large/deep values."""
        _marker = {
            "__truncated__": True,
            "type": type_name,
        }
        for _key, _val in metadata.items():
            if _val is not None:
                _marker[_key] = _val
        return _marker

    @staticmethod
    def _fit_configuration_sort_key(value):
        """Return a deterministic ordering key for unordered containers."""
        _type = type(value)
        return (
            f"{_type.__module__}.{_type.__name__}:"
            f"{Lightcurve._fit_configuration_safe_repr(value, max_length=120)}"
        )

    @staticmethod
    def _sanitize_fit_configuration_value(
        value,
        *,
        max_items=20,
        max_string_length=240,
        max_depth=8,
        _depth=0,
        _seen=None,
    ):
        """Return a compact, JSON-safe representation of a fit-configuration value.

        Fit-configuration snapshots are stored inside fit-history entries and
        are intended to be serializable (via ``json.dumps``), portable across
        processes, and compact enough to display in a notebook.  Raw Python
        objects fail these requirements in several ways:

        * **Tensors and arrays** hold large numerical payloads that should not
          be stored verbatim; they also contain device/dtype metadata that is
          not JSON-native.  Arrays up to ``max_items`` elements are expanded;
          larger ones are replaced by a summary dict with a ``"preview"`` list.
        * **Classes, callables, and opaque runtime objects** are
          environment-specific references that cannot be round-tripped through
          JSON; they are replaced by explicit placeholder dicts so downstream
          tooling can detect them.
        * **Constraint / prior objects** from gpytorch / pyro carry internal
          state that is non-serializable; they are reduced to readable,
          structured summaries instead of raw repr dumps.
        * **Non-finite floats** (NaN, ±Inf) are not valid JSON values and are
          converted to explicit marker dicts rather than leaking through.
        * **Circular references** would cause infinite recursion and are
          detected via an identity set; they are replaced by a sentinel string.
        * **Sets and tuples** are converted to JSON arrays (lists), with sets
          sorted deterministically.

        The output is deterministic for a given input type: the same Python
        type always produces the same JSON-safe representation.

        This sanitizer is intentionally stricter than the general
        ``_sanitize_fit_history_value`` helper: fit-configuration snapshots
        must remain compact and human-readable, not just technically safe.

        Parameters
        ----------
        value : object
            The value to sanitize.  Any Python object is accepted.
        max_items : int, optional
            Maximum number of elements to expand for sequences, dicts,
            arrays, and tensors before switching to a summary representation.
            Defaults to 20.
        _depth : int, optional
            Internal recursion-depth guard.  Do not pass from user code.
        _seen : set or None, optional
            Internal identity-set for circular-reference detection.
            Do not pass from user code.

        Returns
        -------
        bool | int | str | float | list | dict | None
            A JSON-safe value.  Scalars remain scalars where possible; large
            strings and containers become explicit truncation markers; opaque
            unsupported objects become explicit placeholder dicts.
        """
        if _seen is None:
            _seen = set()
        if _depth > max_depth:
            return Lightcurve._fit_configuration_make_truncation_marker(
                type(value).__name__,
                reason="max_depth",
                max_depth=max_depth,
            )
        _added_to_seen = False
        try:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                if len(value) <= max_string_length:
                    return value
                return Lightcurve._fit_configuration_make_truncation_marker(
                    "str",
                    length=len(value),
                    preview=value[:max_string_length],
                    truncated_chars=len(value) - max_string_length,
                )
            if isinstance(value, float):
                if math.isfinite(value):
                    return value
                if math.isnan(value):
                    _value_name = "nan"
                elif value > 0:
                    _value_name = "inf"
                else:
                    _value_name = "-inf"
                return {
                    "__non_finite__": True,
                    "value": _value_name,
                }
            if isinstance(value, np.generic):
                return Lightcurve._sanitize_fit_configuration_value(
                    value.item(),
                    max_items=max_items,
                    max_string_length=max_string_length,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, enum.Enum):
                return f"{type(value).__name__}.{value.name}"
            if (
                type(value).__module__ == "torch"
                and type(value).__name__ in {"device", "dtype"}
            ):
                return str(value)
            if isinstance(value, Interval):
                return {
                    "type": type(value).__name__,
                    "module": type(value).__module__,
                    "lower": Lightcurve._sanitize_fit_configuration_value(
                        value.lower_bound,
                        max_items=max_items,
                        max_string_length=max_string_length,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _seen=_seen,
                    ),
                    "upper": Lightcurve._sanitize_fit_configuration_value(
                        value.upper_bound,
                        max_items=max_items,
                        max_string_length=max_string_length,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _seen=_seen,
                    ),
                }
            if isinstance(value, gpytorch.priors.Prior):
                return {
                    "type": type(value).__name__,
                    "module": type(value).__module__,
                    "repr": Lightcurve._fit_configuration_safe_repr(value),
                }
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                try:
                    _fields = dataclasses.asdict(value)
                except Exception:
                    return (
                        Lightcurve._fit_configuration_make_unserializable_placeholder(
                            value
                        )
                    )
                return {
                    "type": type(value).__name__,
                    "module": type(value).__module__,
                    "fields": Lightcurve._sanitize_fit_configuration_value(
                        _fields,
                        max_items=max_items,
                        max_string_length=max_string_length,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _seen=_seen,
                    ),
                }

            _obj_id = id(value)
            if _obj_id in _seen:
                return Lightcurve._fit_configuration_make_truncation_marker(
                    type(value).__name__,
                    reason="recursive_reference",
                )
            if isinstance(value, dict | list | tuple | set | np.ndarray) or (
                torch.is_tensor(value)
            ):
                _seen.add(_obj_id)
                _added_to_seen = True

            if isinstance(value, np.ndarray):
                _size = int(value.size)
                if _size <= max_items:
                    return Lightcurve._sanitize_fit_configuration_value(
                        value.tolist(),
                        max_items=max_items,
                        max_string_length=max_string_length,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _seen=_seen,
                    )
                _flat_preview = value.reshape(-1)[:max_items].tolist()
                return {
                    "__truncated__": True,
                    "type": "ndarray",
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "size": _size,
                    "preview": [
                        Lightcurve._sanitize_fit_configuration_value(
                            v,
                            max_items=max_items,
                            max_string_length=max_string_length,
                            max_depth=max_depth,
                            _depth=_depth + 1,
                            _seen=_seen,
                        )
                        for v in _flat_preview
                    ],
                    "truncated_items": max(_size - max_items, 0),
                }

            if torch.is_tensor(value):
                _numel = int(value.numel())
                if _numel == 1:
                    return Lightcurve._sanitize_fit_configuration_value(
                        value.item(),
                        max_items=max_items,
                        max_string_length=max_string_length,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _seen=_seen,
                    )
                if _numel <= max_items:
                    return Lightcurve._sanitize_fit_configuration_value(
                        value.detach().cpu().tolist(),
                        max_items=max_items,
                        max_string_length=max_string_length,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _seen=_seen,
                    )
                _flat_preview = value.detach().cpu().reshape(-1)[:max_items].tolist()
                return {
                    "__truncated__": True,
                    "type": "tensor",
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "numel": _numel,
                    "device": str(value.device),
                    "preview": [
                        Lightcurve._sanitize_fit_configuration_value(
                            v,
                            max_items=max_items,
                            max_string_length=max_string_length,
                            max_depth=max_depth,
                            _depth=_depth + 1,
                            _seen=_seen,
                        )
                        for v in _flat_preview
                    ],
                    "truncated_items": max(_numel - max_items, 0),
                }

            if isinstance(value, dict):
                _items = sorted(value.items(), key=lambda item: str(item[0]))
                _truncated = _depth > 0 and len(_items) > max_items
                if _truncated:
                    _items = _items[:max_items]
                _out = {
                    str(k): Lightcurve._sanitize_fit_configuration_value(
                        v,
                        max_items=max_items,
                        max_string_length=max_string_length,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _seen=_seen,
                    )
                    for k, v in _items
                }
                if _truncated:
                    return Lightcurve._fit_configuration_make_truncation_marker(
                        "dict",
                        length=len(value),
                        preview=_out,
                        truncated_items=int(len(value) - max_items),
                    )
                return _out

            if isinstance(value, list | tuple | set):
                _seq = list(value)
                if isinstance(value, set):
                    _seq = sorted(_seq, key=Lightcurve._fit_configuration_sort_key)
                _truncated = len(_seq) > max_items
                _seq = _seq[:max_items] if _truncated else _seq
                _out = [
                    Lightcurve._sanitize_fit_configuration_value(
                        v,
                        max_items=max_items,
                        max_string_length=max_string_length,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                        _seen=_seen,
                    )
                    for v in _seq
                ]
                if _truncated:
                    return Lightcurve._fit_configuration_make_truncation_marker(
                        type(value).__name__,
                        length=len(value),
                        preview=_out,
                        truncated_items=int(len(value) - max_items),
                    )
                return _out

            if isinstance(value, type):
                return Lightcurve._fit_configuration_make_unserializable_placeholder(
                    value,
                    type_name="type",
                    module_name=getattr(value, "__module__", type(value).__module__),
                    extra_fields={
                        "qualname": getattr(value, "__qualname__", None),
                    },
                )
            if callable(value):
                return Lightcurve._fit_configuration_make_unserializable_placeholder(
                    value,
                    type_name="callable",
                    module_name=getattr(value, "__module__", type(value).__module__),
                    extra_fields={
                        "qualname": getattr(value, "__qualname__", None)
                        or getattr(value, "__name__", None),
                    },
                )
        except Exception:
            return Lightcurve._fit_configuration_make_unserializable_placeholder(
                value
            )
        finally:
            if _added_to_seen:
                _seen.discard(_obj_id)

        return Lightcurve._fit_configuration_make_unserializable_placeholder(value)

    @staticmethod
    def _collect_fit_configuration_snapshot(
        *,
        fit_kwargs=None,
        context=None,
    ):
        """Return a compact, JSON-safe reproducibility snapshot for a fit call.

        This helper is called once per outermost :meth:`fit` invocation,
        before GP training begins, and the result is stored in the fit-history
        entry under the ``"fit_configuration"`` key.  It is intentionally
        best-effort: all exceptions are caught so that history recording is
        never blocked by serialization failures.

        Data sources
        ------------
        The snapshot draws from two sources:

        * **fit_kwargs** — the raw keyword arguments passed by the user to
          :meth:`fit`.  These capture what the user explicitly requested.
        * **context** — normalized/resolved values computed by :meth:`fit`
          before training begins.  These reflect what the code will actually
          use (e.g. a resolved model class name rather than the raw ``"model"``
          shorthand string the user passed).

        When both sources contain a value for the same concept, ``context``
        takes precedence because it holds the authoritative internal value.

        The distinction matters for auditing: ``user_kwargs`` stores what the
        caller wrote; all other fields store what the library understood.

        Parameters
        ----------
        fit_kwargs : dict, optional
            The raw ``**kwargs`` dict passed to :meth:`fit`.
        context : dict, optional
            Normalized / resolved values assembled by :meth:`fit` before
            calling :meth:`_fit_core` (e.g. resolved model class name,
            resolved training_iter, resolved backend string).

        Returns
        -------
        dict or None
            A JSON-safe snapshot dict (see schema below), or ``None`` if
            all recovery paths fail.
        """
        # ----------------------------------------------------------------
        # Schema of the returned dictionary
        # ----------------------------------------------------------------
        # The snapshot is a flat dict.  All values are JSON-safe after the
        # final _sanitize_fit_configuration_value pass.  Unresolvable values
        # are None.
        #
        # {
        #   "fit_strategy"    : str | None  — routing key ("standard", …)
        #   "model_class"     : str | None  — resolved class name, NOT the
        #                                     raw "model" kwarg the user passed
        #   "backend"         : str | None  — "cpu" or "cuda"
        #   "training_iter"   : int | None
        #   "learning_rate"   : float | None
        #   "optimizer"       : str | None  — callable reduced to its name
        #   "num_mixtures"    : int | None
        #   "use_best_band_init" : bool | None
        #   "use_gp_validation"  : bool | None
        #   "constraint_set"  : str | None  — human-readable label
        #   "prior_set"       : str | None
        #   "max_samples"     : int | None
        #   "max_samples_per_band" : int | None
        #   "min_period"      : float | None
        #   "max_period"      : float | None
        #   "frequency_bounds": [min, max] | None  — from explicit kwarg or
        #                        assembled from min_frequency/max_frequency
        #   "period_bounds"   : [min, max] | None
        #   "wavelength_bounds": [min, max] | None
        #   "xtransform"      : str | None
        #   "ytransform"      : str | None
        #   "normalize"       : bool | None
        #   "detrend"         : bool | None
        #   "consensus_configuration" : dict | None  (non-None only if any
        #       consensus kwarg was set); contains: constrain_consensus,
        #       consensus_method, consensus_sigma_clip, consensus_sigma,
        #       consensus_tolerance, consensus_max_harmonic,
        #       min_consensus_inliers, use_gp_validation
        #   "min_consensus_inliers" : int | None
        #   "outlier_thresholds" : dict | None  (non-None only if any outlier
        #       kwarg was set); contains: outlier_sigma_threshold,
        #       outlier_threshold, max_outlier_fraction
        #   "random_initialization_flags" : dict | None  (non-None only if
        #       any init-randomness kwarg was set); contains: use_mls_init,
        #       use_best_band_init, random_init, use_random_init,
        #       randomize_initialization
        #   "user_kwargs"     : dict | None  — raw user kwargs that do not
        #       map to any canonical key above; sanitized but otherwise
        #       uninterpreted; MUST NOT be used for logic
        # }
        #
        # MUST NOT appear here: raw tensors, ndarrays, callables, live GP
        # objects, non-finite floats.  Enforced by the final sanitization
        # pass below.
        # ----------------------------------------------------------------
        _snapshot = {
            "fit_strategy": None,
            "model_class": None,
            "backend": None,
            "training_iter": None,
            "learning_rate": None,
            "optimizer": None,
            "num_mixtures": None,
            "use_best_band_init": None,
            "use_gp_validation": None,
            "constraint_set": None,
            "prior_set": None,
            "max_samples": None,
            "max_samples_per_band": None,
            "min_period": None,
            "max_period": None,
            "frequency_bounds": None,
            "period_bounds": None,
            "wavelength_bounds": None,
            "xtransform": None,
            "ytransform": None,
            "normalize": None,
            "detrend": None,
            "consensus_configuration": None,
            "outlier_thresholds": None,
            "min_consensus_inliers": None,
            "random_initialization_flags": None,
            "user_kwargs": None,
        }

        try:
            _kwargs = fit_kwargs if isinstance(fit_kwargs, dict) else {}
            _context = context if isinstance(context, dict) else {}
            _resolved = {**_kwargs, **_context}

            def _pick(*keys):
                for _k in keys:
                    if _k in _resolved and _resolved[_k] is not None:
                        return _resolved[_k]
                return None

            _model_value = _pick("model_class", "model")
            if _model_value is not None:
                if isinstance(_model_value, str):
                    _snapshot["model_class"] = _model_value
                else:
                    _snapshot["model_class"] = _model_value.__class__.__name__

            _backend = _pick("backend")
            if _backend is None and "cuda" in _resolved:
                _backend = "cuda" if bool(_resolved.get("cuda")) else "cpu"
            _snapshot["backend"] = _backend

            _snapshot["fit_strategy"] = _pick("fit_strategy")
            _snapshot["training_iter"] = _pick("training_iter")
            _snapshot["learning_rate"] = _pick("learning_rate", "lr")
            _snapshot["optimizer"] = _pick("optimizer", "optim")
            _snapshot["num_mixtures"] = _pick("num_mixtures")
            _snapshot["use_best_band_init"] = _pick("use_best_band_init")
            _snapshot["use_gp_validation"] = _pick("use_gp_validation")
            _snapshot["constraint_set"] = _pick("constraint_set")
            _snapshot["prior_set"] = _pick("prior_set")
            _snapshot["max_samples"] = _pick("max_samples")
            _snapshot["max_samples_per_band"] = _pick("max_samples_per_band")
            _snapshot["min_period"] = _pick("min_period")
            _snapshot["max_period"] = _pick("max_period")
            _snapshot["xtransform"] = _pick("xtransform")
            _snapshot["ytransform"] = _pick("ytransform")
            _snapshot["normalize"] = _pick("normalize")
            _snapshot["detrend"] = _pick("detrend")
            _snapshot["min_consensus_inliers"] = _pick("min_consensus_inliers")

            _freq_bounds = _pick("frequency_bounds")
            if _freq_bounds is None:
                _min_freq = _pick("min_frequency", "min_freq")
                _max_freq = _pick("max_frequency", "max_freq")
                if _min_freq is not None or _max_freq is not None:
                    _freq_bounds = [_min_freq, _max_freq]
            _snapshot["frequency_bounds"] = _freq_bounds

            _period_bounds = _pick("period_bounds")
            if _period_bounds is None:
                _min_period = _pick("min_period")
                _max_period = _pick("max_period")
                if _min_period is not None or _max_period is not None:
                    _period_bounds = [_min_period, _max_period]
            _snapshot["period_bounds"] = _period_bounds

            _wavelength_bounds = _pick("wavelength_bounds")
            if _wavelength_bounds is None:
                _min_w = _pick("min_wavelength", "min_lambda")
                _max_w = _pick("max_wavelength", "max_lambda")
                if _min_w is not None or _max_w is not None:
                    _wavelength_bounds = [_min_w, _max_w]
            _snapshot["wavelength_bounds"] = _wavelength_bounds

            _consensus_config_keys = [
                "constrain_consensus",
                "consensus_method",
                "consensus_sigma_clip",
                "consensus_sigma",
                "consensus_tolerance",
                "consensus_max_harmonic",
                "min_consensus_inliers",
                "use_gp_validation",
            ]
            _consensus_configuration = {
                _k: _resolved.get(_k) for _k in _consensus_config_keys
            }
            if any(_v is not None for _v in _consensus_configuration.values()):
                _snapshot["consensus_configuration"] = _consensus_configuration

            _outlier_threshold_keys = [
                "outlier_sigma_threshold",
                "outlier_threshold",
                "max_outlier_fraction",
            ]
            _outlier_thresholds = {
                _k: _resolved.get(_k) for _k in _outlier_threshold_keys
            }
            if any(_v is not None for _v in _outlier_thresholds.values()):
                _snapshot["outlier_thresholds"] = _outlier_thresholds

            _random_init_keys = [
                "use_mls_init",
                "use_best_band_init",
                "random_init",
                "use_random_init",
                "randomize_initialization",
            ]
            _random_flags = {_k: _resolved.get(_k) for _k in _random_init_keys}
            if any(_v is not None for _v in _random_flags.values()):
                _snapshot["random_initialization_flags"] = _random_flags

            _canonical_keys = set(_snapshot).union(
                {
                    "model",
                    "cuda",
                    "lr",
                    "optim",
                    "min_frequency",
                    "max_frequency",
                    "min_freq",
                    "max_freq",
                    "min_wavelength",
                    "max_wavelength",
                    "min_lambda",
                    "max_lambda",
                }
            )
            _user_kwargs = {
                _k: _v for _k, _v in _kwargs.items() if _k not in _canonical_keys
            }
            _snapshot["user_kwargs"] = _user_kwargs if _user_kwargs else None
        except Exception:
            pass

        try:
            return Lightcurve._sanitize_fit_configuration_value(_snapshot)
        except Exception:
            try:
                return {
                    _k: Lightcurve._sanitize_fit_configuration_value(_v)
                    for _k, _v in _snapshot.items()
                }
            except Exception:
                return None

    @staticmethod
    def _validate_fit_configuration_snapshot(snapshot):
        """Raise if the fit-configuration snapshot is not safe to store.

        This is a lightweight defensive check applied after
        :meth:`_collect_fit_configuration_snapshot` produces a snapshot and
        before it is embedded in a fit-history entry.  It is not a full schema
        validator; its purpose is to catch gross serialization failures early
        so that bugs surface as loud errors during development rather than
        silent corruption of history records.

        In production code the caller wraps this in a ``try/except`` so that
        a validation failure never prevents history from being recorded.

        Checks performed
        ----------------
        1. The snapshot is a non-empty ``dict``.
        2. A minimum set of required top-level keys is present.
        3. No raw tensors, ndarrays, non-finite floats, or callable objects
           remain anywhere in the nested structure.
        4. The snapshot is JSON-serializable via ``json.dumps(..., allow_nan=False)``.
        5. Explicit warnings are emitted for placeholder objects, truncation
           markers, non-finite replacements, and unusually deep nesting.

        Parameters
        ----------
        snapshot : object
            The value returned by
            :meth:`_collect_fit_configuration_snapshot`.

        Raises
        ------
        TypeError
            If ``snapshot`` is not a dict.
        RuntimeError
            If required keys are absent, prohibited types are found, or the
            snapshot is not JSON-serializable.
        """
        _REQUIRED_KEYS = frozenset(
            {
                "fit_strategy",
                "model_class",
                "training_iter",
                "backend",
                "user_kwargs",
            }
        )
        if not isinstance(snapshot, dict):
            raise TypeError(
                f"fit_configuration snapshot must be a dict, "
                f"got {type(snapshot).__name__!r}"
            )
        _missing = _REQUIRED_KEYS - snapshot.keys()
        if _missing:
            raise RuntimeError(
                f"fit_configuration snapshot is missing required keys: "
                f"{sorted(_missing)}"
            )

        def _warn(message):
            warnings.warn(message, UserWarning, stacklevel=2)

        def _walk(value, path="fit_configuration", depth=0):
            if torch.is_tensor(value):
                raise RuntimeError(
                    f"fit_configuration snapshot contains a raw tensor at "
                    f"{path}; sanitize first"
                )
            if isinstance(value, np.ndarray):
                raise RuntimeError(
                    f"fit_configuration snapshot contains a raw ndarray at "
                    f"{path}; sanitize first"
                )
            if callable(value):
                raise RuntimeError(
                    f"fit_configuration snapshot contains a callable at "
                    f"{path}; store a structured placeholder instead"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(
                    f"fit_configuration snapshot contains a non-finite float at "
                    f"{path}; sanitize first"
                )
            if depth == 7 and isinstance(value, dict | list):
                _warn(
                    "fit_configuration snapshot is deeply nested at "
                    f"{path}; review whether this structure is scientifically "
                    "necessary for provenance."
                )
            if isinstance(value, dict):
                if value.get("__unserializable__") is True:
                    _warn(
                        "fit_configuration snapshot contains an unsupported "
                        f"object placeholder at {path} "
                        f"({value.get('module')}.{value.get('type')})."
                    )
                if value.get("__truncated__") is True:
                    _warn(
                        "fit_configuration snapshot contains truncated data at "
                        f"{path} ({value.get('type')})."
                    )
                if value.get("__non_finite__") is True:
                    _warn(
                        "fit_configuration snapshot replaced a non-finite value "
                        f"at {path} ({value.get('value')})."
                    )
                for _key, _val in value.items():
                    _child_path = f"{path}.{_key}"
                    _walk(_val, path=_child_path, depth=depth + 1)
                return
            if isinstance(value, list):
                for _idx, _val in enumerate(value):
                    _walk(_val, path=f"{path}[{_idx}]", depth=depth + 1)

        _walk(snapshot)
        try:
            json.dumps(snapshot, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"fit_configuration snapshot is not JSON-serializable: {exc}"
            ) from exc

    @staticmethod
    def _fit_history_package_directory():
        """Return the installed package directory used for provenance lookup."""
        try:
            return Path(__file__).resolve().parent
        except Exception:
            return None

    @staticmethod
    def _collect_git_provenance():
        """Return best-effort git provenance for the installed package."""
        provenance = {
            "git_commit_hash": None,
            "git_branch": None,
            "git_dirty_worktree": None,
            "git_remote_url": None,
        }

        def _run_git_command(args, cwd):
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
            except Exception:
                return None
            if result.returncode != 0:
                return None
            return result.stdout.strip() or None

        try:
            package_dir = Lightcurve._fit_history_package_directory()
            if package_dir is None:
                return provenance

            repo_root = _run_git_command(["rev-parse", "--show-toplevel"], package_dir)
            if repo_root is None:
                return provenance

            repo_root_path = Path(repo_root)
            provenance["git_commit_hash"] = _run_git_command(
                ["rev-parse", "HEAD"],
                repo_root_path,
            )
            provenance["git_branch"] = _run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                repo_root_path,
            )
            _dirty = _run_git_command(
                ["status", "--porcelain"],
                repo_root_path,
            )
            provenance["git_dirty_worktree"] = (
                bool(_dirty) if _dirty is not None else None
            )
            provenance["git_remote_url"] = _run_git_command(
                ["config", "--get", "remote.origin.url"],
                repo_root_path,
            )
        except Exception:
            return provenance
        return provenance

    @staticmethod
    def _collect_rng_provenance():
        """Return best-effort RNG-state and determinism provenance.

        Canonical field names:
        ``numpy_rng_state_token``, ``python_rng_state_token``,
        and ``torch_initial_seed``.

        Backward-compatible aliases:
        ``numpy_random_seed``, ``python_random_seed``,
        and ``torch_random_seed``.

        NumPy and Python's stdlib random module do not generally expose the
        original user-provided seed once the RNG has advanced, so the NumPy and
        Python values recorded here are current RNG-state tokens rather than
        guaranteed original seeds.
        """
        provenance = {
            "numpy_rng_state_token": None,
            "numpy_random_seed": None,
            "torch_initial_seed": None,
            "torch_random_seed": None,
            "python_rng_state_token": None,
            "python_random_seed": None,
            "torch_deterministic_algorithms": None,
            "torch_cudnn_deterministic": None,
            "torch_cudnn_benchmark": None,
        }

        try:
            _np_state = np.random.get_state()
            if len(_np_state) > 1 and len(_np_state[1]) > 0:
                _numpy_state_token = int(_np_state[1][0])
                provenance["numpy_rng_state_token"] = _numpy_state_token
                provenance["numpy_random_seed"] = _numpy_state_token
        except Exception:
            pass

        try:
            _py_state = random.getstate()
            if len(_py_state) > 1 and len(_py_state[1]) > 0:
                _python_state_token = int(_py_state[1][0])
                provenance["python_rng_state_token"] = _python_state_token
                provenance["python_random_seed"] = _python_state_token
        except Exception:
            pass

        try:
            _torch_initial_seed = int(torch.initial_seed())
            provenance["torch_initial_seed"] = _torch_initial_seed
            provenance["torch_random_seed"] = _torch_initial_seed
        except Exception:
            pass

        try:
            provenance["torch_deterministic_algorithms"] = bool(
                torch.are_deterministic_algorithms_enabled()
            )
        except Exception:
            pass

        try:
            provenance["torch_cudnn_deterministic"] = bool(
                torch.backends.cudnn.deterministic
            )
        except Exception:
            pass

        try:
            provenance["torch_cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        except Exception:
            pass

        return provenance

    @staticmethod
    def _collect_environment_provenance():
        """Return consolidated environment provenance for fit-history entries."""
        env = {"python_version": sys.version.split()[0]}
        try:
            from . import __version__ as _pgmuvi_version
        except Exception:
            _pgmuvi_version = None
        env["pgmuvi_version"] = _pgmuvi_version

        if "torch" in globals():
            env["torch_version"] = getattr(torch, "__version__", None)
        else:
            env["torch_version"] = None

        if "gpytorch" in globals():
            env["gpytorch_version"] = getattr(gpytorch, "__version__", None)
        else:
            env["gpytorch_version"] = None

        env["git"] = Lightcurve._collect_git_provenance()
        env["rng"] = Lightcurve._collect_rng_provenance()
        return env

    @staticmethod
    def _fit_history_environment_metadata():
        """Return lightweight environment provenance for fit-history entries."""
        return Lightcurve._sanitize_fit_history_value(
            Lightcurve._collect_environment_provenance()
        )

    def _append_fit_history(
        self,
        *,
        timestamp_utc=None,
        model_class=None,
        fit_strategy=None,
        success=None,
        failed=None,
        exception_type=None,
        exception_message=None,
        training_iter=None,
        num_mixtures=None,
        elapsed_seconds=None,
        backend=None,
        constrained=None,
        constrained_fit=None,
        constraint_set=None,
        bands=None,
        uses_frequency_space=None,
        uses_period_space=None,
        fit_configuration=None,
        environment=None,
        notes=None,
    ):
        """Append a JSON-safe fit-history entry.

        This helper is intentionally defensive and must never raise.

        Parameters
        ----------
        bands : list of str, optional
            Unique band labels present in the lightcurve at fit time.
        constrained_fit : bool, optional
            Explicit flag for whether the fit used a constraint set.
            If not provided, falls back to ``constrained``.
        constraint_set : str, optional
            Name or description of the constraint set used, if any.
        uses_frequency_space : bool, optional
            Whether the model is parameterized in frequency space.
        uses_period_space : bool, optional
            Whether the model is parameterized in period space.
        """
        try:
            if not hasattr(self, "fit_history") or not isinstance(
                self.fit_history, list
            ):
                self.fit_history = []

            _context = getattr(self, "_fit_history_context", {})
            if not isinstance(_context, dict):
                _context = {}

            if timestamp_utc is None:
                timestamp_utc = datetime.datetime.now(
                    datetime.UTC
                ).isoformat()

            _constrained_resolved = (
                constrained
                if constrained is not None
                else _context.get("constrained")
            )
            # constrained_fit is a structured alias for constrained, falling
            # back to the ``constrained`` value if not explicitly provided.
            _constrained_fit_resolved = (
                constrained_fit
                if constrained_fit is not None
                else _context.get("constrained_fit", _constrained_resolved)
            )

            _entry = {
                "fit_history_schema_version": _FIT_HISTORY_SCHEMA_VERSION,
                "timestamp_utc": timestamp_utc,
                "model_class": (
                    model_class
                    if model_class is not None
                    else _context.get("model_class")
                ),
                "fit_strategy": (
                    fit_strategy
                    if fit_strategy is not None
                    else _context.get("fit_strategy")
                ),
                "success": success,
                "failed": failed,
                "exception_type": exception_type,
                "exception_message": exception_message,
                "training_iter": (
                    training_iter
                    if training_iter is not None
                    else _context.get("training_iter")
                ),
                "num_mixtures": (
                    num_mixtures
                    if num_mixtures is not None
                    else _context.get("num_mixtures")
                ),
                "elapsed_seconds": elapsed_seconds,
                "backend": (
                    backend if backend is not None else _context.get("backend")
                ),
                "constrained": _constrained_resolved,
                "constrained_fit": _constrained_fit_resolved,
                "constraint_set": (
                    constraint_set
                    if constraint_set is not None
                    else _context.get("constraint_set")
                ),
                "bands": (
                    bands if bands is not None else _context.get("bands")
                ),
                "uses_frequency_space": (
                    uses_frequency_space
                    if uses_frequency_space is not None
                    else _context.get("uses_frequency_space")
                ),
                "uses_period_space": (
                    uses_period_space
                    if uses_period_space is not None
                    else _context.get("uses_period_space")
                ),
                "fit_configuration": (
                    fit_configuration
                    if fit_configuration is not None
                    else _context.get("fit_configuration")
                ),
                "environment": (
                    environment
                    if environment is not None
                    else _context.get(
                        "environment", self._fit_history_environment_metadata()
                    )
                ),
                "notes": notes,
            }

            _entry = {
                key: self._sanitize_fit_history_value(val)
                for key, val in _entry.items()
            }
            self._validate_fit_history_entry(_entry)
            self.fit_history.append(_entry)
        except Exception:
            return

    @staticmethod
    def _validate_fit_history_entry(entry):
        """Sanitize a fit-history entry dict in-place; never raises.

        Coerces field values to the expected types when possible.  Invalid
        values are replaced conservatively (``None`` or empty list) rather
        than raising.  This lets manually constructed or legacy entries be
        consumed safely by :meth:`get_fit_history_summary`.

        Parameters
        ----------
        entry : dict
            A fit-history record to sanitize.

        Returns
        -------
        dict
            The same dict, mutated in-place and returned.
        """
        if not isinstance(entry, dict):
            return entry
        try:
            # elapsed_seconds: must be a finite float or None.
            _elapsed = entry.get("elapsed_seconds")
            if _elapsed is not None:
                try:
                    _f = float(_elapsed)
                    entry["elapsed_seconds"] = (
                        None if not math.isfinite(_f) else _f
                    )
                except (TypeError, ValueError):
                    entry["elapsed_seconds"] = None

            # constrained_fit: coerce truthy/falsy values to bool.
            _cf = entry.get("constrained_fit")
            if _cf is not None and not isinstance(_cf, bool):
                try:
                    entry["constrained_fit"] = bool(_cf)
                except (TypeError, ValueError):
                    entry["constrained_fit"] = None

            # bands: normalize to a list of unique strings (insertion order).
            _bands = entry.get("bands")
            if _bands is not None:
                if not isinstance(_bands, list):
                    try:
                        _bands = list(_bands)
                    except (TypeError, ValueError):
                        _bands = []
                _seen: set[str] = set()
                _normalized: list[str] = []
                for _b in _bands:
                    try:
                        _bs = str(_b)
                        if _bs not in _seen:
                            _seen.add(_bs)
                            _normalized.append(_bs)
                    except Exception:
                        pass
                entry["bands"] = _normalized

            # timestamp_utc: coerce to string if not already one.
            _ts = entry.get("timestamp_utc")
            if _ts is not None and not isinstance(_ts, str):
                try:
                    entry["timestamp_utc"] = str(_ts)
                except Exception:
                    entry["timestamp_utc"] = None

            # fit_configuration: ensure compact JSON-safe dict or None.
            # _validate_fit_configuration_snapshot is called defensively;
            # any validation failure is silenced here because history
            # recording must never raise.
            _cfg = entry.get("fit_configuration")
            if _cfg is not None:
                _cfg_sanitized = Lightcurve._sanitize_fit_configuration_value(
                    _cfg
                )
                entry["fit_configuration"] = _cfg_sanitized
                try:
                    Lightcurve._validate_fit_configuration_snapshot(
                        _cfg_sanitized
                    )
                except (TypeError, RuntimeError):
                    pass
        except Exception:
            pass
        return entry

    @staticmethod
    def _normalize_imported_fit_history_entry(entry):
        """Return a sanitized copy of an imported fit-history entry.

        Parameters
        ----------
        entry : dict
            Candidate imported history entry.

        Returns
        -------
        dict or None
            Sanitized entry dict, or ``None`` if the input is not a valid
            entry-shaped mapping.
        """
        if not isinstance(entry, dict):
            return None
        _entry_copy = copy.deepcopy(entry)
        _normalized = {
            key: Lightcurve._sanitize_fit_history_value(value)
            for key, value in _entry_copy.items()
        }

        _raw_version = _normalized.get("fit_history_schema_version")
        _version = None
        if _raw_version is not None:
            try:
                _candidate = int(_raw_version)
                if _candidate > 0:
                    _version = _candidate
            except (TypeError, ValueError):
                _version = None
        if _version is None:
            _version = 1
        _normalized["fit_history_schema_version"] = _version

        # Ensure common keys exist in legacy entries so downstream summary and
        # reporting paths are stable.
        _normalized.setdefault("timestamp_utc", None)
        _normalized.setdefault("success", None)
        _normalized.setdefault("failed", None)
        _normalized.setdefault("fit_configuration", None)
        _normalized.setdefault("environment", None)
        _normalized.setdefault("notes", None)

        Lightcurve._validate_fit_history_entry(_normalized)
        return _normalized

    def _validate_imported_fit_history(self, payload):
        """Validate imported fit-history payload and return normalized entries.

        Parameters
        ----------
        payload : dict or list
            Parsed JSON payload produced by :meth:`export_fit_history_json`,
            or a legacy raw list of fit-history entries.

        Returns
        -------
        tuple
            ``(entries, diagnostics)`` where ``entries`` is a normalized list
            of fit-history dicts and ``diagnostics`` is a list of warning
            strings collected during validation.
        """
        _diagnostics = []
        _raw_entries = None

        if isinstance(payload, dict):
            _raw_entries = payload.get("fit_history", [])
            _supported_versions = payload.get("fit_history_supported_schema_versions")
            if _supported_versions is not None and not isinstance(
                _supported_versions, list
            ):
                _diagnostics.append(
                    "fit_history_supported_schema_versions is malformed; ignoring."
                )
        elif isinstance(payload, list):
            _raw_entries = payload
        else:
            _diagnostics.append(
                "Imported payload is neither a dict nor a list; nothing loaded."
            )
            return [], _diagnostics

        if not isinstance(_raw_entries, list):
            _diagnostics.append("fit_history is not a list; nothing loaded.")
            return [], _diagnostics

        _normalized_entries = []
        for _idx, _entry in enumerate(_raw_entries):
            _normalized = self._normalize_imported_fit_history_entry(_entry)
            if _normalized is None:
                _diagnostics.append(
                    f"Skipping malformed entry at index {_idx}: not a mapping."
                )
                continue
            _entry_version = _normalized.get("fit_history_schema_version")
            if _entry_version not in _FIT_HISTORY_SUPPORTED_SCHEMA_VERSIONS:
                _diagnostics.append(
                    "Entry at index "
                    f"{_idx} has schema version {_entry_version}; importing "
                    "with best-effort normalization."
                )
            _normalized_entries.append(_normalized)

        return _normalized_entries, _diagnostics

    def export_fit_history_json(self, path, include_text_summary=True):
        """Export fit-history provenance to a human-readable JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path.
        include_text_summary : bool, optional
            If ``True``, include plain-text summary blocks in the exported
            payload.
        """
        _path = Path(path)
        _history = self.get_fit_history()
        _normalized_history = []
        _diagnostics = []
        for _idx, _entry in enumerate(_history):
            _normalized = self._normalize_imported_fit_history_entry(_entry)
            if _normalized is None:
                _diagnostics.append(
                    f"Skipped malformed in-memory history entry at index {_idx}."
                )
                continue
            _normalized_history.append(_normalized)

        _payload = {
            "exported_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "fit_history_schema_version": _FIT_HISTORY_SCHEMA_VERSION,
            "fit_history_supported_schema_versions": sorted(
                _FIT_HISTORY_SUPPORTED_SCHEMA_VERSIONS
            ),
            "fit_history_entry_count": len(_normalized_history),
            "fit_history": _normalized_history,
            "fit_history_summary": self._sanitize_fit_history_value(
                self.get_fit_history_summary()
            ),
            "latest_fit_configuration": self._sanitize_fit_history_value(
                self.get_last_fit_configuration()
            ),
        }
        if _diagnostics:
            _payload["export_warnings"] = _diagnostics
        if include_text_summary:
            _payload["fit_history_text_summary"] = self.fit_history_to_text()
            _payload["fit_history_summary_text"] = self.print_fit_history_summary(
                print_summary=False
            )
            _payload["latest_fit_configuration_text"] = self.fit_configuration_to_text(
                fit_configuration=self.get_last_fit_configuration()
            )

        _payload = self._sanitize_fit_history_value(_payload)
        _path.parent.mkdir(parents=True, exist_ok=True)
        with _path.open("w", encoding="utf-8") as _f:
            json.dump(_payload, _f, ensure_ascii=False, indent=2, sort_keys=True)

    def load_fit_history_json(self, path, merge=False):
        """Load fit-history provenance exported as JSON.

        Parameters
        ----------
        path : str or pathlib.Path
            Source JSON file path.
        merge : bool, optional
            If ``False`` (default), replace current history with imported
            entries. If ``True``, append imported entries.

        Returns
        -------
        int
            Number of imported entries accepted into history.
        """
        _path = Path(path)
        with _path.open("r", encoding="utf-8") as _f:
            _payload = json.load(_f)

        _entries, _diagnostics = self._validate_imported_fit_history(_payload)
        for _message in _diagnostics:
            warnings.warn(_message, UserWarning, stacklevel=2)

        if merge:
            _current = self.get_fit_history()
            _merged = [copy.deepcopy(_entry) for _entry in _current]
            _merged.extend(copy.deepcopy(_entry) for _entry in _entries)
            self.fit_history = _merged
        else:
            self.fit_history = [copy.deepcopy(_entry) for _entry in _entries]

        return len(_entries)

    def get_fit_history(self):
        """Return a deep copy of fit-history entries."""
        if not hasattr(self, "fit_history") or not isinstance(self.fit_history, list):
            return []
        return copy.deepcopy(self.fit_history)

    def clear_fit_history(self):
        """Clear all fit-history entries."""
        if not hasattr(self, "fit_history") or not isinstance(self.fit_history, list):
            self.fit_history = []
            return
        self.fit_history.clear()

    def get_fit_history_summary(self):
        """Return aggregate statistics for fit-history entries.

        Returns a dict suitable for JSON serialisation.  All scalar counts
        are plain Python ``int`` or ``float``; timestamps are ISO-8601
        strings or ``None``.

        The returned dict includes both a nested structure for backward
        compatibility (``counts_by_parameterization``, ``counts_by_constraint_mode``,
        ``unique_bands_used``) and flat convenience keys
        (``frequency_space_attempts``, ``period_space_attempts``,
        ``constrained_fits``, ``unconstrained_fits``, ``unique_bands``,
        ``counts_by_fit_strategy``) added to support the enhanced summary
        specification.

        Missing fields in older history entries are silently ignored; such
        entries contribute to ``total_attempts`` but not to per-category
        counts where the relevant field is absent.
        """
        _history = self.get_fit_history()
        _total = len(_history)
        _successful = sum(1 for entry in _history if entry.get("success") is True)
        _failed = sum(1 for entry in _history if entry.get("failed") is True)
        _success_fraction = (_successful / _total) if _total > 0 else 0.0

        _last_success_ts = None
        _last_failure_ts = None
        _earliest_ts = None
        _latest_ts = None
        _runtime_seconds = []
        _counts_by_backend = {}
        _counts_by_model_class = {}
        _counts_by_fit_strategy = {}
        _counts_by_parameterization = {
            "frequency_space": 0,
            "period_space": 0,
            "unknown": 0,
        }
        _counts_by_constraint_mode = {
            "constrained": 0,
            "unconstrained": 0,
            "unknown": 0,
        }
        for entry in _history:
            _ts = entry.get("timestamp_utc")
            if entry.get("success") is True:
                _last_success_ts = _ts
            if entry.get("failed") is True:
                _last_failure_ts = _ts

            if isinstance(_ts, str):
                try:
                    _parsed = datetime.datetime.fromisoformat(_ts)
                except ValueError:
                    _parsed = None
                if _parsed is not None:
                    if _earliest_ts is None or _parsed < _earliest_ts:
                        _earliest_ts = _parsed
                    if _latest_ts is None or _parsed > _latest_ts:
                        _latest_ts = _parsed

            _elapsed = entry.get("elapsed_seconds")
            if isinstance(_elapsed, int | float) and math.isfinite(float(_elapsed)):
                _runtime_seconds.append(float(_elapsed))

            _backend = str(entry.get("backend") or "unknown")
            _counts_by_backend[_backend] = _counts_by_backend.get(_backend, 0) + 1

            _model_class = str(entry.get("model_class") or "unknown")
            _counts_by_model_class[_model_class] = (
                _counts_by_model_class.get(_model_class, 0) + 1
            )

            # fit_strategy counts — entries without a strategy go under "unknown"
            _fit_strategy_key = str(entry.get("fit_strategy") or "unknown")
            _counts_by_fit_strategy[_fit_strategy_key] = (
                _counts_by_fit_strategy.get(_fit_strategy_key, 0) + 1
            )

            # Parameterization space: prefer the explicit flag stored in the
            # entry; fall back to inference from model class / fit strategy for
            # older entries that predate the ``uses_frequency_space`` field.
            _uses_freq = entry.get("uses_frequency_space")
            _uses_period = entry.get("uses_period_space")
            if _uses_freq is True:
                _param_space = "frequency_space"
            elif _uses_period is True:
                _param_space = "period_space"
            else:
                # Inference from model class / fit strategy for legacy entries
                _model_l = _model_class.lower()
                _fit_strategy_l = _fit_strategy_key.lower()
                if (
                    "spectralmixture" in _model_l
                    or "separable" in _model_l
                    or _fit_strategy_l == "consensus"
                ):
                    _param_space = "frequency_space"
                elif "periodic" in _model_l or "quasiperiodic" in _model_l:
                    _param_space = "period_space"
                else:
                    _param_space = "unknown"
            _counts_by_parameterization[_param_space] += 1

            # Constraint mode: ``constrained_fit`` takes precedence over the
            # older ``constrained`` field so that both legacy and new entries
            # are handled correctly.
            _constrained = entry.get("constrained_fit")
            if _constrained is None:
                _constrained = entry.get("constrained")
            if _constrained is True:
                _counts_by_constraint_mode["constrained"] += 1
            elif _constrained is False:
                _counts_by_constraint_mode["unconstrained"] += 1
            else:
                _counts_by_constraint_mode["unknown"] += 1

        _total_runtime_seconds = float(sum(_runtime_seconds))
        _mean_runtime_seconds: float | None = (
            _total_runtime_seconds / len(_runtime_seconds)
            if _runtime_seconds
            else None
        )

        # Collect unique bands from history entries; entries that predate the
        # ``bands`` field contribute nothing to the set.  If no history entry
        # recorded band information, fall back to the current self.band array.
        _bands_from_history: set[str] = set()
        for _he in _history:
            _he_bands = _he.get("bands")
            if isinstance(_he_bands, list):
                for _hb in _he_bands:
                    if _hb is not None:
                        try:
                            _bands_from_history.add(str(_hb))
                        except Exception:
                            pass
        if _bands_from_history:
            _unique_bands = sorted(_bands_from_history)
        elif self.band is not None:
            _unique_bands = sorted(
                {str(b) for b in np.asarray(self.band, dtype=np.str_)}
            )
        else:
            _unique_bands = []

        return {
            "total_attempts": _total,
            "successful_fits": _successful,
            "failed_fits": _failed,
            "success_fraction": _success_fraction,
            "last_success_timestamp": _last_success_ts,
            "last_failure_timestamp": _last_failure_ts,
            "counts_by_backend": _counts_by_backend,
            "counts_by_model_class": _counts_by_model_class,
            # counts_by_fit_strategy is new; absent in earlier summaries
            "counts_by_fit_strategy": _counts_by_fit_strategy,
            "counts_by_parameterization": _counts_by_parameterization,
            "counts_by_constraint_mode": _counts_by_constraint_mode,
            "total_runtime_seconds": _total_runtime_seconds,
            "mean_runtime_seconds": _mean_runtime_seconds,
            "earliest_timestamp": (
                _earliest_ts.isoformat() if _earliest_ts is not None else None
            ),
            "latest_timestamp": (
                _latest_ts.isoformat() if _latest_ts is not None else None
            ),
            # ``unique_bands_used`` is the canonical name; ``unique_bands`` is
            # a flat alias included for the enhanced summary specification.
            "unique_bands_used": _unique_bands,
            "unique_bands": _unique_bands,
            # Flat convenience scalars derived from the nested dicts above.
            # Kept consistent with ``counts_by_parameterization`` so callers
            # can use whichever form they prefer.
            "frequency_space_attempts": (
                _counts_by_parameterization["frequency_space"]
            ),
            "period_space_attempts": _counts_by_parameterization["period_space"],
            "constrained_fits": _counts_by_constraint_mode["constrained"],
            "unconstrained_fits": _counts_by_constraint_mode["unconstrained"],
        }

    @staticmethod
    def _fit_history_abbreviate_message(message, max_len=56):
        """Return a compact one-line failure reason for table display."""
        if message is None:
            return ""
        text = " ".join(str(message).split())
        if len(text) <= max_len:
            return text
        return f"{text[: max_len - 1]}…"

    def fit_history_to_text(
        self,
        max_entries=None,
        success_only=False,
        failed_only=False,
    ):
        """Return a human-readable table of fit-history entries.

        This helper is designed for notebook usage and concise run logs,
        providing a quick provenance trail for scientific reproducibility.
        """
        if success_only and failed_only:
            raise ValueError("success_only and failed_only cannot both be True.")
        if max_entries is not None:
            if isinstance(max_entries, bool) or not isinstance(
                max_entries, (int, np.integer)
            ):
                raise ValueError("max_entries must be an integer or None.")
            if int(max_entries) < 1:
                raise ValueError("max_entries must be >= 1 when provided.")

        history = self.get_fit_history()
        if success_only:
            history = [entry for entry in history if entry.get("success") is True]
        if failed_only:
            history = [entry for entry in history if entry.get("failed") is True]
        if max_entries is not None:
            history = history[-int(max_entries) :]

        if not history:
            return "No fit-history entries."

        timestamp_w = 19
        model_w = max(
            12,
            min(
                28,
                max(
                    len(str(entry.get("model_class") or "N/A"))
                    for entry in history
                ),
            ),
        )
        success_w = 7
        runtime_w = 10
        reason_w = 28

        header = (
            f"{'#':>3}  {'Timestamp':<{timestamp_w}}  {'Model':<{model_w}}  "
            f"{'Success':<{success_w}}  {'Runtime(s)':>{runtime_w}}  "
            f"{'Failure reason':<{reason_w}}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for idx, entry in enumerate(history, start=1):
            timestamp = str(entry.get("timestamp_utc") or "N/A")[:timestamp_w]
            model = str(entry.get("model_class") or "N/A")
            model = model[:model_w]

            success_val = entry.get("success")
            if success_val is True:
                success_str = "True"
            elif success_val is False:
                success_str = "False"
            else:
                success_str = "N/A"

            elapsed = entry.get("elapsed_seconds")
            if isinstance(elapsed, int | float) and math.isfinite(float(elapsed)):
                runtime_str = f"{float(elapsed):.3g}"
            else:
                runtime_str = "N/A"

            failure_reason = ""
            if entry.get("failed") is True:
                failure_reason = self._fit_history_abbreviate_message(
                    entry.get("exception_message")
                )
            failure_reason = failure_reason[:reason_w]

            lines.append(
                f"{idx:>3}  {timestamp:<{timestamp_w}}  {model:<{model_w}}  "
                f"{success_str:<{success_w}}  {runtime_str:>{runtime_w}}  "
                f"{failure_reason:<{reason_w}}"
            )

        lines.append(sep)
        return "\n".join(lines)

    def get_last_fit_configuration(self):
        """Return a deep copy of the most recent fit-configuration snapshot.

        The snapshot is the dict stored under the ``"fit_configuration"`` key
        of the last fit-history entry.  It captures the fit strategy, model
        class, training hyperparameters, and user-supplied kwargs as they
        were resolved at the start of the last :meth:`fit` call.

        See :meth:`_collect_fit_configuration_snapshot` for the full schema
        of the returned dict.

        A deep copy is returned so that callers cannot accidentally mutate
        the stored history entry.

        Returns
        -------
        dict or None
            A deep copy of the most recent fit-configuration dict, or
            ``None`` if no fit history exists or the last entry does not
            contain a configuration record.

        See Also
        --------
        fit_configuration_to_text : Human-readable rendering of the snapshot.
        get_fit_history : Full raw history list.
        """
        _history = self.get_fit_history()
        if not _history:
            return None
        _cfg = _history[-1].get("fit_configuration")
        if _cfg is None:
            return None
        return copy.deepcopy(_cfg)

    @staticmethod
    def _fit_configuration_display_value(value):
        """Return a concise scalar display string for fit-configuration text."""
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, float):
            return f"{value:.6g}" if math.isfinite(value) else "None"
        if isinstance(value, int | str):
            return str(value)
        if isinstance(value, dict):
            if value.get("__unserializable__") is True:
                _module = value.get("module") or "unknown"
                _type = value.get("type") or "object"
                _qualname = value.get("qualname")
                _repr = value.get("repr")
                _head = f"[UNSERIALIZABLE] {_module}.{_type}"
                if _qualname:
                    _head += f" ({_qualname})"
                if _repr:
                    _head += f"\nrepr: {_repr}"
                return _head
            if value.get("__non_finite__") is True:
                return f"[NON-FINITE] {value.get('value')}"
            if value.get("__truncated__") is True:
                _payload = {
                    _k: _v
                    for _k, _v in value.items()
                    if _k != "__truncated__"
                }
                return "[TRUNCATED]\n" + json.dumps(
                    _payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            if "shape" in value and ("type" in value or "dtype" in value):
                return json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return Lightcurve._fit_configuration_safe_repr(value)

    def fit_configuration_to_text(self, fit_configuration=None):
        """Return a multi-section, human-readable fit-configuration summary.

        Renders the fit-configuration snapshot as a formatted string split
        into three clearly labelled sections that mirror the separation
        maintained inside the snapshot:

        * **USER INPUTS** — non-canonical kwargs the caller passed explicitly
          to :meth:`fit` that do not map to any known internal key.
        * **RESOLVED INTERNAL SETTINGS** — the canonical values the library
          actually used: model class, fit strategy, training hyperparameters,
          constraint/prior sets, bounds.
        * **RUNTIME METADATA** — derived settings assembled internally:
          backend, consensus configuration, outlier thresholds,
          initialization flags.

        Output is stable-ordered and consistently indented so that it can be
        read comfortably in a Jupyter notebook or terminal session.  Long
        arrays and large dicts are summarized rather than dumped verbatim.

        Parameters
        ----------
        fit_configuration : dict, optional
            A snapshot dict to render.  When not provided, the most recent
            snapshot from fit history is used via
            :meth:`get_last_fit_configuration`.

        Returns
        -------
        str
            The formatted summary string.  Returns a short message when no
            configuration is available.

        See Also
        --------
        get_last_fit_configuration : Retrieve the raw snapshot dict.
        """
        _cfg = fit_configuration
        if _cfg is None:
            _cfg = self.get_last_fit_configuration()
        if _cfg is None:
            return "No fit configuration available."

        _cfg = self._sanitize_fit_configuration_value(_cfg)
        _sep = "-" * 50
        _lines = [_sep, "Fit Configuration", _sep]

        def _fv(val):
            return self._fit_configuration_display_value(val)

        def _section(title):
            _lines.append("")
            _lines.append(f"  {title}")
            _lines.append("  " + "-" * 48)

        def _row(label, val, indent=4):
            if val is None:
                return
            _text = _fv(val)
            if "\n" not in _text:
                _lines.append(f"{'':>{indent}}{label:<28}: {_text}")
                return
            _lines.append(f"{'':>{indent}}{label:<28}:")
            for _line in _text.splitlines():
                _lines.append(f"{'':>{indent + 2}}{_line}")

        def _dict_rows(label, dct, indent=4):
            """Append a labelled sub-section for a nested dict."""
            if not isinstance(dct, dict):
                return
            _visible = {k: v for k, v in dct.items() if v is not None}
            if not _visible:
                return
            _lines.append(f"{'':>{indent}}{label}:")
            for _k, _v in sorted(_visible.items()):
                _text = _fv(_v)
                if "\n" not in _text:
                    _lines.append(
                        f"{'':>{indent + 2}}{_k:<26}: {_text}"
                    )
                    continue
                _lines.append(f"{'':>{indent + 2}}{_k:<26}:")
                for _line in _text.splitlines():
                    _lines.append(f"{'':>{indent + 4}}{_line}")

        # ---- USER INPUTS ---------------------------------------------------
        _section("USER INPUTS")
        _user_kw = _cfg.get("user_kwargs")
        if isinstance(_user_kw, dict) and _user_kw:
            for _k, _v in sorted(_user_kw.items()):
                _row(_k, _v)
        else:
            _lines.append("    (no non-canonical user kwargs recorded)")

        # ---- RESOLVED INTERNAL SETTINGS ------------------------------------
        _section("RESOLVED INTERNAL SETTINGS")
        _row("fit_strategy", _cfg.get("fit_strategy"))
        _row("model_class", _cfg.get("model_class"))
        _row("training_iter", _cfg.get("training_iter"))
        _row("num_mixtures", _cfg.get("num_mixtures"))
        _row("learning_rate", _cfg.get("learning_rate"))
        _row("optimizer", _cfg.get("optimizer"))
        _row("constraint_set", _cfg.get("constraint_set"))
        _row("prior_set", _cfg.get("prior_set"))
        _row("use_best_band_init", _cfg.get("use_best_band_init"))
        _row("use_gp_validation", _cfg.get("use_gp_validation"))
        _row("min_period", _cfg.get("min_period"))
        _row("max_period", _cfg.get("max_period"))
        _row("frequency_bounds", _cfg.get("frequency_bounds"))
        _row("period_bounds", _cfg.get("period_bounds"))
        _row("wavelength_bounds", _cfg.get("wavelength_bounds"))
        _row("min_consensus_inliers", _cfg.get("min_consensus_inliers"))
        _row("normalize", _cfg.get("normalize"))
        _row("detrend", _cfg.get("detrend"))
        _row("xtransform", _cfg.get("xtransform"))
        _row("ytransform", _cfg.get("ytransform"))
        _row("max_samples", _cfg.get("max_samples"))
        _row("max_samples_per_band", _cfg.get("max_samples_per_band"))

        # ---- RUNTIME METADATA ----------------------------------------------
        _section("RUNTIME METADATA")
        _row("backend", _cfg.get("backend"))
        _dict_rows(
            "consensus_configuration",
            _cfg.get("consensus_configuration"),
        )
        _dict_rows("outlier_thresholds", _cfg.get("outlier_thresholds"))
        _rand_flags = _cfg.get("random_initialization_flags")
        _dict_rows("random_initialization_flags", _rand_flags)

        _lines.append("")
        _lines.append(_sep)
        return "\n".join(_lines)

    def print_fit_history(
        self,
        max_entries=None,
        success_only=False,
        failed_only=False,
    ):
        """Print a formatted fit-history table and return the rendered text."""
        text = self.fit_history_to_text(
            max_entries=max_entries,
            success_only=success_only,
            failed_only=failed_only,
        )
        print(text)
        return text

    def print_fit_history_summary(
        self,
        print_summary=True,
        indent=2,
    ):
        """Return (and optionally print) a human-readable fit-history summary.

        This method calls :meth:`get_fit_history_summary` internally and
        formats the result as a multi-line textual report suitable for
        notebook output.  The report aligns category counts in columns and
        handles all missing or ``None`` values gracefully.

        Parameters
        ----------
        print_summary : bool, optional
            If ``True`` (the default), the formatted string is printed to
            stdout in addition to being returned.
        indent : int, optional
            Number of spaces to use for indented lines (default 2).

        Returns
        -------
        str
            The formatted summary string.
        """
        _s = self.get_fit_history_summary()
        _pad = " " * max(0, int(indent))
        _sep = "-" * 50
        _lines: list[str] = [_sep, "Fit History Summary", _sep, ""]

        # --- top-level counts -------------------------------------------
        _total = _s.get("total_attempts", 0)
        _ok = _s.get("successful_fits", 0)
        _fail = _s.get("failed_fits", 0)
        _lines.append(f"Total fits   : {_total}")
        _lines.append(f"  Successful : {_ok}")
        _lines.append(f"  Failed     : {_fail}")

        # --- runtime -------------------------------------------------------
        _tot_rt = _s.get("total_runtime_seconds")
        _mean_rt = _s.get("mean_runtime_seconds")
        _lines.append("")
        if isinstance(_tot_rt, int | float):
            _lines.append(f"Total runtime : {_tot_rt:.3g} s")
        else:
            _lines.append("Total runtime : N/A")
        if isinstance(_mean_rt, int | float):
            _lines.append(f"Mean runtime  : {_mean_rt:.3g} s")
        else:
            _lines.append("Mean runtime  : N/A")

        # --- timestamps ----------------------------------------------------
        _earliest = _s.get("earliest_timestamp")
        _latest = _s.get("latest_timestamp")
        _lines.append("")
        _lines.append("Time span:")
        _lines.append(
            f"{_pad}Earliest fit : {_earliest or 'N/A'}"
        )
        _lines.append(
            f"{_pad}Latest fit   : {_latest or 'N/A'}"
        )

        def _format_counts(section_title, counts_dict):
            """Append a left-aligned section of key : count rows."""
            _lines.append("")
            _lines.append(f"{section_title}:")
            if counts_dict:
                _max_key = max(len(str(k)) for k in counts_dict)
                for _k, _v in sorted(
                    counts_dict.items(), key=lambda kv: -kv[1]
                ):
                    _lines.append(
                        f"{_pad}{_k!s:<{_max_key}} : {_v}"
                    )
            else:
                _lines.append(f"{_pad}(none recorded)")

        _format_counts("Fit strategies", _s.get("counts_by_fit_strategy", {}))
        _format_counts("Backends", _s.get("counts_by_backend", {}))
        _format_counts("Models", _s.get("counts_by_model_class", {}))

        # --- parameterization -----------------------------------------------
        _pcounts = _s.get("counts_by_parameterization", {})
        _freq_n = _pcounts.get("frequency_space", 0)
        _per_n = _pcounts.get("period_space", 0)
        _lines.append("")
        _lines.append("Parameterization:")
        _lines.append(f"{_pad}Frequency-space fits : {_freq_n}")
        _lines.append(f"{_pad}Period-space fits    : {_per_n}")

        # --- constraints ---------------------------------------------------
        _ccounts = _s.get("counts_by_constraint_mode", {})
        _con_n = _ccounts.get("constrained", 0)
        _unc_n = _ccounts.get("unconstrained", 0)
        _unk_n = _ccounts.get("unknown", 0)
        _lines.append("")
        _lines.append("Constraints:")
        _lines.append(f"{_pad}Constrained fits   : {_con_n}")
        _lines.append(f"{_pad}Unconstrained fits : {_unc_n}")
        if _unk_n:
            _lines.append(f"{_pad}Unknown            : {_unk_n}")

        # --- bands ---------------------------------------------------------
        _bands = _s.get("unique_bands") or []
        _lines.append("")
        _lines.append("Bands encountered:")
        if _bands:
            for _band in _bands:
                _lines.append(f"{_pad}{_band}")
        else:
            _lines.append(f"{_pad}(none recorded)")

        _lines.append("")
        _lines.append(_sep)

        _text = "\n".join(_lines)
        if print_summary:
            print(_text)
        return _text

    def generate_reproducibility_report(
        self,
        latest_only=False,
        include_history=True,
        include_configurations=True,
        include_environment=True,
    ):
        """Return a long-form human-readable reproducibility report."""
        _sep = "-" * 50
        _lines = [_sep, "PGMUVI Reproducibility Report", _sep, ""]

        _history = self.get_fit_history()
        _working_history = list(_history)
        if latest_only:
            _working_history = _working_history[-1:] if _working_history else []

        _normalized_history = []
        _normalization_warnings = []
        for _idx, _entry in enumerate(_working_history):
            _normalized = self._normalize_imported_fit_history_entry(_entry)
            if _normalized is None:
                _normalization_warnings.append(
                    f"history[{_idx}] is malformed and was skipped."
                )
                continue
            _normalized_history.append(_normalized)

        try:
            _summary = self.get_fit_history_summary()
        except Exception:
            _summary = {
                "total_attempts": len(_normalized_history),
                "successful_fits": sum(
                    1 for _entry in _normalized_history if _entry.get("success") is True
                ),
                "failed_fits": sum(
                    1 for _entry in _normalized_history if _entry.get("failed") is True
                ),
                "counts_by_fit_strategy": {},
                "counts_by_model_class": {},
                "counts_by_backend": {},
                "total_runtime_seconds": 0.0,
                "mean_runtime_seconds": None,
                "unique_bands": [],
            }
            _normalization_warnings.append(
                "Could not compute canonical fit-history summary; using "
                "best-effort summary from normalized entries."
            )
            for _entry in _normalized_history:
                _strategy = str(_entry.get("fit_strategy") or "unknown")
                _summary["counts_by_fit_strategy"][_strategy] = (
                    _summary["counts_by_fit_strategy"].get(_strategy, 0) + 1
                )
                _model = str(_entry.get("model_class") or "unknown")
                _summary["counts_by_model_class"][_model] = (
                    _summary["counts_by_model_class"].get(_model, 0) + 1
                )
                _backend = str(_entry.get("backend") or "unknown")
                _summary["counts_by_backend"][_backend] = (
                    _summary["counts_by_backend"].get(_backend, 0) + 1
                )
                _elapsed = _entry.get("elapsed_seconds")
                if isinstance(_elapsed, int | float) and math.isfinite(float(_elapsed)):
                    _summary["total_runtime_seconds"] += float(_elapsed)
                _entry_bands = _entry.get("bands")
                if isinstance(_entry_bands, list):
                    for _band in _entry_bands:
                        if _band is not None:
                            _summary["unique_bands"].append(str(_band))
            if _summary["total_attempts"] > 0:
                _summary["mean_runtime_seconds"] = (
                    _summary["total_runtime_seconds"] / _summary["total_attempts"]
                )
            _summary["unique_bands"] = sorted(set(_summary["unique_bands"]))
        if latest_only:
            _total = len(_normalized_history)
            _successful = sum(
                1 for _entry in _normalized_history if _entry.get("success") is True
            )
            _failed = sum(
                1 for _entry in _normalized_history if _entry.get("failed") is True
            )
            _summary = dict(_summary)
            _summary["total_attempts"] = _total
            _summary["successful_fits"] = _successful
            _summary["failed_fits"] = _failed

            _counts_by_fit_strategy = {}
            _counts_by_model = {}
            _counts_by_backend = {}
            _runtime = []
            _bands = set()
            for _entry in _normalized_history:
                _strategy = str(_entry.get("fit_strategy") or "unknown")
                _counts_by_fit_strategy[_strategy] = (
                    _counts_by_fit_strategy.get(_strategy, 0) + 1
                )
                _model = str(_entry.get("model_class") or "unknown")
                _counts_by_model[_model] = _counts_by_model.get(_model, 0) + 1
                _backend = str(_entry.get("backend") or "unknown")
                _counts_by_backend[_backend] = _counts_by_backend.get(_backend, 0) + 1
                _elapsed = _entry.get("elapsed_seconds")
                if isinstance(_elapsed, int | float) and math.isfinite(float(_elapsed)):
                    _runtime.append(float(_elapsed))
                _entry_bands = _entry.get("bands")
                if isinstance(_entry_bands, list):
                    for _band in _entry_bands:
                        if _band is not None:
                            _bands.add(str(_band))

            _summary["counts_by_fit_strategy"] = _counts_by_fit_strategy
            _summary["counts_by_model_class"] = _counts_by_model
            _summary["counts_by_backend"] = _counts_by_backend
            _summary["total_runtime_seconds"] = float(sum(_runtime))
            _summary["mean_runtime_seconds"] = (
                float(sum(_runtime)) / len(_runtime) if _runtime else None
            )
            _summary["unique_bands"] = sorted(_bands)

        _latest_entry = _normalized_history[-1] if _normalized_history else None
        _latest_cfg = None
        if _latest_entry is not None:
            _latest_cfg = self._sanitize_fit_history_value(
                _latest_entry.get("fit_configuration")
            )
        if _latest_cfg is None:
            try:
                _latest_cfg = self.get_last_fit_configuration()
            except Exception:
                _latest_cfg = None

        if include_environment:
            _lines.append("Environment")
            _lines.append("-----------")
            _env = {}
            if _latest_entry is not None and isinstance(
                _latest_entry.get("environment"), dict
            ):
                _env = _latest_entry.get("environment", {})
            _git = _env.get("git") if isinstance(_env, dict) else None
            _python_version = (
                _env.get("python_version") if isinstance(_env, dict) else None
            )
            _torch_version = (
                _env.get("torch_version") if isinstance(_env, dict) else None
            )
            _git_commit = (
                _git.get("git_commit_hash") if isinstance(_git, dict) else None
            )
            _git_dirty = (
                _git.get("git_dirty_worktree") if isinstance(_git, dict) else None
            )
            _lines.append(f"Python version: {_python_version or 'N/A'}")
            _lines.append(f"Torch version: {_torch_version or 'N/A'}")
            _lines.append(f"Git commit: {_git_commit or 'N/A'}")
            _lines.append(
                "Git dirty tree: "
                + ("N/A" if _git_dirty is None else str(bool(_git_dirty)))
            )
            _lines.append("")

        _lines.append("Fit Summary")
        _lines.append("-----------")
        _lines.append(f"Total fits: {_summary.get('total_attempts', 0)}")
        _lines.append(f"Successful fits: {_summary.get('successful_fits', 0)}")
        _lines.append(f"Failed fits: {_summary.get('failed_fits', 0)}")
        _lines.append("")

        _lines.append("Strategies")
        _lines.append("----------")
        _strategy_counts = _summary.get("counts_by_fit_strategy", {}) or {}
        if _strategy_counts:
            for _key, _value in sorted(_strategy_counts.items()):
                _lines.append(f"{_key}: {_value}")
        else:
            _lines.append("(none recorded)")
        _lines.append("")

        _lines.append("Models")
        _lines.append("------")
        _model_counts = _summary.get("counts_by_model_class", {}) or {}
        if _model_counts:
            for _key, _value in sorted(_model_counts.items()):
                _lines.append(f"{_key}: {_value}")
        else:
            _lines.append("(none recorded)")
        _lines.append("")

        _lines.append("Runtime Statistics")
        _lines.append("------------------")
        _total_runtime = _summary.get("total_runtime_seconds")
        _mean_runtime = _summary.get("mean_runtime_seconds")
        if isinstance(_total_runtime, int | float):
            _lines.append(f"Total runtime (s): {float(_total_runtime):.6g}")
        else:
            _lines.append("Total runtime (s): N/A")
        if isinstance(_mean_runtime, int | float):
            _lines.append(f"Mean runtime (s): {float(_mean_runtime):.6g}")
        else:
            _lines.append("Mean runtime (s): N/A")
        _lines.append("")

        _lines.append("Bands")
        _lines.append("-----")
        _bands = _summary.get("unique_bands", []) or []
        if _bands:
            _lines.append(", ".join(str(_b) for _b in _bands))
        else:
            _lines.append("(none recorded)")
        _lines.append("")

        if include_configurations:
            _lines.append("Latest Fit Configuration")
            _lines.append("------------------------")
            if _latest_cfg is None:
                _lines.append("No fit configuration available.")
            else:
                _lines.append(
                    self.fit_configuration_to_text(fit_configuration=_latest_cfg)
                )
            _lines.append("")

        if include_history:
            _lines.append("History Summary")
            _lines.append("---------------")
            if _normalized_history:
                _lines.append(
                    self.fit_history_to_text(
                        max_entries=1 if latest_only else None,
                    )
                )
            else:
                _lines.append("No fit-history entries.")
            _lines.append("")

        _lines.append("Warnings / Truncations")
        _lines.append("----------------------")
        if _normalization_warnings:
            _lines.extend(_normalization_warnings)
        else:
            _lines.append("(none)")
        _lines.append(_sep)
        return "\n".join(_lines)

    def _fit_history_plot_annotation_text(self):
        """Return a compact provenance string for optional plot annotations."""
        history = self.get_fit_history()
        if not history:
            return ""
        entry = history[-1]
        model = str(entry.get("model_class") or "N/A")
        timestamp = str(entry.get("timestamp_utc") or "N/A")
        runtime = entry.get("elapsed_seconds")
        if isinstance(runtime, int | float) and math.isfinite(float(runtime)):
            runtime_str = f"{float(runtime):.3g}s"
        else:
            runtime_str = "N/A"
        return (
            f"model: {model}\n"
            f"runtime: {runtime_str}\n"
            f"timestamp: {timestamp}"
        )

    @staticmethod
    def _fit_history_provenance_position(provenance_location="lower left"):
        """Return axes-relative coordinates and alignment for provenance text."""
        _positions = {
            "lower left": (0.02, 0.02, "left", "bottom"),
            "lower right": (0.98, 0.02, "right", "bottom"),
            "upper left": (0.02, 0.98, "left", "top"),
            "upper right": (0.98, 0.98, "right", "top"),
        }
        if provenance_location not in _positions:
            raise ValueError(
                "provenance_location must be one of "
                "'lower left', 'lower right', 'upper left', 'upper right'."
            )
        return _positions[provenance_location]

    def _plot_fit_history_provenance(
        self,
        ax,
        *,
        provenance_location="lower left",
    ):
        """Annotate an axes with compact fit-history provenance, if available."""
        _prov = self._fit_history_plot_annotation_text()
        if not _prov:
            return None
        _x, _y, _ha, _va = self._fit_history_provenance_position(
            provenance_location=provenance_location
        )
        return ax.text(
            _x,
            _y,
            _prov,
            transform=ax.transAxes,
            ha=_ha,
            va=_va,
            fontsize=7,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    def _reset_fit_state(
        self,
        *,
        clear_failure=False,
        clear_model_state=False,
        clear_consensus=False,
    ):
        """Reset cached fit artifacts to prevent stale-state leakage."""
        self.__FITTED_MAP = False
        self.__FITTED_MCMC = False
        self.is_fitted = False
        self.gp_model = None

        for attr_name in (
            "results",
            "mcmc_results",
            "posterior_samples",
            "x_fine_transformed",
            "expanded_test_x",
            "consensus_failure_summary",
            "_period_summary_cache",
            "_last_consensus_fit_info",
            "optimizer",
        ):
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)

        if clear_consensus and hasattr(self, "consensus_diagnostics"):
            self.consensus_diagnostics = None

        if clear_model_state:
            for attr_name in ("model", "likelihood", "_model_pars"):
                if hasattr(self, attr_name):
                    setattr(self, attr_name, None)

        if clear_failure:
            self.fit_failed = False
            self.failure_reason = None
            self.failure_diagnostics = None
            self.failure_summary = None

    def _record_failure_state(
        self,
        *,
        reason,
        message,
        diagnostics=None,
        clear_model_state=False,
        clear_consensus=False,
    ):
        """Set canonical failed-fit state and return a failure summary object."""
        _diagnostics = self._consensus_make_json_safe(dict(diagnostics or {}))
        _diagnostics.setdefault("status", "failed")
        _diagnostics.setdefault("reason", reason)

        self._reset_fit_state(
            clear_failure=False,
            clear_model_state=clear_model_state,
            clear_consensus=clear_consensus,
        )
        self.fit_failed = True
        self.is_fitted = False
        self.failure_reason = reason
        self.failure_diagnostics = _diagnostics
        self.failure_summary = FitFailureSummary(
            status="failed",
            reason=reason,
            message=message,
            diagnostics=_diagnostics,
        )
        self.consensus_failure_summary = self.failure_summary
        self._append_fit_history(
            success=False,
            failed=True,
            exception_type="ConsensusFitError",
            exception_message=message,
            notes={"reason": reason, "source": "_record_failure_state"},
        )
        self._fit_history_recorded = True
        return self.failure_summary

    def get_failure_summary(self):
        """Return the most recent failure summary object, if available."""
        return self.failure_summary

    def _raise_if_fit_failed(self, action_message):
        """Raise a clean error for plot/summary requests after fit failure."""
        if not self.fit_failed:
            return

        _base_message = (
            "Cannot generate "
            f"{action_message}: the most recent consensus fit failed"
        )
        _reason_messages = {
            "no_accepted_bands": (
                "because the bands did not support a common periodicity."
            ),
            "insufficient_consensus_inliers": (
                "because too few reliable bands survived the consensus filtering stage."
            ),
            "frequency_aggregation_error": (
                "because robust frequency aggregation could not build a "
                "stable consensus."
            ),
            "invalid_consensus_frequency": (
                "because the inferred consensus frequency was not physically valid."
            ),
        }
        _tail = _reason_messages.get(
            self.failure_reason,
            "because the data did not support a coherent shared period.",
        )
        _message = f"{_base_message} {_tail}"
        exc = ConsensusFitError(
            _message,
            failure_diagnostics=self.failure_diagnostics or {"status": "failed"},
        )
        exc.failure_summary = self.failure_summary
        self._append_fit_history(
            success=False,
            failed=True,
            exception_type=exc.__class__.__name__,
            exception_message=_message,
            notes={
                "reason": self.failure_reason,
                "source": "_raise_if_fit_failed",
                "action": action_message,
            },
        )
        raise exc

    def transform_x(self, values):
        if self.xtransform is None:
            return values
        elif isinstance(self.xtransform, Transformer):
            return self.xtransform.transform(values)

    def transform_y(self, values):
        if self.ytransform is None:
            return values
        elif isinstance(self.xtransform, Transformer):
            return self.xtransform.transform(values)

    def set_likelihood(self, likelihood=None, variance=False, **kwargs):
        """Set the likelihood function for the model

        Parameters
        ----------
        likelihood : string, None or instance of
                     gpytorch.likelihoods.likelihood.Likelihood or Constraint,
                     optional
            The likelihood function to use for the GP, by default None.

            If ``likelihood`` is ``None`` and per-point uncertainties on the
            data are available (i.e. ``yerr`` has been set), a
            :class:`gpytorch.likelihoods.FixedNoiseGaussianLikelihood` is
            constructed and given a tensor of noise *variances* derived from
            those uncertainties. If ``likelihood`` is ``None`` and no
            uncertainties are available, a standard
            :class:`gpytorch.likelihoods.GaussianLikelihood` is used.

            If a string, it must be ``'learn'``. When ``'learn'`` is used and
            uncertainties are available, a
            :class:`gpytorch.likelihoods.FixedNoiseGaussianLikelihood` is
            created with the same noise variance tensor and with
            ``learn_additional_noise=True``. If ``'learn'`` is used and no
            uncertainties are available, a
            :class:`gpytorch.likelihoods.GaussianLikelihood` is created with
            ``learn_additional_noise=True``.

            If an instance of a :class:`~gpytorch.likelihoods.likelihood.Likelihood`
            object is passed, that object is used directly. If a Constraint
            object is passed, a :class:`gpytorch.likelihoods.GaussianLikelihood`
            is constructed with the constraint passed as the
            ``noise_constraint`` argument; this overrides any other keyword
            arguments to the likelihood.

            You can also provide a likelihood *class* (rather than an
            instance), in which case the class will be instantiated with the
            provided ``kwargs`` under the assumption that it is a
            :class:`~gpytorch.likelihoods.likelihood.Likelihood` subclass. If
            per-point uncertainties are available, a first positional argument
            will also be passed containing the noise tensor. This tensor is in
            units of variance by default (see the ``variance`` parameter
            below).
        variance : bool, optional
            Controls how stored per-point uncertainties are interpreted when
            constructing the noise tensor passed to likelihoods that require
            per-observation noise (e.g. :class:`gpytorch.likelihoods.
            FixedNoiseGaussianLikelihood` or user-supplied likelihood classes
            that accept a noise argument).

            If ``False`` (default), the uncertainties stored in the lightcurve
            are assumed to be errors (standard deviations). They are squared
            to produce noise *variances* before being passed to the likelihood.

            If ``True``, the stored uncertainties are assumed to already be
            variances and are passed through unchanged as the noise tensor.
        """

        # Prepare the noise tensor: gpytorch likelihoods expect variances.
        # By default (variance=False) we square the stored errors; if the
        # caller has already supplied variances, we use them as-is.
        _has_noise = hasattr(self, "_yerr_transformed")
        if _has_noise:
            noise = (
                self._yerr_transformed
                if variance
                else self._yerr_transformed ** 2
            )

        if _has_noise and likelihood is None:
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise
            )
        elif _has_noise and likelihood == "learn":
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise,
                learn_additional_noise=True,
            )
        elif likelihood == "learn":
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                learn_additional_noise=True
            )
        elif "Constraint" in [t.__name__ for t in type(likelihood).__mro__]:
            # In this case, the likelihood has been passed a constraint, which
            # means we want a constrained GaussianLikelihood
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=likelihood
            )
        elif likelihood is None:
            # We're just going to make the simplest assumption
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # Also add a case for if it is a Likelihood object
        elif isinstance(likelihood, gpytorch.likelihoods.likelihood.Likelihood):
            self.likelihood = likelihood
        elif isclass(likelihood):
            if _has_noise:
                self.likelihood = likelihood(noise, **kwargs)
            else:
                self.likelihood = likelihood(**kwargs)
        else:
            raise ValueError(
                f"""Expected a string, a constraint, a Likelihood
                              instance or a class to be instantiated as a
                              Likelihood instance, but got {type(likelihood)}.
                              Please provide a suitable likelihood input."""
            )
        self.__SET_LIKELIHOOD_CALLED = True

    def set_model(
        self, model=None, likelihood=None, num_mixtures=None, variance=False, **kwargs
    ):
        """Set the model for the lightcurve

        Parameters
        ----------
        model : string or instance of gpytorch.models.GP, optional
            The model to use for the GP, by default None. If None, an
            error will be raised. If a string, it must be one of the
            following:

            Spectral mixture models (default):
                '1D': SpectralMixtureGPModel
                '2D': TwoDSpectralMixtureGPModel
                '1DLinear': SpectralMixtureLinearMeanGPModel
                '2DLinear': TwoDSpectralMixtureLinearMeanGPModel
                '1DSKI': SpectralMixtureKISSGPModel
                '2DSKI': TwoDSpectralMixtureKISSGPModel
                '1DLinearSKI': SpectralMixtureLinearMeanKISSGPModel
                '2DLinearSKI': TwoDSpectralMixtureLinearMeanKISSGPModel
                '2DPowerLaw': TwoDSpectralMixturePowerLawMeanGPModel
                '2DPowerLawSKI': TwoDSpectralMixturePowerLawMeanKISSGPModel
                '2DDust': TwoDSpectralMixtureDustMeanGPModel
                '2DDustSKI': TwoDSpectralMixtureDustMeanKISSGPModel

            Alternative 1D models:
                '1DQuasiPeriodic': QuasiPeriodicGPModel
                '1DMatern': MaternGPModel
                '1DPeriodicStochastic': PeriodicPlusStochasticGPModel
                '1DLinearQuasiPeriodic': LinearMeanQuasiPeriodicGPModel

            Separable 2D models:
                '2DSeparable': SeparableGPModel
                '2DAchromatic': AchromaticGPModel
                '2DWavelengthDependent': WavelengthDependentGPModel
                '2DDustMean': DustMeanGPModel
                '2DPowerLawMean': PowerLawMeanGPModel


            If an instance of a GP class, that object will be used.
            _description_, by default None
        likelihood : string, None or instance of
                     gpytorch.likelihoods.likelihood.Likelihood or Constraint,
                     optional
            If likelihood is passed, it will be passed along to `set_likelihood()`
            and used to set the likelihood function for the model. For details, see
            the documentation for `set_likelihood()`.
        num_mixtures : int, optional
            The number of mixtures to use in the spectral mixture kernel, by
            default None. If None, a default value will be used. This value
            is passed to the constructor for the model if a string is passed
            as the model argument.
        variance : bool, optional
            Passed to `set_likelihood()`.  If False (default), stored
            uncertainties are treated as errors and squared before being used
            as noise variances.  Set to True if the stored uncertainties
            already represent variances.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the model constructor.
        """
        self.__SET_MODEL_CALLED = True
        if isinstance(model, str):
            self._model_str = model
            self._model_instance = None
        elif "GP" in [t.__name__ for t in type(model).__mro__]:
            # GP instance provided directly — store it so it can be rebound
            # to new training data after band filtering.
            self._model_instance = model
            self._model_str = None
        else:
            self._model_str = None
            self._model_instance = None
        self._model_num_mixtures = num_mixtures
        self._fit_num_mixtures_effective = num_mixtures
        self._fit_num_mixtures_requested = num_mixtures
        model_dic_1 = {
            "2D": TwoDSpectralMixtureGPModel,
            "1D": SpectralMixtureGPModel,
            "1DLinear": SpectralMixtureLinearMeanGPModel,
            "2DLinear": TwoDSpectralMixtureLinearMeanGPModel,
            "2DPowerLaw": TwoDSpectralMixturePowerLawMeanGPModel,
            "2DDust": TwoDSpectralMixtureDustMeanGPModel,
        }

        model_dic_2 = {
            "1DSKI": SpectralMixtureKISSGPModel,
            "2DSKI": TwoDSpectralMixtureKISSGPModel,
            "1DLinearSKI": SpectralMixtureLinearMeanKISSGPModel,
            "2DLinearSKI": TwoDSpectralMixtureLinearMeanKISSGPModel,
            "2DPowerLawSKI": TwoDSpectralMixturePowerLawMeanKISSGPModel,
            "2DDustSKI": TwoDSpectralMixtureDustMeanKISSGPModel,
        }

        # Alternative kernel models — do not require num_mixtures
        model_dic_alt = {
            "1DQuasiPeriodic": QuasiPeriodicGPModel,
            "1DMatern": MaternGPModel,
            "1DPeriodicStochastic": PeriodicPlusStochasticGPModel,
            "1DLinearQuasiPeriodic": LinearMeanQuasiPeriodicGPModel,
            "2DSeparable": SeparableGPModel,
            "2DAchromatic": AchromaticGPModel,
            "2DWavelengthDependent": WavelengthDependentGPModel,
            "2DDustMean": DustMeanGPModel,
            "2DPowerLawMean": PowerLawMeanGPModel,
        }

        if not hasattr(self, "likelihood"):
            self.set_likelihood(likelihood, variance=variance, **kwargs)
        elif not self.__SET_LIKELIHOOD_CALLED and likelihood is None:
            # if no likelihood is passed, we only want to set the likelihood
            # if it hasn't already been set
            self.set_likelihood(likelihood, variance=variance, **kwargs)
        elif likelihood is not None:
            self.set_likelihood(likelihood, variance=variance, **kwargs)

        if "GP" in [t.__name__ for t in type(model).__mro__]:
            # check if it is or inherets from a GPyTorch model
            self.model = model
        elif model in model_dic_1:
            self.model = model_dic_1[model](
                self._xdata_transformed,
                self._ydata_transformed,
                self.likelihood,
                num_mixtures=num_mixtures,
                **kwargs,
            )
        elif model in model_dic_2:
            self.model = model_dic_2[model](
                self._xdata_transformed,
                self._ydata_transformed,
                self.likelihood,
                num_mixtures=num_mixtures,
                **kwargs,
            )
        elif model in model_dic_alt:
            self.model = model_dic_alt[model](
                self._xdata_transformed,
                self._ydata_transformed,
                self.likelihood,
                num_mixtures=num_mixtures,
                **kwargs,
            )
        else:
            raise ValueError("Insert a valid model")

        # now we've got a model set up, we're going to make some handy lookups
        # for the parameters and modules that we'll need to access later
        self._make_parameter_dict()
        # self.set_default_constraints()

    def _make_parameter_dict(self):
        """Make a dictionary of the model parameters

        This function is used to make a dictionary of the model parameters,
        providing a convenient way to access them. The dictionary is stored
        in the _model_pars attribute.
        """
        self._model_pars = {}
        # there are a few parameters that we want to make sure we expose a
        # direct link to if we need them!
        _special_pars = ["noise", "mixture_means", "mixture_scales", "mixture_weights"]

        for param_name, param in self.model.named_parameters():
            comps = list(param_name.split("."))
            pn_root = comps[-1]
            param_dict = {
                "full_name": param_name,
                "root_name": pn_root,
                "chain": [],
                "constrained": False,
            }
            if "raw" in param_name:
                # This is a constrained parameter, so we need to get the
                # unconstrained value
                pn_const = comps[-1].lstrip("raw_")
                param_dict["constrained"] = True
                param_dict["constrained_name"] = pn_const
                pn = ".".join([c.lstrip("raw_") for c in comps])
                param_dict["constrained_full_name"] = pn
            tmp = self.model.__getattr__(comps[0])
            param_dict["chain"].append(tmp)
            for i in range(1, len(comps)):
                c = comps[i] if "raw" not in comps[i] else comps[i].lstrip("raw_")
                try:
                    tmp = tmp.__getattr__(c)
                except AttributeError:
                    tmp = tmp.__getattribute__(c)
                param_dict["chain"].append(tmp)
            param_dict["module"] = param_dict["chain"][-2]
            if param_dict["constrained"]:
                param_dict["param"] = param_dict["chain"][-1]
                try:
                    param_dict["raw_param"] = param_dict["chain"][-2].__getattr__(
                        comps[-1]
                    )
                except AttributeError:
                    param_dict["raw_param"] = param_dict["chain"][-2].__getattribute__(
                        comps[-1]
                    )
            else:
                param_dict["param"] = param_dict["chain"][-1]
                param_dict["raw_param"] = param_dict["param"]
            if any(s in pn_root for s in _special_pars):
                # it's a special parameter that we want extra easy access to!
                param_dict["special"] = True
                j = np.argmax([s in pn_root for s in _special_pars])
                self._model_pars[_special_pars[j]] = param_dict

            #     pars[pn] = tmp.data
            # else:
            #     # Either we actually want the raw values, or it's not a
            #     # constrained parameter
            #     pars[param_name] = param.data
            self._model_pars[param_name] = param_dict
            if param_dict["constrained"]:
                self._model_pars[pn] = param_dict  # so we also alias the full
                # name for the constrained
                # parameter

    def set_prior(self, prior=None, **kwargs):
        """Set the prior for the model parameters

        Parameters
        ----------
        prior : dict, optional
            A dictionary of the priors to use for the model parameters. The
            keys should be the names of the parameters, and the values should
            be instances of gpytorch.priors.Prior. If None, no priors will be
            used. If a prior is passed for a parameter that is not a model
            parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the Prior constructors.
        """
        self.__PRIORS_SET = True
        pass

    def set_constraint(self, constraint, debug=False, **kwargs):
        """Set the constraint for the model parameters

        Parameters
        ----------
        constraint : dict, optional
            A dictionary of the constraints to use for the model parameters.
            The keys should be the names of the parameters, and the values
            should be instances of gpytorch.constraints.Constraint. If None, no
            constraints will be used. If a constraint is passed for a parameter
            that is not a model parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the Constraint
            constructors.
        """
        # which paramaters need to have their constraints transformed? and how?
        pars_to_transform = {
            "x": ["mixture_means", "mixture_scales"],
            "y": ["noise", "mean_module"],
        }

        for key in constraint:
            if key in self._model_pars:
                if debug:
                    print(f"Found parameter {key} in model parameters")
                    print(f"Parameter {key} will have constraint: {constraint[key]}")
                    print("which may be transformed")
                # constraints must be registered to raw parameters!
                k = key.split(".")[-1] if "raw_" in key else f"raw_{key.split('.')[-1]}"
                if all(
                    p not in key
                    for p in pars_to_transform["y"] + pars_to_transform["x"]
                ):  # no transform needed!
                    register_constraint_preserving_value(
                        self._model_pars[key]["module"],
                        k,
                        constraint[key],
                    )
                elif any(p in key for p in pars_to_transform["x"]):
                    # now apply the x transform
                    # remember that the means and scales are in fourier space
                    # so we need to transform them back to real space
                    # before applying the transform
                    # and then transform them back to fourier space
                    # luckily, when the shift is removed from the transform,
                    # the factors of 2pi cancel out for the scales
                    # so we can just do 1/ for both means and scales
                    if self.xtransform is not None:
                        # now things get complicated...
                        # if we have gotten to here, we know that the parameter
                        # is a mixture mean or scale, so we need to transform
                        # it to real space, apply the constraint, and then
                        # transform it back to fourier space
                        # luckily, when the shift is removed from the transform,
                        # the factors of 2pi cancel out for the scales
                        # so we can just do 1/ for both means and scales
                        if debug:
                            print(constraint[key])
                        if constraint[key].lower_bound not in [
                            torch.tensor(0),
                            torch.tensor(-torch.inf),
                        ]:
                            # we need to transform the lower bound
                            # NOTE: For 2D data, GPyTorch constraints are scalar
                            # and apply element-wise to all parameter elements
                            # (time and wavelength).
                            # We transform using dimension 0 (time) as it's
                            # typically the primary independent variable.
                            # Users setting manual constraints should be aware
                            # that the same constraint applies to both
                            # dimensions.
                            transformed_bound = 1.0 / self.xtransform.transform(
                                1.0 / constraint[key].lower_bound, shift=False
                            )
                            # Handle both 1D and 2D cases
                            if transformed_bound.numel() > 1:
                                # For 2D case, use the first dimension's
                                # transformation. Take element [0, 0] to get
                                # a scalar
                                transformed_bound = transformed_bound.flatten()[0]
                            if debug:
                                print(f"Transformed lower bound: {transformed_bound}")
                            constraint[key].lower_bound = torch.tensor(
                                transformed_bound.item()
                            )
                            if debug:
                                print(constraint[key].lower_bound)
                                print(constraint[key])
                        if constraint[key].upper_bound not in [
                            torch.tensor(0),
                            torch.tensor(torch.inf),
                        ]:
                            # we need to transform the upper bound
                            # (Same dimension-0 transformation logic as
                            # lower_bound above)
                            transformed_bound = 1.0 / self.xtransform.transform(
                                1.0 / constraint[key].upper_bound, shift=False
                            )
                            # Handle both 1D and 2D cases
                            if transformed_bound.numel() > 1:
                                # For 2D case, use the first dimension's
                                # transformation. Take element [0, 0] to get
                                # a scalar
                                transformed_bound = transformed_bound.flatten()[0]
                            constraint[key].upper_bound = torch.tensor(
                                transformed_bound.item()
                            )
                            if debug:
                                print(constraint[key].upper_bound)
                                print(constraint[key])
                        if debug:
                            print(constraint[key])
                    register_constraint_preserving_value(
                        self._model_pars[key]["module"],
                        k,
                        constraint[key],
                    )
                elif any(p in key for p in pars_to_transform["y"]):
                    if self.ytransform is not None:
                        if debug:
                            print(constraint[key])
                        if isinstance(constraint[key], Positive) and (
                            isinstance(self.ytransform, ZScore | RobustZScore)
                        ):
                            # convert constraint to an interval with minimum equal to
                            # what zero is in the untransformed space
                            constraint[key] = Interval(
                                self.ytransform.transform(0), torch.inf
                            )
                            if debug:
                                print(constraint[key])

                        elif constraint[key].lower_bound not in [
                            torch.tensor(0),
                            torch.tensor(-torch.inf),
                        ]:
                            # we need to transform the lower bound
                            constraint[key].lower_bound = torch.tensor(
                                self.ytransform.transform(
                                    constraint[key].lower_bound
                                ).item()
                            )
                            if debug:
                                print(constraint[key].lower_bound)
                                print(constraint[key])
                        if constraint[key].upper_bound not in [
                            torch.tensor(0),
                            torch.tensor(torch.inf),
                        ]:
                            # we need to transform the upper bound
                            constraint[key].upper_bound = torch.tensor(
                                self.ytransform.transform(
                                    constraint[key].upper_bound
                                ).item()
                            )
                            if debug:
                                print(constraint[key].upper_bound)
                                print(constraint[key])
                    if debug:
                        print(constraint[key])
                    register_constraint_preserving_value(
                        self._model_pars[key]["module"],
                        k,
                        constraint[key],
                    )
                if debug:
                    try:
                        print(f"Registered constraint {constraint[key]}")
                    except TypeError:
                        print("Registered constraint")
                        print(constraint[key])
                    print(f"to parameter {key}")
            else:
                print(f"Parameter {key} not found in model parameters,")
                print("this constraint will be ignored.")
                print("Available parameters are:")
                print(self._model_pars.keys())
                print("(Beware, several of these are aliases!)")

    def set_default_priors(self, prior_set=None, **kwargs):
        """Set the default priors for the model and likelihood parameters

        The default priors are as follows:
            - Parameters that must be positive are given LogNormal, HalfNormal
            or HalfCauchy priors, depending on the parameter.
            - The noise is given a HalfNormal prior with a scale of 1/10 of the
            smallest uncertainty on the y-data, if uncertainties are given, or
            1/10 of the standard deviation of the y data.
            - The mean of the GP is given a Gaussian prior with a mean of the
            mean of the y-data and a standard deviation of 1/10 of the standard
            deviation of the y-data.
            - For spectral-mixture models the mixture means, scales and weights
            receive LogNormal(0, 1) priors.
            - If *prior_set* is provided, :meth:`set_period_prior` is called
            with that prior set to register a physically motivated prior on
            the period/frequency parameter of the model.

        Parameters
        ----------
        prior_set : str or None, optional
            Name of a predefined prior set to apply to the period/frequency
            parameter (e.g. ``"LPV"``).  If ``None`` (default), no period
            prior is added.  See :meth:`set_period_prior` and
            :data:`~pgmuvi.priors.PRIOR_SETS` for available options.
        **kwargs : dict, optional
            Any keyword arguments to be passed to the Prior constructors.
        """

        # Gpytorch currently crashes if you try to do MCMC while learning additional
        # diagonal noise with the FixedNoiseGaussianLikelihood. So we only need to
        # set priors for the noise if we don't have uncertainties on the data.
        if not hasattr(self, "_yerr_transformed"):
            try:
                noise_scale = np.minimum(1e-4, self._yerr_transformed.min() / 10)
            except AttributeError:
                noise_scale = 1e-4 * self._ydata_transformed.std()
            # noise_prior = gpytorch.priors.HalfCauchyPrior(noise_scale)
            noise_prior = gpytorch.priors.LogNormalPrior(
                torch.log(noise_scale), noise_scale
            )
            self._model_pars["noise"]["module"].register_prior(
                "noise_prior", noise_prior, "noise"
            )
        with contextlib.suppress(RuntimeError):
            mean_prior = gpytorch.priors.NormalPrior(
                self._ydata_transformed.mean(), self._ydata_transformed.std() / 10
            )
            for key in self._model_pars:
                if "mean_module.constant" in key:
                    self._model_pars[key]["module"].register_prior(
                        "mean_prior", mean_prior, "constant"
                    )
        # we use a lognormal prior for the means, because we want to make sure
        # that the means are positive, but we don't want to restrict them to
        # be close to zero. In fact, we want to penalise both very high and very low
        # frequencies, so we use a lognormal prior with mu = 0 and sigma = 1
        if "mixture_means" in self._model_pars:
            mixture_means_prior = gpytorch.priors.LogNormalPrior(
                0, 1
            )  # /self._xdata_transformed.max())
            self._model_pars["mixture_means"]["module"].register_prior(
                "mixture_means_prior", mixture_means_prior, "mixture_means"
            )

        # now we need a prior for the mixture scales
        # we want to penalise very large scales, so we use a half-cauchy prior
        # with a scale of 1/10 of the maximum frequency
        # mixture_scales_prior = gpytorch.priors.HalfCauchyPrior(1/self._xdata_transformed.max())  # noqa: E501
        if "mixture_scales" in self._model_pars:
            mixture_scales_prior = gpytorch.priors.LogNormalPrior(
                0, 1
            )  # 1/self._xdata_transformed.max())
            self._model_pars["mixture_scales"]["module"].register_prior(
                "mixture_scales_prior", mixture_scales_prior, "mixture_scales"
            )
        # we use a LogNormal prior for the mixture weights, because we want to
        # make sure that they are positive (but never zero) and we don't want
        # to restrict them to be close to zero. In fact, we want to penalise
        # both very high and very low weights, so we use a LogNormal prior
        # with a scale of 1/10 of the maximum frequency
        if "mixture_weights" in self._model_pars:
            mixture_weights_prior = gpytorch.priors.LogNormalPrior(
                0, 1
            )  # 1/self._xdata_transformed.max())
            self._model_pars["mixture_weights"]["module"].register_prior(
                "mixture_weights_prior", mixture_weights_prior, "mixture_weights"
            )

        # Apply a period/frequency prior if a prior_set is requested
        if prior_set is not None:
            self.set_period_prior(prior_set=prior_set)

        # need a more general way to assign default priors to everything, but for now
        # let's see if this works!
        self.__PRIORS_SET = True

    def get_priors(self):
        """Return the priors currently registered on the model.

        Iterates over all priors registered on the model (via GPyTorch's
        ``named_priors``) and returns a dictionary mapping each prior name to
        the corresponding prior object.  A formatted summary is also printed to
        standard output.

        Returns
        -------
        dict
            A dictionary mapping prior names (str) to their
            :class:`gpytorch.priors.Prior` objects. Returns an empty dict if
            no priors have been registered.

        Raises
        ------
        RuntimeError
            If the model has not been set yet (call :meth:`set_model` first).

        See Also
        --------
        set_default_priors
        get_period_prior

        Examples
        --------
        ::

            lc.set_model("1D", num_mixtures=4)
            lc.set_default_priors()
            priors = lc.get_priors()
        """
        if not hasattr(self, "_model_pars"):
            raise RuntimeError(
                "Model has not been set yet. Call set_model() before "
                "get_priors()."
            )
        priors = {}
        for name, _module, prior, _closure, _setting_closure in (
            self.model.named_priors()
        ):
            priors[name] = prior
        print("Registered priors:")
        if priors:
            for name, prior in priors.items():
                print(f"  {name}: {prior}")
        else:
            print("  (none)")
        return priors

    def set_period_prior(
        self,
        prior_set=None,
        prior_type="lognormal",
        mu=5.0,
        sigma=1.0,
        mean=300.0,
        std=75.0,
        lower_period=None,
        upper_period=None,
        period=True,
    ):
        """Set a prior on the period or frequency parameter of the model.

        This method detects whether the model represents periodicity as a
        period (e.g. ``period_length`` in
        :class:`~pgmuvi.gps.QuasiPeriodicGPModel`) or as a frequency (e.g.
        ``mixture_means`` in
        :class:`~pgmuvi.gps.SpectralMixtureGPModel`) and registers an
        appropriate prior on the relevant parameter.

        For frequency-based models the period-space prior is transformed to
        frequency space with the correct change-of-variables Jacobian (see
        :class:`~pgmuvi.priors.LogNormalFrequencyPrior` and
        :class:`~pgmuvi.priors.NormalFrequencyPrior`).

        Models with no periodicity parameter (e.g.
        :class:`~pgmuvi.gps.MaternGPModel`) are silently skipped with a
        warning.

        Parameters
        ----------
        prior_set : str or None, optional
            Name of a predefined prior set (e.g. ``"LPV"``).  When given,
            the ``prior_type``, ``mu``, ``sigma``, ``mean``, ``std`` and
            ``lower_period`` / ``upper_period`` defaults are taken from
            :data:`~pgmuvi.priors.PRIOR_SETS`.  Any explicitly supplied
            keyword arguments override the prior-set values.
        prior_type : str, optional
            Which prior family to use.  Either ``"lognormal"`` (the default,
            LogNormal in period space with ``mu`` and ``sigma``) or
            ``"normal"`` (Normal in period space with ``mean`` and ``std``).
            Case-insensitive.
        mu : float, optional
            Mean of the underlying normal distribution for the Log-Normal
            period prior (i.e. the log-mean).  Default ``5.0``
            (median period ≈ 148 days).  Dimensionless (logarithmic units).
        sigma : float, optional
            Standard deviation of the underlying normal distribution for the
            Log-Normal period prior (i.e. the log-standard-deviation).
            Default ``1.0``.  Dimensionless (logarithmic units).
        mean : float, optional
            Mean for the Normal period prior (days).  Default ``300.0``.
        std : float, optional
            Standard deviation for the Normal period prior (days).
            Default ``75.0``.
        lower_period : float or None, optional
            Lower bound on period.  When ``period=True`` (default), this is
            in days (the assumed time unit of the data).  When ``period=False``
            this is a lower bound in frequency units (1/days).
            Values outside this bound receive ``-inf`` log-prob.
            If ``None`` and a *prior_set* is provided, the bound is taken
            from the prior set.
        upper_period : float or None, optional
            Upper bound on period (days when ``period=True``, 1/days
            when ``period=False``).
        period : bool, optional
            Controls the interpretation of ``lower_period`` and
            ``upper_period`` for *frequency-parameterised* models (i.e.
            spectral-mixture models whose periodicity is encoded as
            ``mixture_means``).  If ``True`` (default), bounds are in period
            units (days).  If ``False``, bounds are in frequency units
            (1/days).  Has no effect for period-parameterised models
            (e.g. ``QuasiPeriodicGPModel``), which always use period units.

        Raises
        ------
        ValueError
            If *prior_set* is not a recognised name or if *prior_type* is not
            ``"lognormal"`` or ``"normal"``.

        Notes
        -----
        The model must have been set (via :meth:`set_model` or :meth:`fit`)
        before calling this method.

        For spectral-mixture models the prior is registered on
        ``mixture_means`` and applies element-wise to all mixture
        components.

        For quasi-periodic models the prior is registered on each
        ``period_length`` parameter found in the model (there is typically
        only one).

        See Also
        --------
        pgmuvi.priors.LogNormalPeriodPrior
        pgmuvi.priors.LogNormalFrequencyPrior
        pgmuvi.priors.NormalPeriodPrior
        pgmuvi.priors.NormalFrequencyPrior
        pgmuvi.priors.PRIOR_SETS

        Examples
        --------
        Set the LPV default prior on a spectral-mixture model::

            lc.set_model("1D", num_mixtures=4)
            lc.set_period_prior(prior_set="LPV")

        Set a Normal period prior explicitly::

            lc.set_model("1DQuasiPeriodic")
            lc.set_period_prior(prior_type="normal", mean=300.0, std=75.0,
                                lower_period=100.0)
        """
        if not hasattr(self, "_model_pars"):
            raise RuntimeError(
                "Model has not been set yet. Call set_model() before "
                "set_period_prior()."
            )

        # Normalise prior_type to lower case so callers can use any case
        prior_type = prior_type.lower()

        # If bounds are in frequency units, convert to period units now so the
        # rest of the logic always works in period space.  The prior_set always
        # stores bounds in period space, so we only convert user-provided bounds.
        if not period:
            # lower frequency ↔ upper period, and vice versa
            if lower_period is not None and lower_period <= 0:
                raise ValueError(
                    f"lower_period as a frequency bound must be positive, "
                    f"got {lower_period}."
                )
            if upper_period is not None and upper_period <= 0:
                raise ValueError(
                    f"upper_period as a frequency bound must be positive, "
                    f"got {upper_period}."
                )
            lower_period, upper_period = (
                (1.0 / upper_period if upper_period is not None else None),
                (1.0 / lower_period if lower_period is not None else None),
            )

        # --- Resolve prior-set defaults ---
        if prior_set is not None:
            ps = get_prior_set(prior_set)
            # Use prior-set values as defaults; explicit kwargs override them
            if prior_type == "lognormal":
                mu = ps["lognormal"].get("mu", mu)
                sigma = ps["lognormal"].get("sigma", sigma)
            elif prior_type == "normal":
                mean = ps["normal"].get("mean", mean)
                std = ps["normal"].get("std", std)
            pb = ps["period_bounds"]
            if lower_period is None:
                lower_val, lower_active = pb["lower"]
                lower_period = lower_val if lower_active else None
            if upper_period is None:
                upper_val, upper_active = pb["upper"]
                upper_period = upper_val if upper_active else None

        if prior_type not in ("lognormal", "normal"):
            raise ValueError(
                f"prior_type must be 'lognormal' or 'normal', got {prior_type!r}"
            )

        # --- Compute scale factor to convert raw period bounds to model space ---
        # In model (transformed) space the period is related to the raw period by
        #   period_model = period_raw * (x_trans_span / x_orig_span)
        # For linear transforms this equals period_raw when no transform is used.
        if self.ndim > 1:
            x_orig_span = float(
                self._xdata_raw[:, 0].max() - self._xdata_raw[:, 0].min()
            )
            x_trans_span = float(
                self._xdata_transformed[:, 0].max()
                - self._xdata_transformed[:, 0].min()
            )
        else:
            if hasattr(self, "_xdata_raw"):
                x_orig_span = float(
                    self._xdata_raw.max() - self._xdata_raw.min()
                )
            else:
                x_orig_span = float(
                    self._xdata_transformed.max() - self._xdata_transformed.min()
                )
            x_trans_span = float(
                self._xdata_transformed.max() - self._xdata_transformed.min()
            )
        period_scale = (
            x_trans_span / x_orig_span if x_orig_span > 0 else 1.0
        )

        # Convert raw period bounds to model-space period bounds
        lower_model = (
            float(lower_period) * period_scale if lower_period is not None else None
        )
        upper_model = (
            float(upper_period) * period_scale if upper_period is not None else None
        )

        # --- Detect model type and register the prior ---
        # Case 1: frequency-based model (spectral mixture) → mixture_means
        if "mixture_means" in self._model_pars:
            if prior_type == "lognormal":
                prior = LogNormalFrequencyPrior(
                    mu=mu, sigma=sigma,
                    lower_period=lower_model,
                    upper_period=upper_model,
                    period=True,
                )
            else:
                prior = NormalFrequencyPrior(
                    mean=mean, std=std,
                    lower_period=lower_model,
                    upper_period=upper_model,
                    period=True,
                )
            self._model_pars["mixture_means"]["module"].register_prior(
                "mixture_means_prior", prior, "mixture_means"
            )
            return

        # Case 2: period-based model (quasi-periodic) → period_length parameters
        period_keys = [
            k for k in self._model_pars
            if "period_length" in k and "raw_" not in k
        ]
        if period_keys:
            if prior_type == "lognormal":
                prior = LogNormalPeriodPrior(
                    mu=mu, sigma=sigma,
                    lower_bound=lower_model,
                    upper_bound=upper_model,
                )
            else:
                prior = NormalPeriodPrior(
                    mean=mean, std=std,
                    lower_bound=lower_model,
                    upper_bound=upper_model,
                )
            for key in period_keys:
                module = self._model_pars[key]["module"]
                module.register_prior("period_length_prior", prior, "period_length")
            return

        # Case 3: no periodicity parameter
        warnings.warn(
            "No period or frequency parameter found in the model. "
            "set_period_prior() has no effect for this model type.",
            stacklevel=2,
        )

    def get_period_prior(self):
        """Return the period or frequency prior currently registered on the model.

        Searches for a prior on the period or frequency parameter of the model
        (``mixture_means_prior`` for spectral-mixture models,
        ``period_length_prior`` for quasi-periodic models) and returns a
        dictionary of the priors found, keyed by the full parameter path.  A
        formatted summary is also printed to standard output.

        Returns
        -------
        dict
            A dictionary mapping prior names (str) to their prior objects.
            Returns an empty dict if no period/frequency prior has been
            registered or the model has no periodicity parameter.

        Raises
        ------
        RuntimeError
            If the model has not been set yet (call :meth:`set_model` first).

        See Also
        --------
        set_period_prior
        get_priors

        Examples
        --------
        ::

            lc.set_model("1D", num_mixtures=4)
            lc.set_period_prior(prior_set="LPV")
            prior_info = lc.get_period_prior()
        """
        if not hasattr(self, "_model_pars"):
            raise RuntimeError(
                "Model has not been set yet. Call set_model() before "
                "get_period_prior()."
            )
        period_priors = {}
        for name, _module, prior, _closure, _setting_closure in (
            self.model.named_priors()
        ):
            if "mixture_means_prior" in name or "period_length_prior" in name:
                period_priors[name] = prior

        print("Registered period/frequency priors:")
        if period_priors:
            for name, prior in period_priors.items():
                prior_type = type(prior).__name__
                line = f"  {name}: {prior_type}"
                params = []
                if hasattr(prior, "loc"):
                    params.append(f"loc={float(prior.loc):.4g}")
                if hasattr(prior, "scale"):
                    params.append(f"scale={float(prior.scale):.4g}")
                if hasattr(prior, "lower_period") and prior.lower_period is not None:
                    params.append(f"lower_period={float(prior.lower_period):.4g}")
                if hasattr(prior, "upper_period") and prior.upper_period is not None:
                    params.append(f"upper_period={float(prior.upper_period):.4g}")
                if hasattr(prior, "lower_bound") and prior.lower_bound is not None:
                    params.append(f"lower_bound={float(prior.lower_bound):.4g}")
                if hasattr(prior, "upper_bound") and prior.upper_bound is not None:
                    params.append(f"upper_bound={float(prior.upper_bound):.4g}")
                if params:
                    line += f"({', '.join(params)})"
                print(line)
        else:
            print("  (none)")
        return period_priors

    def _validate_2d_setup(self):
        """Validate that the 2D setup is correct

        This method checks that:
        - Data shapes are correct for 2D (xdata has shape [n_samples, 2])
        - Model has appropriate ard_num_dims parameter
        - Transforms can handle 2D data

        Raises
        ------
        ValueError
            If the 2D setup is invalid

        Warnings
        --------
        If there are potential issues with the setup
        """
        if self.ndim <= 1:
            return  # Only validate for 2D data

        # Check xdata shape
        if self._xdata_transformed.dim() != 2:
            raise ValueError(
                f"For 2D data, xdata must be 2-dimensional, "
                f"got {self._xdata_transformed.dim()}D"
            )

        if self._xdata_transformed.shape[1] != 2:
            raise ValueError(
                f"For 2D data, xdata must have 2 columns (time, wavelength), "
                f"got {self._xdata_transformed.shape[1]} columns"
            )

        # Check if model is set
        if not hasattr(self, "model"):
            warnings.warn(
                "Model not set yet. Cannot validate ard_num_dims. "
                "Ensure your model has ard_num_dims=2 for 2D data.",
                stacklevel=2,
            )
            return

        # Check if the model's kernel has ard_num_dims set correctly.
        # Separable models using ProductKernel + active_dims do not set
        # ard_num_dims (it stays None) — they are valid 2D models.
        # Only raise if ard_num_dims is explicitly set to a non-2 value.
        if hasattr(self.model, "covar_module"):
            covar = self.model.covar_module
            # For KISS-GP models, check the base_kernel
            if hasattr(covar, "base_kernel"):
                covar = covar.base_kernel

            if hasattr(covar, "ard_num_dims") and covar.ard_num_dims is not None:
                if covar.ard_num_dims != 2:
                    raise ValueError(
                        f"Model's ard_num_dims is {covar.ard_num_dims}, "
                        f"but data has {self.ndim} dimensions. "
                        "Use a 2D model (e.g., '2D', '2DLinear', '2DSKI', "
                        "'2DLinearSKI', '2DSeparable', '2DAchromatic', "
                        "'2DWavelengthDependent', '2DPowerLaw', '2DPowerLawSKI', "
                        "'2DDust', '2DDustSKI', '2DDustMean', '2DPowerLawMean')."
                    )

        # Check transform compatibility
        if self.xtransform is not None:
            if not hasattr(self.xtransform, "transform"):
                raise ValueError("xtransform must have a 'transform' method")

    def set_default_constraints(self, constraint_set=None, **kwargs):
        """Set the default constraints for the model and likelihood parameters

        The default constraints are as follows:
            - All parameters are constrained to be positive, except the mean of
            the GP, which is constrained to be in the range of the y-data (a correction
            will be needed if the data are censored!)
            - The noise is constrained to be less than the standard deviation of
            the y-data, and greater than either 1e-4 or 1/10 of the smallest
            uncertainty on the y-data, if uncertainties are given, or 1e-4
            times the standard deviation of the y data.
            - The mixture means greater than the frequency corresponding to
            the separation between the earliest and latest points in the data
            and less than the frequency corresponding to the separation between
            the two closest data points (should be updated to account for the
            window function and whatever we're really sensitive to)
            - The mixture scales and weights are left with their default
            constraints as defined in GPyTorch.

        Parameters
        ----------
        constraint_set : str or None, optional
            Name of a pre-defined source-type constraint set to apply on top
            of the default constraints.  When provided, the constraints
            defined for the named set (see
            :data:`pgmuvi.constraints.CONSTRAINT_SETS`) are merged into the
            default mixture-means constraint.  Currently supported values:

            ``"LPV"``
                Long-Period Variable stars.  Enforces a lower period limit of
                20 in the same time units as the input ``xdata`` (typically
                interpreted as 20 days for LPV light curves) so that the fit is
                not pulled toward unphysically short periods.  If ``xdata``
                is provided in different time units, this numerical limit
                applies in those units.

            Pass ``None`` (the default) to use only the data-driven defaults.
        **kwargs : dict, optional
            Any keyword arguments to be passed to the Constraint constructors.
        """
        if "noise" in self._model_pars:
            # only apply the noise constraint if we're using a learnable noise
            try:
                noise_min = np.minimum(1e-4, self._yerr_transformed.min() / 10)
            except AttributeError:
                noise_min = 1e-4 * self._ydata_transformed.std()
            noise_max = self._ydata_transformed.std()  # for a non-periodic source,
            # the noise should be less than
            # the standard deviation
            noise_constraint = Interval(noise_min, noise_max)
            register_constraint_preserving_value(
                self._model_pars["noise"]["module"],
                "raw_noise",
                noise_constraint,
            )
        with contextlib.suppress(RuntimeError):
            mean_const_constraint = Interval(
                self._ydata_transformed.min(), self._ydata_transformed.max()
            )
            for key in self._model_pars:
                if "mean_module.constant" in key:
                    register_constraint_preserving_value(
                        self._model_pars[key]["module"],
                        "raw_constant",
                        mean_const_constraint,
                    )
        # Apply frequency constraints only for spectral-mixture models that
        # have a mixture_means parameter.  Models using alternative kernels
        # (e.g. Matérn, quasi-periodic, separable) do not have this parameter
        # and must not be constrained here; they would raise KeyError otherwise.
        if "mixture_means" in self._model_pars:
            # this should correspond to the longest frequency entirely
            # contained in the dataset:
            if self.ndim > 1:
                # For 2D spectral-mixture models the mixture_means parameter
                # has shape (num_mixtures, 1, ard_num_dims), where ard_num_dims
                # equals the number of input dimensions (typically 2: time and
                # wavelength).  GPyTorch applies a *single scalar* constraint
                # element-wise to every entry in that tensor — it is not
                # possible to set different lower bounds for the time dimension
                # and the wavelength dimension simultaneously via the standard
                # register_constraint API.
                #
                # We therefore base the lower bound exclusively on the *time*
                # dimension (column 0 of xdata_transformed):
                #
                #   lower_bound = 1 / time_span
                #
                # This guarantees that the time-axis frequencies are always
                # >= 1/time_span, i.e., that the inferred periods are not
                # longer than the observational baseline — a physically
                # meaningful and stable lower bound.
                #
                # Note that this same lower bound is also applied to the
                # wavelength-axis frequency elements of mixture_means.  In
                # practice, wavelength frequencies represent the spatial
                # frequency of the SED variation across bands; constraining
                # them to be >= 1/time_span is conservative (frequencies
                # corresponding to structures much narrower in wavelength than
                # the observation baseline are still allowed), and is
                # preferable to using min(time_bound, wavelength_bound) which
                # would make the time lower bound arbitrarily permissive
                # whenever the wavelength span is large or the wavelength
                # range is zero.
                #
                # Users who need achromatic behaviour (wavelength-frequency
                # near zero) should use the separable model classes
                # (AchromaticGPModel, WavelengthDependentGPModel) which apply
                # kernels to each dimension independently, avoiding this
                # limitation entirely.
                time_span = (
                    self._xdata_transformed[:, 0].max()
                    - self._xdata_transformed[:, 0].min()
                )
                if float(time_span) <= 0.0:
                    raise ValueError(
                        "set_default_constraints requires a dataset whose "
                        "timestamps span a positive time range, but all "
                        "timestamps in the 2D input are identical "
                        "(time_span = 0). Ensure the training data covers "
                        "more than one distinct observation time."
                    )
                lower_frequency = 1.0 / time_span

                # Compute the Nyquist upper bound from the minimum positive
                # gap between consecutive sorted timestamps (O(N log N), O(N)
                # memory — avoids the O(N²) pairwise-difference matrix).
                t_sorted = self._xdata_transformed[:, 0].sort().values
                consecutive_diffs = (t_sorted[1:] - t_sorted[:-1])
                positive_diffs = consecutive_diffs[consecutive_diffs > 0]
                if positive_diffs.numel() > 0:
                    min_diff = positive_diffs.min()
                    max_freq = 1 / (2 * min_diff)  # Nyquist based on time sampling
                    mixture_means_constraint = Interval(lower_frequency, max_freq)
                else:
                    # time_span > 0 guarantees at least two distinct timestamps,
                    # so positive_diffs is always non-empty here.  This branch
                    # is unreachable in practice.
                    raise ValueError(  # pragma: no cover
                        "Unexpected degenerate timestamps: time_span > 0 but "
                        "no consecutive positive differences found."
                    )
            else:
                # 1D case: base the lower-frequency bound on the time span
                # (max - min) rather than the absolute maximum. This prevents
                # allowing periods longer than the observational baseline and
                # is consistent with the 2D logic above.
                time_span = (
                    self._xdata_transformed.max()
                    - self._xdata_transformed.min()
                )
                if float(time_span) <= 0.0:
                    raise ValueError(
                        "set_default_constraints requires a dataset whose "
                        "timestamps span a positive time range, but all "
                        "timestamps in the 1D input are identical "
                        "(time_span = 0). Ensure the training data covers "
                        "more than one distinct observation time."
                    )
                mixture_means_constraint = GreaterThan(1 / time_span)

            # Apply any constraint_set period bounds to the mixture_means
            # constraint
            if constraint_set is not None:
                cs = get_constraint_set(constraint_set)
                if "period" in cs:
                    period_bounds = cs["period"]
                    lower_val, lower_active = period_bounds["lower"]
                    upper_val, upper_active = period_bounds["upper"]

                    # Compute the scale factor to convert a period in original
                    # (untransformed) units to a frequency in transformed space.
                    # For any linear rescaling transform:
                    #   freq_transformed = freq_original * (x_orig_span /
                    #                                       x_trans_span)
                    if self.ndim > 1:
                        x_orig_span = float(
                            self._xdata_raw[:, 0].max()
                            - self._xdata_raw[:, 0].min()
                        )
                        x_trans_span = float(
                            self._xdata_transformed[:, 0].max()
                            - self._xdata_transformed[:, 0].min()
                        )
                    else:
                        x_orig_span = float(
                            self._xdata_raw.max() - self._xdata_raw.min()
                        )
                        x_trans_span = float(
                            self._xdata_transformed.max()
                            - self._xdata_transformed.min()
                        )
                    freq_scale = (
                        x_orig_span / x_trans_span if x_trans_span > 0 else 1.0
                    )

                    # Period lower limit → frequency upper limit
                    if lower_active and lower_val is not None:
                        max_freq_from_period = freq_scale / lower_val
                        cur_lower = float(mixture_means_constraint.lower_bound)
                        if max_freq_from_period > cur_lower:
                            if isinstance(mixture_means_constraint, GreaterThan):
                                mixture_means_constraint = Interval(
                                    cur_lower, max_freq_from_period
                                )
                            else:
                                # Already an Interval: tighten the upper bound
                                cur_upper = float(
                                    mixture_means_constraint.upper_bound
                                )
                                mixture_means_constraint = Interval(
                                    cur_lower,
                                    min(cur_upper, max_freq_from_period),
                                )

                    # Period upper limit → frequency lower limit
                    if upper_active and upper_val is not None:
                        min_freq_from_period = freq_scale / upper_val
                        cur_lower = float(mixture_means_constraint.lower_bound)
                        cur_upper = (
                            float(mixture_means_constraint.upper_bound)
                            if isinstance(mixture_means_constraint, Interval)
                            else float("inf")
                        )
                        new_lower = max(cur_lower, min_freq_from_period)
                        if new_lower < cur_upper:
                            if isinstance(mixture_means_constraint, GreaterThan):
                                mixture_means_constraint = GreaterThan(new_lower)
                            else:
                                mixture_means_constraint = Interval(
                                    new_lower, cur_upper
                                )

            register_constraint_preserving_value(
                self._model_pars["mixture_means"]["module"],
                "raw_mixture_means",
                mixture_means_constraint,
            )

        # to-do - check if constraints on mixture scales are useful!
        self.__CONTRAINTS_SET = True

    def get_constraints(self):
        """Return the constraints currently registered on the model.

        Iterates over all constraints registered on the model (via GPyTorch's
        ``named_constraints``) and returns a dictionary mapping each constraint
        name to the corresponding constraint object.  A formatted summary is
        also printed to standard output.

        Returns
        -------
        dict
            A dictionary mapping constraint names (str) to their
            :class:`gpytorch.constraints.Constraint` objects. Returns an empty
            dict if no constraints have been registered.

        Raises
        ------
        RuntimeError
            If the model has not been set yet (call :meth:`set_model` first).

        See Also
        --------
        set_default_constraints

        Examples
        --------
        ::

            lc.set_model("1D", num_mixtures=4)
            lc.set_default_constraints()
            constraints = lc.get_constraints()
        """
        if not hasattr(self, "_model_pars"):
            raise RuntimeError(
                "Model has not been set yet. Call set_model() before "
                "get_constraints()."
            )
        constraints = {}
        for name, constraint in self.model.named_constraints():
            constraints[name] = constraint
        print("Registered constraints:")
        if constraints:
            for name, constraint in constraints.items():
                print(f"  {name}: {constraint}")
        else:
            print("  (none)")
        return constraints

    def set_hypers(self, hypers=None, debug=False, **kwargs):
        """Set the hyperparameters for the model and likelihood. This is a
        convenience function that calls the model.initialize() to set the
        hyperparameters. However, first it applies any transforms to the
        hyperparameters, so that the user can pass the hyperparameters in
        the original data space if they wish.

        Parameters
        ----------
        hypers : dict, optional
            A dictionary of the hyperparameters to use for the model and
            likelihood. The keys should be the names of the parameters, and the
            values should be Tensors containing the values of the parameters.
            If None, no hyperparameters will be set. If a hyperparameter is
            passed for a parameter that is not a model or likelihood
            parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the initialize.
        """

        if hypers is None:
            return
        pars_to_transform = {
            "x": ["mixture_means", "mixture_scales"],
            "y": ["noise", "mean_module"],
        }
        if debug:
            print("hypers before transform:")
            print(hypers)
        for key in hypers:
            # first, check if the parameter needs to be transformed:
            if any(p in key for p in pars_to_transform["x"]):
                # now apply the x transform
                # remember that the means and scales are in fourier space
                # so we need to transform them back to real space
                # before applying the transform
                # and then transform them back to fourier space
                # luckily, when the shift is removed from the transform,
                # the factors of 2pi cancel out for the scales
                # so we can just do 1/ for both means and scales
                if self.xtransform is not None:
                    if debug:
                        print(f"Applying x-transform to {key}")
                    # Check if the parameter is 2D (for multi-dimensional data)
                    if hypers[key].dim() == 2:
                        # For 2D hyperparameters (num_mixtures, ard_num_dims),
                        # the transform should be applied considering each
                        # dimension's range. Since transform was fit on
                        # (n_samples, 2) data, we need to handle this
                        # carefully
                        _num_mixtures, ard_num_dims = hypers[key].shape
                        transformed = torch.zeros_like(hypers[key])

                        # For each dimension of the 2D parameter
                        for dim in range(ard_num_dims):
                            # Get the range for this dimension from the
                            # fitted transformer
                            if (
                                hasattr(self.xtransform, "range")
                                and self.xtransform.range.shape[0] > dim
                            ):
                                # Apply dimension-specific scaling to the
                                # Fourier space parameters
                                # Formula: f_transformed = 1 / ((1 / f_raw)
                                # / range)
                                # This accounts for the data transformation
                                # applied to each dimension
                                # The 1/x transformations handle the Fourier
                                # space representation
                                dim_values = hypers[key][:, dim]
                                # Transform back to real space, apply
                                # scaling, then back to Fourier
                                transformed[:, dim] = 1 / (
                                    (1 / dim_values) / self.xtransform.range[0, dim]
                                )
                            else:
                                # Fallback: just copy the values
                                transformed[:, dim] = hypers[key][:, dim]
                        hypers[key] = transformed
                    else:
                        # 1D case - original behavior
                        hypers[key] = 1 / self.xtransform.transform(
                            1 / hypers[key], shift=False
                        )
            elif any(p in key for p in pars_to_transform["y"]):
                # now apply the y transform
                # the mean function and noise are not defined in fourier
                # space, so we can just apply the transform directly
                if self.ytransform is not None:
                    if debug:
                        print(f"Applying y-transform to {key}")
                    hypers[key] = self.ytransform.transform(hypers[key])
        if debug:
            print("hypers after transform:")
            print(hypers)
        self.model.initialize(**hypers, **kwargs)

    def init_hypers_from_LombScargle(self, **kwargs):
        pass

    def _set_hypers_raw(self, hypers=None, **kwargs):
        pass

    def cpu(self):
        self.device = torch.device("cpu")
        super().cpu()
        for _ in dict_walk_generator(self._model_pars):
            with contextlib.suppress(AttributeError):
                _.cpu()

    def cuda(self, device=0):
        # First we should check that CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot call cuda() if CUDA is not available")
        # next we should log that we're using CUDA
        self.device = torch.device(f"cuda:{device}")

        # now we need to make sure that the usual nn.Module.cuda() method
        # is called, so that all of the modules and buffers are moved to the GPU
        super().cuda(device=device)

        # but we've created a few extra things that need to be tracked.
        # We have to make sure any tensors in those are also moved to the same device
        for _ in dict_walk_generator(self._model_pars):
            with contextlib.suppress(AttributeError):
                _.cuda(device=device)

        # for key in self._model_pars:
        #     with contextlib.suppress(AttributeError):
        #         self._model_pars[key]['param'] = self._model_pars[key]['param'].cuda(device=device) # noqa: E501
        # try:
        #     self.model.cuda()
        #     self.likelihood.cuda()
        # except AttributeError as e:
        #     errmsg = "You must first set the model and likelihood"
        #     _reraise_with_note(e, errmsg)

    def _train(self):
        try:
            self.model.train()
            self.likelihood.train()
        except AttributeError as e:
            errmsg = "You must first set the model and likelihood"
            _reraise_with_note(e, errmsg)

    def _eval(self):
        try:
            self.model.eval()
            self.likelihood.eval()
        except AttributeError as e:
            errmsg = "You must first set the model and likelihood"
            _reraise_with_note(e, errmsg)

    def fit_LS(
        self,
        freq_only: bool = False,
        num_peaks: int = 1,
        single_threshold: float = 0.05,
        Nyquist_factor: int = 5,
        fap_method: str | None = None,
        use_best_band_init: bool = False,
        return_full: bool = False,
        **kwargs,
    ) -> tuple:
        """
        Compute the (multiband) Lomb-Scargle periodogram.
        Periods returned for the num_peaks highest peaks in the periodogram.
        For a 1D lightcurve, the false-alarm probability is used
            to estimate the significance of the periods, which are also
            returned. These can be used to filter out insignificant periods.
        For multi-band lightcurves (2D data), LombScargleMultiband is used
            to compute periods across all bands simultaneously.

        The method can also be used to return the entire grid of frequencies,
        which can be used by other methods such as compute_psd and plot_psd.

        Parameters:
        ----------------
        - freq_only: bool, optional, default=False
            If True, only the frequency grid will be returned.
            This can be useful for methods such as compute_psd and plot_psd.
        - num_peaks: int, optional, default=1
            The number of peaks to extract from the Lomb-Scargle periodogram.
            If fewer peaks are found, only the available peaks will be returned.
        - single_threshold: float, optional, default=0.05
            The false alarm probability threshold for a single peak to be
            considered significant.
        - Nyquist_factor: int, optional, default=5
            The factor by which to multiply the Nyquist frequency to
            determine the maximum frequency to search for in the
            Lomb-Scargle periodogram.
            This will be approximately the number of points sampling
            the maximum in the resulting periodogram.
        - fap_method: str or None, optional, default=None
            Method used to compute the false-alarm probability (FAP) of the
            *maximum* periodogram peak (the global significance test).
            For 1D lightcurves the default is ``'davies'`` (fast analytical
            upper bound; equivalent to ``'baluev'`` for practical purposes
            but significantly faster). Other valid astropy options are
            ``'baluev'`` and ``'bootstrap'``. Note: ``'single'`` is a
            valid astropy option that computes the FAP for a single
            pre-specified frequency and is not appropriate for ``fap_max``
            (a warning is issued and ``'baluev'`` is used instead); it is
            however used internally as the per-frequency p-value when
            applying the Benjamini-Hochberg correction.
            For multi-band lightcurves the default is ``'phase_scramble'``.
            Slower but more accurate options are ``'bootstrap'``, ``'calibrated'``,
            and ``'analytical'``  (fast Baluev-style approximation) (see
            :class:`~pgmuvi.multiband_ls_significance.MultibandLSWithSignificance`).
        - use_best_band_init: bool, optional, default=False
            If True and the lightcurve is multiband (ndim > 1), the
            Lomb-Scargle frequency grid is derived from the band with the
            most observations rather than from the full multiband dataset.
            This yields a finer frequency resolution focused on the most
            informative band, which can speed up and improve the
            periodogram search when sampling is highly heterogeneous
            across bands.  Has no effect for 1D lightcurves.
        - return_full: bool, optional, default=False
            If True and ``freq_only=False``, also return the complete
            frequency grid and power spectrum alongside the peak frequencies
            and significance mask (see return values below).  Ignored when
            ``freq_only=True``.  The periodogram itself is not recomputed,
            but returning the full grid may still allocate and/or copy the
            frequency and power tensors before returning them.
        - kwargs: dict, optional
            Additional keyword arguments to be passed to the
            LombScargle(Multiband) constructor.

        Returns:
        ----------------
        The return value depends on the combination of ``freq_only`` and
        ``return_full``:

        * ``freq_only=True`` (``return_full`` is ignored):
          ``(freq_grid, power_grid)``

          - freq_grid: torch.Tensor of floats — the full frequency grid.
          - power_grid: torch.Tensor of floats — periodogram power at each
            frequency.

        * ``freq_only=False, return_full=False`` (default):
          ``(peak_freqs, significance_mask)``

          - peak_freqs: torch.Tensor of floats — frequencies of the
            ``num_peaks`` highest periodogram peaks.
          - significance_mask: torch.Tensor of bool — True for peaks that
            are statistically significant after Benjamini-Hochberg FDR
            correction.

        * ``freq_only=False, return_full=True``:
          ``(peak_freqs, significance_mask, freq_grid, power_grid)``

          - peak_freqs: torch.Tensor of floats — as above.
          - significance_mask: torch.Tensor of bool — as above.
          - freq_grid: torch.Tensor of floats — the full frequency grid
            (already computed internally; returned at no extra cost).
          - power_grid: torch.Tensor of floats — periodogram power at each
            frequency (already computed internally).
        """
        from astropy.timeseries import LombScargle
        from scipy.signal import find_peaks
        from .multiband_ls_significance import MultibandLSWithSignificance

        def fdr_bh(fap_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
            """
            Benjamini-Hochberg procedure to control the False Discovery Rate.

            The Benjamini-Hochberg (BH) procedure is a method for controlling the
            False Discovery Rate (FDR) when performing multiple hypothesis tests.
            It works by:
            1. Sorting the p-values (FAPs) in ascending order
            2. Finding the largest i such that p(i) <= (i/N) * alpha
            3. Rejecting all hypotheses with p-values <= p(i)

            This is less conservative than Bonferroni correction while still
            controlling the expected proportion of false discoveries.

            See https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.fdrcorrection.html
                for the statsmodels implementation

            Parameters:
            ----------------
            - fap_values: Array of false alarm probabilities (FAP) for peaks.
            - alpha: Desired FDR threshold (e.g., 0.05 for 5% FDR).

            Returns:
            ----------------
            - result: array(bool), True for statistically significant peaks.
            """
            # Sort FAP values in ascending order and get their original indices
            sorted_indices = np.argsort(fap_values)
            sorted_fap = fap_values[sorted_indices]
            N = len(fap_values)
            # Find the largest i such that fap(i) <= (i / N) * alpha
            threshold = np.arange(1, N + 1) / N * alpha
            significant = sorted_fap <= threshold
            # If there are significant results, keep the largest index
            if np.any(significant):
                max_signif_index = np.where(significant)[0].max()
                significant_indices = sorted_indices[: max_signif_index + 1]
                result = np.zeros(N, dtype=bool)
                result[significant_indices] = True
            else:
                result = np.zeros(N, dtype=bool)
            return result

        # Build working arrays from the stored (already finite) data.
        _has_yerr = (
            hasattr(self, "_yerr_transformed")
            and self._yerr_transformed is not None
        )
        _xdata = self.xdata
        _ydata = self.ydata
        _yerr = self.yerr if _has_yerr else None

        if self.ndim > 1:
            # Multi-band case: _xdata[:, 0] is time, _xdata[:, 1] is band/wavelength
            t = _xdata[:, 0]
            bands = _xdata[:, 1]
            y = _ydata

            # Default FAP method for multiband: phase_scramble
            _fap_method = fap_method if fap_method is not None else 'phase_scramble'

            if _yerr is not None:
                yerr = _yerr
                LS = MultibandLSWithSignificance(t, y, bands, dy=yerr, **kwargs)
            else:
                LS = MultibandLSWithSignificance(t, y, bands, **kwargs)

            if use_best_band_init:
                # Use the most-sampled band's 1D autofrequency as the grid
                # for the multiband LS.  The best-sampled band has finer
                # temporal resolution (more data points), yielding a denser
                # frequency grid that improves period recovery when sampling
                # is highly heterogeneous across bands.
                _unique_bands, _band_counts = torch.unique(
                    bands, return_counts=True
                )
                _best_val = _unique_bands[_band_counts.argmax()]
                _best_mask = bands == _best_val
                _t_best = t[_best_mask].detach().cpu().numpy()
                _y_best = y[_best_mask].detach().cpu().numpy()
                if _has_yerr:
                    _dy_best = yerr[_best_mask].detach().cpu().numpy()
                    _ls_1d_best = LombScargle(_t_best, _y_best, _dy_best)
                else:
                    _ls_1d_best = LombScargle(_t_best, _y_best)
                freq = _ls_1d_best.autofrequency(nyquist_factor=Nyquist_factor)
                power = _ls_1d_best.power(freq)
            else:
                freq = LS.autofrequency(nyquist_factor=Nyquist_factor)
                power = LS.power(freq)

            if freq_only:
                return (
                    torch.as_tensor(
                        freq, dtype=self.xdata.dtype, device=self.xdata.device
                    ),
                    torch.as_tensor(
                        power, dtype=self.xdata.dtype, device=self.xdata.device
                    ),
                )

            # Build full-grid tensors only when they are requested to avoid
            # unnecessary allocation/copy on the default path.
            if return_full:
                _freq_t = torch.as_tensor(
                    freq, dtype=self.xdata.dtype, device=self.xdata.device
                )
                _power_t = torch.as_tensor(
                    power, dtype=self.xdata.dtype, device=self.xdata.device
                )

            # Find peaks in the multiband periodogram
            peaks, _ = find_peaks(power, distance=Nyquist_factor)
            peaks = peaks[np.argsort(power[peaks])][::-1]

            # Handle case when no peaks found
            if len(peaks) == 0:
                _pf = torch.as_tensor(
                    [], dtype=self.xdata.dtype, device=self.xdata.device
                )
                _sm = torch.as_tensor(
                    [], dtype=torch.bool, device=self.xdata.device
                )
                # return_full=True exposes already-computed LS intermediates
                if return_full:
                    return (_pf, _sm, _freq_t, _power_t)
                return (_pf, _sm)

            # Compute FAP for multiband periodogram
            fap_max = LS.false_alarm_probability(power.max(),
                                                 method=_fap_method,
                                                 freq_grid=freq)
            n_return = min(num_peaks, len(peaks))

            if fap_max > single_threshold:
                # Highest peak is not significant, mark all as insignificant
                _pf = torch.as_tensor(
                    freq[peaks[:n_return]],
                    dtype=self.xdata.dtype,
                    device=self.xdata.device,
                )
                _sm = torch.as_tensor(
                    np.array([False] * n_return),
                    dtype=torch.bool,
                    device=self.xdata.device,
                )
                # return_full=True exposes already-computed LS intermediates
                if return_full:
                    return (_pf, _sm, _freq_t, _power_t)
                return (_pf, _sm)

            # Calculate FAP for each peak independently
            fap_single = LS.false_alarm_probability(power[peaks],
                                                    method=_fap_method,
                                                    freq_grid=freq)

            # Apply the FDR (Benjamini-Hochberg) correction
            significant_mask = fdr_bh(fap_single, alpha=single_threshold)
            significant_mask[0] = True  # since fap_max <= single_threshold

            _pf = torch.as_tensor(
                freq[peaks[:n_return]],
                dtype=self.xdata.dtype,
                device=self.xdata.device,
            )
            _sm = torch.as_tensor(
                significant_mask[:n_return],
                dtype=torch.bool,
                device=self.xdata.device,
            )
            # return_full=True exposes already-computed LS intermediates
            if return_full:
                return (_pf, _sm, _freq_t, _power_t)
            return (_pf, _sm)
        else:
            t, y = _xdata, _ydata

            # Default FAP method for single-band: 'davies' (fast analytical
            # upper bound; same accuracy as 'baluev' for typical use cases
            # but much faster). 'baluev' is another good analytical choice.
            _fap_method = fap_method if fap_method is not None else 'davies'

            if _yerr is not None:
                yerr = _yerr
                LS = LombScargle(t, y, yerr)
            else:
                LS = LombScargle(t, y)
            freq = LS.autofrequency(nyquist_factor=Nyquist_factor)
            # assume_regular_frequency=True: autofrequency() always produces
            # a regular grid, so skip the regularity check for a minor speedup
            power = LS.power(freq, assume_regular_frequency=True)
            if freq_only:
                return (
                    torch.as_tensor(
                        freq, dtype=self.xdata.dtype, device=self.xdata.device
                    ),
                    torch.as_tensor(
                        power, dtype=self.xdata.dtype, device=self.xdata.device
                    ),
                )

            # Build full-grid tensors only when they are requested to avoid
            # unnecessary allocation/copy on the default path.
            if return_full:
                _freq_t = torch.as_tensor(
                    freq, dtype=self.xdata.dtype, device=self.xdata.device
                )
                _power_t = torch.as_tensor(
                    power, dtype=self.xdata.dtype, device=self.xdata.device
                )

            # distance set to Nyquist_factor for LS frequency grid computation
            peaks, _ = find_peaks(power, distance=Nyquist_factor)
            # sort by decreasing power
            peaks = peaks[np.argsort(power[peaks])][::-1]

            # Handle case when no peaks or fewer peaks than requested
            if len(peaks) == 0:
                # No peaks found, return empty tensors
                _pf = torch.as_tensor(
                    [], dtype=self.xdata.dtype, device=self.xdata.device
                )
                _sm = torch.as_tensor(
                    [], dtype=torch.bool, device=self.xdata.device
                )
                # return_full=True exposes already-computed LS intermediates
                if return_full:
                    return (_pf, _sm, _freq_t, _power_t)
                return (_pf, _sm)

            # Calculate the false alarm probability for the highest peak.
            # 'single' is not appropriate for fap_max (it computes the FAP
            # for a single pre-specified frequency, not the global maximum).
            _fap_method_max = _fap_method
            if _fap_method_max == 'single':
                warnings.warn(
                    "fap_method='single' is not appropriate for the false alarm "
                    "probability of the maximum peak (it computes the FAP for a "
                    "single pre-specified frequency, not the global maximum). "
                    "Using method='baluev' for fap_max instead.",
                    UserWarning,
                    stacklevel=2,
                )
                _fap_method_max = 'baluev'
            fap_max = LS.false_alarm_probability(power.max(),
                                                 method=_fap_method_max)
            n_return = min(num_peaks, len(peaks))

            if fap_max > single_threshold:
                _pf = torch.as_tensor(
                    freq[peaks[:n_return]],
                    dtype=self.xdata.dtype,
                    device=self.xdata.device,
                )
                _sm = torch.as_tensor(
                    np.array([False] * n_return),
                    dtype=torch.bool,
                    device=self.xdata.device,
                )
                # return_full=True exposes already-computed LS intermediates
                if return_full:
                    return (_pf, _sm, _freq_t, _power_t)
                return (_pf, _sm)
            # Per-peak FAP for the Benjamini-Hochberg correction.
            # We use method='single' here: it gives the single-frequency FAP
            # (probability that one pre-specified frequency shows at least
            # this power by chance), which is the correct uncorrected p-value
            # to supply to BH.  The 'davies'/'baluev' methods already account
            # for multiple-frequency comparisons and would be too conservative.
            fap_single = LS.false_alarm_probability(power[peaks],
                                                    method='single')
            # Apply the FDR (Benjamini-Hochberg) correction
            significant_mask = fdr_bh(fap_single, alpha=single_threshold)
            significant_mask[0] = True  # since fap_max <= single_threshold
            _pf = torch.as_tensor(
                freq[peaks[:n_return]],
                dtype=self.xdata.dtype,
                device=self.xdata.device,
            )
            _sm = torch.as_tensor(
                significant_mask[:n_return],
                dtype=torch.bool,
                device=self.xdata.device,
            )
            # return_full=True exposes already-computed LS intermediates
            if return_full:
                return (_pf, _sm, _freq_t, _power_t)
            return (_pf, _sm)

    def acf(
        self,
        method="data",
        max_lag=None,
        n_lags=50,
        lag_edges=None,
        normalize=True,
        subtract_mean=True,
        band=None,
        reference_time=None,
    ):
        """Compute the autocorrelation function (ACF) of the light curve.

        Parameters
        ----------
        method : {"data", "gp"}, optional
            ``"data"`` estimates the ACF directly from the observations using
            a pairwise lag-binning estimator (handles uneven sampling).
            ``"gp"`` evaluates the ACF implied by the fitted GP covariance
            function.  Default is ``"data"``.
        max_lag : float or None, optional
            Maximum lag to consider.  If ``None``, defaults to half the
            total time baseline.
        n_lags : int, optional
            Number of lag bins (``method="data"``) or evaluation points
            (``method="gp"``).  Default is 50.  Must be a positive integer.
            For ``method="data"`` the returned arrays have length
            ``n_lags + 1`` because a zero-lag point is prepended.
        lag_edges : array-like or None, optional
            Explicit bin edges for the lag axis (``method="data"`` only).
            If provided, *n_lags* and *max_lag* are ignored.
        normalize : bool, optional
            If ``True``, normalise the ACF so that ``acf(0) == 1``.
            Default is ``True``.
        subtract_mean : bool, optional
            If ``True``, subtract the sample mean before computing products
            (``method="data"`` only).  Default is ``True``.
        band : str or None, optional
            Band label to use when the light curve is 2D (multiband).
            For ``method="data"`` this argument is **required** when the
            light curve is 2D; a :exc:`ValueError` is raised if it is
            ``None``.  For ``method="gp"`` with a 2D light curve the call
            always raises :exc:`NotImplementedError` regardless of *band*,
            because GP-implied ACF for 2D kernels is not yet implemented.
        reference_time : float or None, optional
            Reference time used to anchor the GP covariance evaluation
            (``method="gp"`` only).  If ``None``, the mean of the time axis
            is used.

        Returns
        -------
        ACFResult
            A :class:`ACFResult` instance with attributes ``lag``, ``acf``,
            ``method``, ``counts`` (data method only), ``normalized``, and
            ``band``.

        Raises
        ------
        ValueError
            If *method* is not ``"data"`` or ``"gp"``.
        ValueError
            If *n_lags* is not a positive integer.
        ValueError
            If the light curve is 2D and ``method="data"`` and *band* is
            ``None``.
        ValueError
            If *lag_edges* is invalid (not 1-D, fewer than two values,
            not strictly increasing, or contains non-finite values).
        RuntimeError
            If ``method="gp"`` is requested but :meth:`fit` has not been
            called yet.
        RuntimeError
            If ``method="gp"`` with ``normalize=True`` and the zero-lag
            covariance is non-positive or non-finite.
        NotImplementedError
            If ``method="gp"`` is used with a 2D light curve.

        Examples
        --------
        >>> result = lc.acf(method="data")
        >>> result = lc.acf(method="gp", n_lags=500)
        """
        if method not in ("data", "gp"):
            raise ValueError(
                f"Unknown method {method!r}. Choose 'data' or 'gp'."
            )

        # ------------------------------------------------------------------
        # Handle 2D (multiband) light curves
        # ------------------------------------------------------------------
        if self.ndim > 1:
            if method == "gp":
                raise NotImplementedError(
                    "GP-implied ACF for 2D (multiband) light curves is not "
                    "implemented yet. Fixed-wavelength covariance evaluation "
                    "requires additional support not yet in place."
                )
            # method == "data"
            if band is None:
                raise ValueError(
                    "This is a 2D (multiband) light curve. "
                    "You must specify 'band' to select a single band before "
                    "computing the ACF."
                )
            lc_band = self.select_bands([str(band)])
            return lc_band._acf_data(
                max_lag=max_lag,
                n_lags=n_lags,
                lag_edges=lag_edges,
                normalize=normalize,
                subtract_mean=subtract_mean,
                band=band,
            )

        if method == "data":
            return self._acf_data(
                max_lag=max_lag,
                n_lags=n_lags,
                lag_edges=lag_edges,
                normalize=normalize,
                subtract_mean=subtract_mean,
                band=band,
            )
        # method == "gp"
        return self._acf_gp(
            max_lag=max_lag,
            n_lags=n_lags,
            normalize=normalize,
            band=band,
            reference_time=reference_time,
        )

    def _get_time_axis(self):
        """Return the 1-D time axis from xdata (column 0 for 2D data)."""
        return self.xdata[:, 0] if self.xdata.dim() > 1 else self.xdata

    def _default_max_lag(self, t):
        """Return half the time baseline as the default max lag."""
        max_lag = (t.max() - t.min()) / 2.0
        return max_lag.detach().item()

    @staticmethod
    def _validate_n_lags(n_lags):
        """Raise ValueError if *n_lags* is not a positive integer."""
        if not isinstance(n_lags, (int, np.integer)) or n_lags < 1:
            raise ValueError(
                f"n_lags must be a positive integer; got {n_lags!r}."
            )

    def _acf_data(
        self,
        max_lag=None,
        n_lags=50,
        lag_edges=None,
        normalize=True,
        subtract_mean=True,
        band=None,
    ):
        """Pairwise lag-binning ACF estimator for unevenly sampled data.

        Returns an array of length ``n_lags + 1`` (or ``M + 1`` for *M*
        explicit bins): the first element is the zero-lag point, followed by
        the lag-bin centres.

        This is an O(n²) algorithm in the number of observations. For very
        large datasets (n > ~1000) the computation may be slow.
        """
        t = self._get_time_axis().double()
        y = self.ydata.double()

        # ------------------------------------------------------------------
        # Build lag-bin edges
        # ------------------------------------------------------------------
        if lag_edges is not None:
            edges = torch.as_tensor(
                lag_edges, dtype=torch.float64, device=t.device
            )
            if edges.dim() != 1:
                raise ValueError(
                    "lag_edges must be a 1-D array; "
                    f"got shape {tuple(edges.shape)}."
                )
            if edges.shape[0] < 2:
                raise ValueError(
                    "lag_edges must contain at least two values "
                    f"(got {edges.shape[0]})."
                )
            if not torch.all(torch.isfinite(edges)):
                raise ValueError(
                    "lag_edges must contain only finite values."
                )
            diffs = edges[1:] - edges[:-1]
            if not torch.all(diffs > 0):
                raise ValueError(
                    "lag_edges must be strictly increasing."
                )
            max_lag = edges[-1].detach().item()
            n_bins = edges.shape[0] - 1
        else:
            self._validate_n_lags(n_lags)
            if max_lag is None:
                max_lag = self._default_max_lag(t)
            edges = torch.linspace(0.0, max_lag, n_lags + 1, dtype=t.dtype, device=t.device)
            n_bins = n_lags

        mean = y.mean() if subtract_mean else 0.0
        y_centered = y - mean
        n = t.shape[0]

        # ------------------------------------------------------------------
        # Zero-lag entry: all n self-pairs
        # ------------------------------------------------------------------
        variance = (y_centered**2).mean().detach().item()
        if normalize and not (np.isfinite(variance) and variance > 0):
            raise RuntimeError(
                "Cannot normalize ACF: variance is zero or non-finite. "
                "Input light curve may be constant."
            )
        zero_lag_val = variance  # before normalisation

        # ------------------------------------------------------------------
        # Vectorised pairwise computation (i < j pairs)
        # ------------------------------------------------------------------
        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=t.device)
        dt = (t[idx_j] - t[idx_i]).abs()
        vals = y_centered[idx_i] * y_centered[idx_j]

        # Keep only pairs within max_lag
        in_range = dt <= max_lag
        dt = dt[in_range]
        vals = vals[in_range]

        # Bin dt into lag bins
        bin_idx = torch.searchsorted(edges, dt) - 1
        valid_bins = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx[valid_bins]
        vals = vals[valid_bins]

        bin_sums = torch.zeros(n_bins, dtype=torch.float64, device=t.device)
        bin_counts = torch.zeros(n_bins, dtype=torch.float64, device=t.device)
        bin_sums.scatter_add_(0, bin_idx, vals)
        bin_counts.scatter_add_(0, bin_idx, torch.ones_like(vals))

        # Normalise by counts
        valid = bin_counts > 0
        acf_vals = torch.zeros(n_bins, dtype=torch.float64, device=t.device)
        acf_vals[valid] = bin_sums[valid] / bin_counts[valid]

        if normalize:
            acf_vals = acf_vals / variance
            zero_lag_val = 1.0

        lag_centers = (edges[:-1] + edges[1:]) / 2.0

        # Prepend zero-lag point
        zero_lag_tensor = torch.zeros(1, dtype=torch.float64, device=t.device)
        zero_acf_tensor = torch.tensor(
            [zero_lag_val], dtype=torch.float64, device=t.device
        )
        zero_count_tensor = torch.tensor(
            [float(n)], dtype=torch.float64, device=t.device
        )

        full_lag = torch.cat([zero_lag_tensor, lag_centers])
        full_acf = torch.cat([zero_acf_tensor, acf_vals])
        full_counts = torch.cat([zero_count_tensor, bin_counts])

        return ACFResult(
            lag=full_lag.float(),
            acf=full_acf.float(),
            method="data",
            counts=full_counts.float(),
            normalized=normalize,
            band=band,
        )

    def _acf_gp(
        self,
        max_lag=None,
        n_lags=500,
        normalize=True,
        band=None,
        reference_time=None,
    ):
        """GP-implied ACF evaluated on a uniform lag grid."""
        self._validate_n_lags(n_lags)

        # Use the MAP-fitted flag; set_model() alone is not sufficient.
        if not self.__FITTED_MAP:
            raise RuntimeError(
                "No fitted GP model found. Call fit() before using "
                "method='gp'."
            )

        t = self._get_time_axis()

        if max_lag is None:
            max_lag = self._default_max_lag(t)

        if reference_time is None:
            t_ref = t.mean().detach().item()
        else:
            t_ref = float(reference_time)

        tau = torch.linspace(0.0, max_lag, n_lags, dtype=t.dtype, device=t.device)
        x1 = torch.full((n_lags,), t_ref, dtype=t.dtype, device=t.device)
        x2 = x1 + tau

        # Apply the same input transformation used during training
        if self.xtransform is not None:
            x1 = self.xtransform.transform(x1)
            x2 = self.xtransform.transform(x2)

        # Unsqueeze to (n, 1) as expected by GPyTorch kernels
        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            lazy_cov = self.model.covar_module(x1, x2)
            # Extract diagonal: K(x1[i], x2[i]) = K(t_ref, t_ref + tau[i])
            acf_vals = lazy_cov.diagonal().detach()

            if normalize:
                x_ref = x1[:1]
                lazy_var = self.model.covar_module(x_ref, x_ref)
                variance = lazy_var.diagonal()[0].detach().item()
                if not (np.isfinite(variance) and variance > 0):
                    raise RuntimeError(
                        f"Zero-lag GP covariance is {variance!r}, which is "
                        "non-positive or non-finite. Cannot normalise ACF."
                    )
                acf_vals = acf_vals / variance

        return ACFResult(
            lag=tau.detach(),
            acf=acf_vals.detach(),
            method="gp",
            counts=None,
            normalized=normalize,
            band=band,
        )

    def plot_acf(
        self,
        method="data",
        ax=None,
        **kwargs,
    ):
        """Plot the autocorrelation function of the light curve.

        Calls :meth:`acf` and plots the result.

        Parameters
        ----------
        method : {"data", "gp"}, optional
            Which ACF to compute and plot.  Default is ``"data"``.
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw on.  If ``None``, a new figure and axes are created.
        **kwargs
            Additional keyword arguments passed to :meth:`acf`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the ACF plotted.

        Examples
        --------
        >>> ax = lc.plot_acf(method="data")
        >>> ax = lc.plot_acf(method="gp", n_lags=500)
        """
        result = self.acf(method=method, **kwargs)

        if ax is None:
            _, ax = plt.subplots()

        lag_np = result.lag.detach().cpu().numpy()
        acf_np = result.acf.detach().cpu().numpy()

        ax.plot(lag_np, acf_np)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        title_method = "Data (pairwise)" if method == "data" else "GP model"
        ax.set_title(f"Autocorrelation Function ({title_method})")

        return ax

    def compute_sampling_metrics(self) -> dict:
        """
        Compute temporal sampling quality metrics.

        Returns
        -------
        dict
            Comprehensive sampling metrics (see
            preprocess.quality.compute_sampling_metrics)

        Examples
        --------
        >>> lc = Lightcurve(t, y, yerr)
        >>> metrics = lc.compute_sampling_metrics()
        >>> print(f"Nyquist period: {metrics['nyquist_period']:.2f}")
        """
        from pgmuvi.preprocess.quality import compute_sampling_metrics

        t = self._xdata_raw.detach().cpu().numpy()
        if t.ndim > 1:
            t = t[:, 0]
        y = (
            self._ydata_raw.detach().cpu().numpy()
            if hasattr(self, "_ydata_raw")
            else None
        )
        yerr = (
            self._yerr_raw.detach().cpu().numpy()
            if hasattr(self, "_yerr_raw")
            else None
        )
        return compute_sampling_metrics(t, y, yerr)

    def assess_sampling_quality(self, verbose: bool = True, **kwargs) -> tuple:
        """
        Assess whether lightcurve sampling is adequate for GP fitting.

        Parameters
        ----------
        verbose : bool, default=True
            Print detailed assessment report
        **kwargs : dict
            Quality gate thresholds (see
            preprocess.quality.assess_sampling_quality):

            - min_points: int (default 15)
            - max_gap_fraction: float (default 0.3)
            - min_baseline_factor: float (default 3.0)
            - min_snr: float (default 3.0)
            - min_fraction_good_snr: float (default 0.5)

        Returns
        -------
        passes : bool
            True if all quality gates pass
        diagnostics : dict
            Diagnostic information including gates, metrics, warnings,
            and recommendation

        Examples
        --------
        >>> lc = Lightcurve(t, y, yerr)
        >>> passes, diag = lc.assess_sampling_quality(verbose=True)
        >>> if diag['recommendation'] == 'PROCEED':
        ...     lc.fit(...)
        """
        from pgmuvi.preprocess.quality import assess_sampling_quality

        t = self._xdata_raw.detach().cpu().numpy()
        if t.ndim > 1:
            t = t[:, 0]
        y = (
            self._ydata_raw.detach().cpu().numpy()
            if hasattr(self, "_ydata_raw")
            else None
        )
        yerr = (
            self._yerr_raw.detach().cpu().numpy()
            if hasattr(self, "_yerr_raw")
            else None
        )
        passes, diagnostics = assess_sampling_quality(
            t, y, yerr, verbose=verbose, **kwargs
        )
        return passes, diagnostics

    def compute_sampling_metrics_per_band(self) -> dict:
        """
        Compute sampling metrics independently for each wavelength band.

        Only applicable for 2D (multiband) lightcurves.

        Returns
        -------
        dict
        {
                wavelength1: metrics_dict,
                wavelength2: metrics_dict,
                ...
                'summary': {
                    'n_bands': int,
                    'min_points_across_bands': int,
                    'max_gap_fraction_worst_band': float,
                    'median_nyquist_period': float
                }
            }

        Raises
        ------
        ValueError
            If lightcurve is not 2D (multiband).
        """
        from pgmuvi.preprocess.quality import compute_sampling_metrics

        if self.ndim <= 1:
            raise ValueError(
                "compute_sampling_metrics_per_band() requires 2D (multiband) data. "
                "Use compute_sampling_metrics() for 1D data."
            )

        xdata = self._xdata_raw.detach().cpu().numpy()
        ydata = self._ydata_raw.detach().cpu().numpy()
        yerr = (
            self._yerr_raw.detach().cpu().numpy()
            if hasattr(self, "_yerr_raw")
            else None
        )

        wavelengths = np.unique(xdata[:, 1])
        results = {}

        min_points_list = []
        max_gaps_list = []
        nyquist_list = []

        for wl in wavelengths:
            mask = xdata[:, 1] == wl
            t = xdata[mask, 0]
            y = ydata[mask]
            ye = yerr[mask] if yerr is not None else None

            metrics = compute_sampling_metrics(t, y, ye)
            results[float(wl)] = metrics

            if "n_points" in metrics:
                min_points_list.append(metrics["n_points"])
            if "max_gap_fraction" in metrics:
                max_gaps_list.append(metrics["max_gap_fraction"])
            if "nyquist_period" in metrics:
                nyquist_list.append(metrics["nyquist_period"])

        results["summary"] = {
            "n_bands": len(wavelengths),
            "min_points_across_bands": min(min_points_list) if min_points_list else 0,
            "max_gap_fraction_worst_band": (
                max(max_gaps_list) if max_gaps_list else np.inf
            ),
            "median_nyquist_period": (
                float(np.median(nyquist_list)) if nyquist_list else np.inf
            ),
        }

        return results

    def assess_sampling_quality_per_band(
        self, verbose: bool = True, **kwargs
    ) -> dict:
        """
        Assess sampling quality independently for each wavelength band.

        Only applicable for 2D (multiband) lightcurves.

        Parameters
        ----------
        verbose : bool, default=True
            Print assessment for each band
        **kwargs : dict
            Quality gate thresholds

        Returns
        -------
        dict
            {
                wavelength1: diagnostics_dict,
                wavelength2: diagnostics_dict,
                ...
                'summary': {
                    'n_bands': int,
                    'n_passing': int,
                    'passing_wavelengths': list[float],
                    'failing_wavelengths': list[float]
                    }
            }

        Raises
        ------
        ValueError
            If lightcurve is not 2D (multiband).
        """
        from pgmuvi.preprocess.quality import assess_sampling_quality

        if self.ndim <= 1:
            raise ValueError(
                "assess_sampling_quality_per_band() requires 2D (multiband) data. "
                "Use assess_sampling_quality() for 1D data."
            )

        xdata = self._xdata_raw.detach().cpu().numpy()
        ydata = self._ydata_raw.detach().cpu().numpy()
        yerr = (
            self._yerr_raw.detach().cpu().numpy()
            if hasattr(self, "_yerr_raw")
            else None
        )

        wavelengths = np.unique(xdata[:, 1])
        results = {}
        passing_bands = []
        failing_bands = []

        for wl in wavelengths:
            mask = xdata[:, 1] == wl
            t = xdata[mask, 0]
            y = ydata[mask]
            ye = yerr[mask] if yerr is not None else None

            if verbose:
                print(f"\n{'=' * 70}")
                print(f"BAND: \u03bb = {wl}")
                print(f"{'=' * 70}")

            passes, diag = assess_sampling_quality(t, y, ye, verbose=verbose, **kwargs)
            results[float(wl)] = diag

            if passes:
                passing_bands.append(float(wl))
            else:
                failing_bands.append(float(wl))

        results["summary"] = {
            "n_bands": len(wavelengths),
            "n_passing": len(passing_bands),
            "passing_wavelengths": passing_bands,
            "failing_wavelengths": failing_bands,
        }

        return results

    def filter_well_sampled_bands(self, **kwargs):
        """
        Create new Lightcurve with only well-sampled bands retained.

        Only applicable for 2D (multiband) lightcurves.

        Parameters
        ----------
        **kwargs : dict
            Quality gate thresholds

        Returns
        -------
        Lightcurve
            New instance containing only wavelengths that pass sampling checks

        Raises
        ------
        ValueError
            If lightcurve is not 2D (multiband) or no bands pass sampling
            checks.
        """
        if self.ndim <= 1:
            raise ValueError(
                "filter_well_sampled_bands() requires 2D (multiband) data."
            )

        results = self.assess_sampling_quality_per_band(verbose=False, **kwargs)

        if results["summary"]["n_passing"] == 0:
            raise ValueError(
                "No bands passed sampling quality checks. "
                "Consider relaxing criteria or acquiring more data."
            )

        keep_wl = results["summary"]["passing_wavelengths"]
        xdata = self._xdata_raw
        keep_mask = torch.isin(
            xdata[:, 1],
            torch.tensor(keep_wl, dtype=xdata.dtype, device=xdata.device),
        )
        new_band = self.band[keep_mask.cpu().numpy()] if self.band is not None else None

        return Lightcurve(
            xdata[keep_mask].clone(),
            self._ydata_raw[keep_mask].clone(),
            self._yerr_raw[keep_mask].clone() if hasattr(self, "_yerr_raw") else None,
            band=new_band
        )

    def _get_best_sampled_band_lc(self) -> "Lightcurve":
        """Return a 1D Lightcurve for the band with the most observations.

        For 1D lightcurves, returns ``self`` unchanged.

        Returns
        -------
        Lightcurve
            1D Lightcurve (xdata is the time column only) built from the
            raw (untransformed) data of the most-sampled wavelength band.
            If multiple bands share the same (maximum) number of observations,
            the band with the smallest band value (as returned by
            ``torch.unique``) is returned.
        """
        if self.ndim <= 1:
            return self

        bands = self._xdata_raw[:, 1]
        unique_bands, band_counts = torch.unique(bands, return_counts=True)
        best_band_val = unique_bands[band_counts.argmax()]
        mask = bands == best_band_val

        t = self._xdata_raw[mask, 0]
        y = self._ydata_raw[mask]
        yerr = (
            self._yerr_raw[mask]
            if hasattr(self, "_yerr_raw") and self._yerr_raw is not None
            else None
        )
        return Lightcurve(t, y, yerr=yerr)

    def _get_variability_arrays(self):
        """Return (y, yerr) as float64 NumPy arrays, safe for CPU and GPU tensors."""
        y = self._ydata_raw.detach().cpu().numpy()
        if hasattr(self, "_yerr_raw") and self._yerr_raw is not None:
            yerr = self._yerr_raw.detach().cpu().numpy()
        else:
            yerr = np.ones_like(y)
        return y, yerr

    def check_variability(self, **kwargs) -> dict:
        """
        Check if lightcurve shows significant variability.

        Only applicable for 1-D lightcurves. For multiband data use
        :meth:`check_variability_per_band`.

        Parameters
        ----------
        **kwargs : dict
            Arguments passed to is_variable():
            - alpha: float (default 0.01)
            - fvar_min: float (default 0.05)
            - stetson_k_min: float diagnostic reference (default 0.95)
            - verbose: bool (default False)

        Returns
        -------
          Variability diagnostics from is_variable()
          If the lightcurve is multiband (ndim > 1). Use
          check_variability_per_band() instead.

        Examples
        --------
        >>> lc = Lightcurve(t, y, yerr)
        >>> diag = lc.check_variability(verbose=True)
        >>> print(f"Variable: {diag['decision']}")
        """
        if self.ndim > 1:
            raise ValueError(
                "check_variability() is for 1-D lightcurves. "
                "For multiband data use check_variability_per_band()."
            )
        from pgmuvi.preprocess.variability import is_variable

        y, yerr = self._get_variability_arrays()
        _is_var, diagnostics = is_variable(y, yerr, **kwargs)
        return diagnostics

    def check_variability_per_band(self, **kwargs) -> dict:
        """
        Check variability independently for each wavelength band.

        Only applicable for multiband (2D) lightcurves where
        ``xdata[:, 1]`` encodes the band/wavelength.

        Parameters
        ----------
        **kwargs : dict
            Arguments passed to is_variable()
                    'n_variable': int,
                    'variable_wavelengths': list[float]

        Returns
        -------
            If the lightcurve is not 2-D multiband data (ndim != 2 columns
            with time in column 0 and wavelength in column 1).

        Examples
        --------
        >>> lc2d = Lightcurve(xdata_2d, y, yerr)
        >>> results = lc2d.check_variability_per_band(verbose=True)
        >>> n_var = results['summary']['n_variable']
        >>> n_bands = results['summary']['n_bands']
        >>> print(f"{n_var}/{n_bands} bands variable")
        """
        if self._xdata_raw.dim() != 2 or self._xdata_raw.shape[1] < 2:
            raise ValueError(
                "check_variability_per_band() requires 2-D multiband data "
                "with shape (N, 2) where column 0 is time and column 1 is "
                "the band/wavelength. Got shape "
                f"{tuple(self._xdata_raw.shape)}."
            )
        from pgmuvi.preprocess.variability import is_variable

        # Convert to NumPy once; safe for both CPU and CUDA tensors
        xdata_band = self._xdata_raw[:, 1].detach().cpu().numpy()
        ydata = self._ydata_raw.detach().cpu().numpy()
        if hasattr(self, "_yerr_raw") and self._yerr_raw is not None:
            yerr_data = self._yerr_raw.detach().cpu().numpy()
        else:
            yerr_data = None

        wavelengths = np.unique(xdata_band)
        results = {}
        variable_bands = []

        for wl in wavelengths:
            mask = xdata_band == wl
            y = ydata[mask]
            yerr = yerr_data[mask] if yerr_data is not None else np.ones_like(y)

            is_var, diag = is_variable(y, yerr, **kwargs)
            results[float(wl)] = diag

            if is_var:
                variable_bands.append(float(wl))

        results["summary"] = {
            "n_bands": len(wavelengths),
            "n_variable": len(variable_bands),
            "variable_wavelengths": variable_bands,
        }
        return results



    def filter_variable_bands(self, **kwargs):
        """
        Create new Lightcurve with only variable bands retained.

        Only applicable for multiband (2D) lightcurves where
        ``xdata[:, 1]`` encodes the band/wavelength.


         Parameters
         ----------
            Arguments passed to is_variable()

        Returns
        -------
        lightcurve : Lightcurve
            New instance containing only wavelengths that pass variability tests
        None
            If no bands pass variability tests

        Examples
        --------
        >>> lc2d = Lightcurve(xdata_2d, y, yerr)
        >>> lc_var = lc2d.filter_variable_bands()
        >>> # Check how many bands were retained via the per-band summary
        >>> results = lc2d.check_variability_per_band()
        >>> print(f"Retained {results['summary']['n_variable']} variable bands")
        """
        results = self.check_variability_per_band(**kwargs)

        if results["summary"]["n_variable"] == 0:
            raise ValueError(
                "No bands passed variability tests. "
                "Consider relaxing criteria (alpha, fvar_min); "
                "stetson_k_min is diagnostic."
            )

        keep_wl = results["summary"]["variable_wavelengths"]
        wl_array = self._xdata_raw[:, 1].detach().cpu().numpy()
        keep_mask = np.isin(wl_array, keep_wl)
        keep_tensor = torch.as_tensor(
            keep_mask,
            dtype=torch.bool,
            device=self._xdata_raw.device,
        )

        new_x = self._xdata_raw[keep_tensor].clone()
        new_y = self._ydata_raw[keep_tensor].clone()

        if hasattr(self, "_yerr_raw") and self._yerr_raw is not None:
            new_yerr = self._yerr_raw[keep_tensor].clone()
        else:
            new_yerr = None

        return Lightcurve(new_x, new_y, yerr=new_yerr)

    def auto_select_model(self, verbose=True):
        """Automatically select the best model type based on data characteristics.

        Analyses the data to recommend an appropriate GP model. For 1D data,
        the Lomb-Scargle periodogram is computed and the peak power used to
        decide between a quasi-periodic, periodic+stochastic, or Matérn model.
        For 2D multiwavelength data, per-band periodograms are compared to
        determine whether the variability is achromatic or wavelength-dependent.

        Parameters
        ----------
        verbose : bool, optional
            If True, print a summary of the recommendation, by default True.

        Returns
        -------
        model_str : str
            Recommended model string suitable for passing directly to
            ``fit()`` or ``set_model()``.
        diagnostics : dict
            Dictionary containing:
            - ``'model'`` — same as ``model_str``.
            - ``'reason'`` — human-readable explanation.
            - Additional data-dependent keys (e.g. ``'max_ls_power'``).

        Examples
        --------
        >>> from pgmuvi.lightcurve import Lightcurve
        >>> import torch
        >>> import numpy as np
        >>> t = torch.linspace(0, 20, 100)
        >>> y = torch.sin(2 * np.pi * t / 5)
        >>> lc = Lightcurve(t, y)
        >>> model_str, diag = lc.auto_select_model()
        """
        from .initialization import initialize_separable_from_data

        diagnostics = {}

        if self.ndim == 1:
            # 1D data — use Lomb-Scargle to assess periodicity strength
            _freq, power = self.fit_LS(freq_only=True)
            max_power = float(power.max()) if len(power) > 0 else 0.0
            diagnostics["max_ls_power"] = max_power

            if max_power > 0.5:
                model_str = "1DQuasiPeriodic"
                diagnostics["reason"] = (
                    f"Strong periodic signal detected (LS power={max_power:.2f}); "
                    "quasi-periodic kernel recommended."
                )
            elif max_power > 0.2:
                model_str = "1DPeriodicStochastic"
                diagnostics["reason"] = (
                    f"Moderate periodicity with stochastic component "
                    f"(LS power={max_power:.2f}); "
                    "periodic+stochastic kernel recommended."
                )
            else:
                model_str = "1DMatern"
                diagnostics["reason"] = (
                    f"No strong periodicity detected (LS power={max_power:.2f}); "
                    "Matérn kernel recommended for stochastic variability."
                )
        else:
            # 2D data — check whether periods are consistent across wavelengths
            init_params = initialize_separable_from_data(
                self._xdata_raw,
                self._ydata_raw,
            )
            diagnostics["init_params"] = init_params

            if init_params.get("is_achromatic", True):
                model_str = "2DAchromatic"
                diagnostics["reason"] = (
                    "Periods consistent across wavelengths (achromatic variability); "
                    "achromatic separable kernel recommended."
                )
            else:
                model_str = "2DWavelengthDependent"
                diagnostics["reason"] = (
                    "Periods vary with wavelength (chromatic variability); "
                    "wavelength-dependent separable kernel recommended."
                )

        diagnostics["model"] = model_str

        if verbose:
            sep = "=" * 70
            print(sep)
            print("AUTO MODEL SELECTION")
            print(sep)
            print(f"Recommended model: {model_str}")
            print(f"Reason: {diagnostics['reason']}")
            print(sep)

        return model_str, diagnostics

    def _build_parameter_estimation_context(self):
        """Construct a parameter-estimation context from this light curve."""
        flux_values = self._ydata_raw
        if isinstance(flux_values, torch.Tensor):
            flux_values = flux_values.detach().cpu().numpy()

        flux_values = np.asarray(flux_values, dtype=float)
        flux_values = flux_values[np.isfinite(flux_values)]

        time_values = self._xdata_raw
        if isinstance(time_values, torch.Tensor):
            time_values = time_values.detach().cpu().numpy()

        time_values = np.asarray(time_values, dtype=float)

        if self.ndim > 1:
            times = np.sort(time_values[:, 0])
        else:
            times = np.sort(time_values)

        baseline_duration = None
        median_cadence = None

        if times.size >= 2:
            baseline_duration = float(times.max() - times.min())

            gaps = np.diff(times)
            gaps = gaps[gaps > 0]

            if gaps.size:
                median_cadence = float(np.median(gaps))

        if flux_values.size == 0:
            return ParameterEstimationContext(
                is_multiband=self.ndim > 1,
                global_diagnostics=LightcurveDiagnostics(
                    baseline_duration=baseline_duration,
                    median_cadence=median_cadence,
                ),
            )

        p025, p50, p975 = np.percentile(flux_values, [2.5, 50.0, 97.5])
        return ParameterEstimationContext(
            is_multiband=self.ndim > 1,
            global_diagnostics=LightcurveDiagnostics(
                median_flux=float(p50),
                flux_percentiles={
                    2.5: float(p025),
                    50.0: float(p50),
                    97.5: float(p975),
                },
                n_points=int(flux_values.size),
                baseline_duration=baseline_duration,
                median_cadence=median_cadence,
            ),
        )

    def _apply_parameter_workflow_estimates(self):
        """Apply parameter workflow estimates when the model supports them."""
        self.parameter_workflow_result = None

        if not model_supports_parameter_workflow(self.model):
            return None

        context = self._build_parameter_estimation_context()

        self.parameter_workflow_result = build_and_apply_parameter_estimates(
            model=self.model,
            context=context,
        )

        return self.parameter_workflow_result

    def get_parameter_workflow_summary(self):
        """Return a lightweight summary of parameter workflow results."""
        result = self.parameter_workflow_result

        if result is None:
            return {
                "available": False,
                "applied": 0,
                "skipped": 0,
                "applied_parameters": [],
                "skipped_parameters": [],
                "skipped_reasons": {},
            }

        applied_parameters = []
        skipped_parameters = []
        skipped_reasons = {}

        for name, value in result.items():
            if isinstance(value, dict):
                value_applied = bool(value.get("value"))
                constraint_applied = bool(value.get("constraint"))
                applied = value_applied or constraint_applied
            else:
                applied = bool(value)

            if applied:
                applied_parameters.append(name)
            else:
                skipped_parameters.append(name)

            if isinstance(value, dict):
                value_reason = value.get("value_reason")
                constraint_reason = value.get("constraint_reason")

                if value_reason is not None or constraint_reason is not None:
                    skipped_reasons[name] = {
                        "value_reason": value_reason,
                        "constraint_reason": constraint_reason,
                    }

        return {
            "available": True,
            "applied": len(applied_parameters),
            "skipped": len(skipped_parameters),
            "applied_parameters": applied_parameters,
            "skipped_parameters": skipped_parameters,
            "skipped_reasons": skipped_reasons,
        }

    def get_parameter_workflow_report(self):
        """Return a structured parameter workflow report."""
        result = self.parameter_workflow_result

        if result is None:
            return {
                "available": False,
                "applied": [],
                "skipped": [],
            }

        applied = []
        skipped = []

        for parameter, info in result.items():
            if not isinstance(info, dict):
                if info:
                    applied.append(
                        {
                            "parameter": parameter,
                        }
                    )
                else:
                    skipped.append(
                        {
                            "parameter": parameter,
                        }
                    )
                continue

            value_applied = bool(info.get("value"))
            constraint_applied = bool(info.get("constraint"))

            entry = {
                "parameter": parameter,
                "value_applied": value_applied,
                "constraint_applied": constraint_applied,
                "value_reason": info.get("value_reason"),
                "constraint_reason": info.get("constraint_reason"),
            }

            if value_applied or constraint_applied:
                applied.append(entry)
            else:
                skipped.append(entry)

        return {
            "available": True,
            "applied": applied,
            "skipped": skipped,
        }

    def fit(self, *args, **kwargs):
        """Fit wrapper that records lightweight in-memory fit history."""
        # Nested fit() calls (e.g. from _consensus_standard_fit) delegate
        # to _fit_core directly so that only the outermost call records a
        # single canonical history entry.
        _nesting = getattr(self, "_fit_nesting_depth", 0)
        if _nesting > 0:
            self._fit_nesting_depth = _nesting + 1
            try:
                return self._fit_core(*args, **kwargs)
            finally:
                self._fit_nesting_depth -= 1

        self._fit_nesting_depth = 1
        self.parameter_workflow_result = None
        _fit_start = time.perf_counter()
        _model_arg = kwargs.get("model")
        _fit_strategy = kwargs.get("fit_strategy")
        _training_iter = kwargs.get("training_iter")
        _num_mixtures = kwargs.get("num_mixtures")
        _backend = "cuda" if bool(kwargs.get("cuda", False)) else "cpu"
        _constraint_set = kwargs.get("constraint_set")
        _constrain_consensus = kwargs.get("constrain_consensus")
        if _constraint_set is not None or _constrain_consensus is True:
            _constrained = True
        elif _constrain_consensus is False:
            _constrained = False
        else:
            _constrained = None

        _model_class = None
        if _model_arg is not None:
            _model_class = (
                _model_arg
                if isinstance(_model_arg, str)
                else _model_arg.__class__.__name__
            )

        # Capture the unique bands present at call time for provenance.
        try:
            _bands = (
                sorted({str(b) for b in np.asarray(self.band, dtype=np.str_)})
                if self.band is not None
                else None
            )
        except Exception:
            _bands = None

        # Infer parameterisation space from the model argument.  This is a
        # best-effort guess; the resolved class is used in the success branch.
        _model_class_lower = str(_model_class or "").lower()
        _fit_strategy_lower = str(_fit_strategy or "").lower()
        if (
            "spectralmixture" in _model_class_lower
            or "separable" in _model_class_lower
            or _fit_strategy_lower == "consensus"
            or _model_arg in ("2D", "1D", "2d", "1d")
        ):
            _uses_frequency_space: bool | None = True
            _uses_period_space: bool | None = False
        elif (
            "periodic" in _model_class_lower
            or "quasiperiodic" in _model_class_lower
        ):
            _uses_frequency_space = False
            _uses_period_space = True
        else:
            _uses_frequency_space = None
            _uses_period_space = None

        _fit_configuration = self._collect_fit_configuration_snapshot(
            fit_kwargs=kwargs,
            context={
                "model_class": _model_class,
                "fit_strategy": _fit_strategy,
                "training_iter": _training_iter,
                "num_mixtures": _num_mixtures,
                "backend": _backend,
                "constraint_set": (
                    str(_constraint_set) if _constraint_set is not None else None
                ),
            },
        )

        self._fit_history_context = {
            "model_class": _model_class,
            "fit_strategy": _fit_strategy,
            "training_iter": _training_iter,
            "num_mixtures": _num_mixtures,
            "backend": _backend,
            "constrained": _constrained,
            "constrained_fit": _constrained,
            "constraint_set": (
                str(_constraint_set) if _constraint_set is not None else None
            ),
            "bands": _bands,
            "uses_frequency_space": _uses_frequency_space,
            "uses_period_space": _uses_period_space,
            "fit_configuration": _fit_configuration,
            "environment": self._fit_history_environment_metadata(),
        }
        self._fit_history_recorded = False

        try:
            result = self._fit_core(*args, **kwargs)
        except Exception as exc:
            if not bool(getattr(self, "_fit_history_recorded", False)):
                self._append_fit_history(
                    success=False,
                    failed=True,
                    exception_type=exc.__class__.__name__,
                    exception_message=str(exc),
                    elapsed_seconds=time.perf_counter() - _fit_start,
                    notes={"source": "fit_exception"},
                )
            raise
        else:
            _model_obj = getattr(self, "model", None)
            _resolved_model_class = (
                _model_obj.__class__.__name__
                if _model_obj is not None
                else _model_class
            )
            _resolved_backend = (
                "cuda"
                if bool(getattr(self, "_cuda", False))
                else _backend
            )
            # Refine parameterisation inference from the resolved class name.
            _resolved_class_lower = str(_resolved_model_class or "").lower()
            if (
                "spectralmixture" in _resolved_class_lower
                or "separable" in _resolved_class_lower
            ):
                _resolved_freq = True
                _resolved_period = False
            elif (
                "periodic" in _resolved_class_lower
                or "quasiperiodic" in _resolved_class_lower
            ):
                _resolved_freq = False
                _resolved_period = True
            else:
                _resolved_freq = _uses_frequency_space
                _resolved_period = _uses_period_space
            self._append_fit_history(
                model_class=_resolved_model_class,
                fit_strategy=_fit_strategy,
                success=True,
                failed=False,
                training_iter=_training_iter,
                num_mixtures=_num_mixtures,
                elapsed_seconds=time.perf_counter() - _fit_start,
                backend=_resolved_backend,
                constrained=_constrained,
                constrained_fit=_constrained,
                constraint_set=(
                    str(_constraint_set)
                    if _constraint_set is not None
                    else None
                ),
                bands=_bands,
                uses_frequency_space=_resolved_freq,
                uses_period_space=_resolved_period,
                notes={"source": "fit_success"},
            )
            self._fit_history_recorded = True
            return result
        finally:
            self._fit_nesting_depth = 0
            self._fit_history_context = {}

    def _fit_core(
        self,
        model=None,
        likelihood=None,
        num_mixtures=None,
        guess=None,
        periods=None,
        use_mls_init=True,
        use_best_band_init: bool = False,
        use_parameter_workflow: bool = True,
        constraint_set=None,
        grid_size=2000,
        cuda=False,
        training_iter=300,
        max_cg_iterations=None,
        optim="AdamW",
        miniter=None,
        stop=1e-5,
        lr=0.1,
        stopavg=30,
        variance=False,
        fit_strategy=None,
        **kwargs,
    ):
        """Fit the lightcurve

        Parameters
        ----------
        model : string or instance of gpytorch.models.GP, optional
            The model to use for the GP, by default None. If None, an
            error will be raised. If a string, it must be one of the
            following:

            Spectral mixture models:
                '1D': SpectralMixtureGPModel
                '2D': TwoDSpectralMixtureGPModel
                '1DLinear': SpectralMixtureLinearMeanGPModel
                '2DLinear': TwoDSpectralMixtureLinearMeanGPModel
                '1DSKI': SpectralMixtureKISSGPModel
                '2DSKI': TwoDSpectralMixtureKISSGPModel
                '1DLinearSKI': SpectralMixtureLinearMeanKISSGPModel
                '2DLinearSKI': TwoDSpectralMixtureLinearMeanKISSGPModel
                '2DPowerLaw': TwoDSpectralMixturePowerLawMeanGPModel
                '2DPowerLawSKI': TwoDSpectralMixturePowerLawMeanKISSGPModel
                '2DDust': TwoDSpectralMixtureDustMeanGPModel
                '2DDustSKI': TwoDSpectralMixtureDustMeanKISSGPModel

            Alternative 1D models:
                '1DQuasiPeriodic': QuasiPeriodicGPModel
                '1DMatern': MaternGPModel
                '1DPeriodicStochastic': PeriodicPlusStochasticGPModel
                '1DLinearQuasiPeriodic': LinearMeanQuasiPeriodicGPModel

            Separable 2D models:
                '2DSeparable': SeparableGPModel
                '2DAchromatic': AchromaticGPModel
                '2DWavelengthDependent': WavelengthDependentGPModel
                '2DDustMean': DustMeanGPModel
                '2DPowerLawMean': PowerLawMeanGPModel


            If an instance of a GP class, that object will be used.
        likelihood : string, None or instance of
                        gpytorch.likelihoods.likelihood.Likelihood or Constraint,
                        optional
            If likelihood is passed, it will be passed along to `set_likelihood()`
            and used to set the likelihood function for the model. For details, see
            the documentation for `set_likelihood()`.
        num_mixtures : int or None, optional
            The number of mixtures to use in the spectral mixture kernel.  By
            default ``None``, which lets the MLS initialisation (see
            ``use_mls_init``) choose the value automatically.  When
            ``use_mls_init=True`` and ``periods`` is ``None``, setting
            ``num_mixtures`` to an integer *N* overrides the automatic count:
            the first *N* significant MLS periods are used; if *N* exceeds the
            number of significant periods, non-significant peaks are added to
            make up the difference.  When ``use_mls_init=False`` and
            ``num_mixtures`` is ``None`` a fallback of 4 is used.
        guess : dict, optional
            A dictionary of the hyperparameters to use for the model and
            likelihood. The keys should be the names of the parameters, and the
            values should be Tensors containing the values of the parameters.
            If None, no hyperparameters will be set. If a hyperparameter is
            passed for a parameter that is not a model or likelihood
            parameter, it will be ignored.
        periods : array-like or None, optional
            Initial guesses for the periods (in the same units as the
            lightcurve time axis).  When provided for 1D spectral-mixture
            kernels (i.e. when ``ard_num_dims == 1``), the MLS
            initialisation is skipped entirely: ``num_mixtures`` is set to
            the number of supplied periods and the spectral-mixture kernel
            frequencies are initialised from these values.  If both
            ``periods`` and ``guess`` are supplied, entries in ``guess`` take
            priority over the period-derived frequencies.  For multi-
            dimensional spectral-mixture models (e.g. 2D kernels), the
            current implementation does not use ``periods`` to seed mixture
            means; in those cases, only explicit initial values provided via
            ``guess`` (or the model's own defaults) will be used.
        use_mls_init : bool, optional
            If ``True`` (default) and ``periods`` is ``None`` and a 1D
            spectral-mixture model string is given (``ard_num_dims == 1``),
            the Multiband Lomb-Scargle (MLS) periodogram is run first to
            estimate the number of significant periods and their frequencies,
            which are used as initial guesses for the spectral-mixture kernel
            frequencies.  Set to ``False`` to disable this behaviour and, for
            models that call it, fall back to GPyTorch's
            ``initialize_from_data``.  Note that several 2D spectral-mixture
            models do not currently call ``initialize_from_data`` at all, so
            for those models MLS-based or period-based frequency seeding is
            not applied and the underlying GPyTorch defaults are used
            instead.
        use_best_band_init : bool, optional
            If ``True`` and the lightcurve is multiband (``ndim > 1``) and
            ``use_mls_init=True`` and ``periods`` is ``None``, a 1D
            Lomb-Scargle fit on the most-sampled band is used to seed the
            spectral-mixture frequency initialisation instead of the
            standard multiband LS.  For 2D spectral-mixture models
            (``ard_num_dims == 2``, non-SKI), the fitted temporal
            frequencies are also used to initialise the temporal dimension
            of the kernel mixture means, with the minimum wavelength
            frequency (1/wavelength_span) as the default for the
            wavelength dimension, corresponding to approximately achromatic
            variability.  This can improve convergence for sources with a
            large dynamic range in the number of observations across bands.
            Has no effect for 1D lightcurves or when ``use_mls_init=False``.
        use_parameter_workflow : bool, optional
            Whether to apply schema-driven parameter initialization before
            training. When enabled, models that expose a parameter_schema()
            may automatically receive parameter estimates and constraints
            derived from available light-curve diagnostics. Defaults to True.

            Set to False to disable automatic parameter-workflow application
            and rely only on existing defaults, explicit user guesses,
            consensus/MLS initialization, and manually supplied constraints.

            After fitting, use get_parameter_workflow_summary() or
            get_parameter_workflow_report() to inspect what was applied or skipped.
        constraint_set : str or None, optional
            Name of a pre-defined source-type constraint set to apply via
            :meth:`set_default_constraints`.  When provided, the period bounds
            defined in the constraint set are also used to filter MLS peaks
            *before* the model is constructed: peaks whose frequencies fall
            outside the constraint-set allowed range are excluded from the
            initialisation (with a ``RuntimeWarning``).  Currently supported
            values are ``"LPV"`` (Long-Period Variables, minimum period 100 in
            the native time units).  Pass ``None`` (the default) to use only
            the data-driven bounds.
        grid_size : int, optional
            The number of points to use in the grid for the KISS-GP models,
            by default 2000.
        cuda : bool, optional
            Whether to use CUDA, by default False.
        training_iter : int, optional
            The number of iterations to use for training, by default 300.
        max_cg_iterations : int, optional
            The maximum number of conjugate gradient iterations to use, by
            default None. If None, gpytorch.settings.max_cg_iterations will
            be used.
        optim : str or torch.optim.Optimizer, optional
            The optimizer to use for training, by default "AdamW". If a string,
            it must be one of the following:
                'AdamW': torch.optim.AdamW
                'Adam': torch.optim.Adam
                'SGD': torch.optim.SGD
            Otherwise, it must be an instance of torch.optim.Optimizer.
        miniter : int, optional
            The minimum number of iterations to use for training, by default
            None. If None, training_iter will be used.
        stop : float, optional
            The stopping criterion for the training, by default 1e-5.
        lr : float, optional
            The learning rate to use for the optimizer, by default 0.1.
        stopavg : int, optional
            The number of iterations to use for the stopping criterion, by
            default 30.
        variance : bool, optional
            If False (default), stored uncertainties are treated as errors
            (standard deviations) and are squared before being used as noise
            variances in the likelihood.  Set to True if the stored
            uncertainties already represent variances.
        fit_strategy : {"consensus", "consensus_multicomp",
                        "consensus_relaxed"} or None, optional
            Optional fitting-strategy selector.  The default ``None`` keeps the
            existing general ``fit`` workflow unchanged.  When set to one of
            the listed strategy names, ``fit`` dispatches to the corresponding
            internal consensus-fit pathway.
            ``"consensus"`` runs a deterministic multi-band consensus workflow
            (2D light curves only) that:
            (i) computes per-band sampling diagnostics,
            (ii) extracts one dominant LS frequency per acceptable band,
            (iii) aggregates frequencies with median/MAD outlier rejection,
            and (iv) uses the resulting consensus to seed and optionally
            constrain the spectral-mixture fit.
            ``"consensus_multicomp"`` runs a staged multi-component workflow
            that: (i) extracts per-band multi-component frequency candidates,
            (ii) clusters components across bands in frequency space,
            (iii) aggregates accepted clusters into a multi-component
            consensus, and (iv) initializes the final 2D spectral-mixture fit.
            It currently reuses the existing global constraint system, so all
            accepted components share one broad frequency interval; component-
            specific mixture constraints remain future work.
            ``"consensus_relaxed"`` remains a placeholder and still raises
            ``NotImplementedError``.
            Additional ``"consensus"`` controls accepted via ``**kwargs``:
            ``min_points_per_band``, ``max_gap_fraction``,
            ``min_duty_cycle``, ``outlier_sigma``, ``use_acf``,
            ``constrain_consensus``, and ``consensus_width_factor``.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the model constructor,
            likelihood constructor, or the optimizer.

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            If no model is provided.

        Notes
        -----
        Data validation, quality checks, and subsampling are performed at
        object construction time (see :meth:`__init__`).  Use the
        ``check_sampling``, ``check_variability``, and ``max_samples``
        parameters of :meth:`__init__` to control pre-processing.
        """
        # Capture the caller's original num_mixtures argument before any
        # mutation (MLS init / fallback default).  Used later to decide
        # whether to substitute the stored _model_num_mixtures.
        _num_mixtures_arg = num_mixtures
        self._reset_fit_state(
            clear_failure=True,
            clear_model_state=False,
            clear_consensus=bool(fit_strategy is not None),
        )
        _constraints_were_set_before_fit = bool(self.__CONTRAINTS_SET)

        verbose = kwargs.get("verbose", False)

        # Dispatch alternative fit strategies before any stateful setup from
        # the default/general fit pathway mutates this Lightcurve instance.
        if fit_strategy is not None:
            return self._consensus_fit(
                fit_strategy=fit_strategy,
                model=model,
                likelihood=likelihood,
                num_mixtures=num_mixtures,
                guess=guess,
                periods=periods,
                use_mls_init=use_mls_init,
                use_best_band_init=use_best_band_init,
                # use_parameter_workflow=use_parameter_workflow,
                constraint_set=constraint_set,
                grid_size=grid_size,
                cuda=cuda,
                training_iter=training_iter,
                max_cg_iterations=max_cg_iterations,
                optim=optim,
                miniter=miniter,
                stop=stop,
                lr=lr,
                stopavg=stopavg,
                variance=variance,
                **kwargs,
            )

        if not hasattr(self, "likelihood"):
            self.set_likelihood(likelihood, variance=variance, **kwargs)
        elif not self.__SET_LIKELIHOOD_CALLED and likelihood is None:
            # if no likelihood is passed, we only want to set the likelihood
            # if it hasn't already been set
            self.set_likelihood(likelihood, variance=variance, **kwargs)
        elif likelihood is not None:
            self.set_likelihood(likelihood, variance=variance, **kwargs)
        # if likelihood is None and not hasattr(self, 'likelihood'):
        #     raise ValueError("""You must provide a likelihood function""")
        # elif likelihood is not None:
        #     self.set_likelihood(likelihood, **kwargs)

        # Validate explicitly-provided num_mixtures early.
        if num_mixtures is not None:
            # Must be a (non-bool) integer and strictly positive.
            if isinstance(num_mixtures, bool) or not isinstance(
                num_mixtures, int
            ):
                raise TypeError(
                    "`num_mixtures` must be a positive integer or None, "
                    f"got {num_mixtures!r} of type {type(num_mixtures)!r}."
                )
            if num_mixtures < 1:
                raise ValueError(
                    "`num_mixtures` must be a positive integer or None, "
                    f"got {num_mixtures}."
                )

        # --- MLS-based initialisation ---
        _init_freqs = None  # frequencies (raw units) to seed the SM kernel

        # Minimum frequency in raw data units: the period cannot exceed the
        # total span of the data.  Used to filter obviously unphysical MLS
        # peaks and to generate padding frequencies when not enough peaks are
        # available.
        _t_raw = (
            self._xdata_raw[:, 0] if self.ndim > 1 else self._xdata_raw
        )
        _t_span = float(_t_raw.max() - _t_raw.min())
        _freq_lower = 1.0 / _t_span if _t_span > 0 else 0.0
        _t_sorted = _t_raw.sort().values
        _t_diffs = _t_sorted[1:] - _t_sorted[:-1]
        _pos_diffs = _t_diffs[_t_diffs > 0]
        _freq_upper = (
            1.0 / (2.0 * float(_pos_diffs.min()))
            if len(_pos_diffs) > 0
            else float("inf")
        )

        if periods is not None:
            # User supplied explicit period guesses — skip MLS entirely.
            _periods_tensor = torch.as_tensor(
                periods, dtype=self._xdata_raw.dtype
            ).flatten()

            # Validate user-supplied periods: must be non-empty, finite, and > 0
            if _periods_tensor.numel() == 0:
                raise ValueError(
                    "When providing explicit `periods`, the sequence must be "
                    "non-empty."
                )
            if not torch.isfinite(_periods_tensor).all():
                raise ValueError(
                    "All values in `periods` must be finite (no NaN or inf)."
                )
            if not (_periods_tensor > 0).all():
                raise ValueError(
                    "All values in `periods` must be strictly positive."
                )
            _init_freqs = 1.0 / _periods_tensor
            num_mixtures = len(_init_freqs)
        elif use_mls_init and isinstance(model, str) and model in _SM_MODELS:
            # Compute constraint-set frequency bounds in raw data units.
            # These are used in addition to the data-span bounds to exclude
            # MLS peaks that would lie outside user-requested period limits.
            # Note: fit_LS uses Nyquist_factor > 1, so its frequencies can
            # exceed the standard Nyquist.  We therefore only apply an upper
            # frequency limit when the constraint_set explicitly demands one
            # (via a minimum-period specification); otherwise the upper bound
            # is left unrestricted (inf).
            _cs_freq_lower = _freq_lower  # default: data-span lower bound
            _cs_freq_upper = float("inf")  # no upper cap unless constraint_set
            if constraint_set is not None:
                try:
                    cs = get_constraint_set(constraint_set)
                    if "period" in cs:
                        _pb = cs["period"]
                        _p_lower_val, _p_lower_active = _pb["lower"]
                        _p_upper_val, _p_upper_active = _pb["upper"]
                        # Period lower limit → max allowed frequency
                        if _p_lower_active and _p_lower_val is not None:
                            _cs_freq_upper = min(
                                _cs_freq_upper, 1.0 / _p_lower_val
                            )
                        # Period upper limit → min allowed frequency
                        if _p_upper_active and _p_upper_val is not None:
                            _cs_freq_lower = max(
                                _cs_freq_lower, 1.0 / _p_upper_val
                            )
                except (ValueError, KeyError):
                    warnings.warn(
                        f"constraint_set={constraint_set!r} is not recognised "
                        "and will be ignored for MLS peak filtering. "
                        "Only the data-span frequency bounds will be applied.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    # Normalise invalid constraint_set so that later code does not
                    # attempt to apply or validate an unknown set again.
                    constraint_set = None

            # Run the MLS periodogram to choose num_mixtures and seed frequencies.
            try:
                _max_peaks = max(num_mixtures or 1, 10)
                if use_best_band_init and self.ndim > 1:
                    # Use a 1D LS on the most-sampled band to get reliable
                    # temporal frequency estimates instead of the multiband LS.
                    # This is beneficial when sampling is highly heterogeneous
                    # across bands: the best-sampled band provides the most
                    # accurate period constraints.
                    _best_band_lc = self._get_best_sampled_band_lc()
                    ls_freqs, ls_sig = _best_band_lc.fit_LS(
                        num_peaks=_max_peaks
                    )
                    # Compute the best-band's own Nyquist as the upper
                    # frequency bound.  The best-band 1D LS may find alias
                    # peaks above this Nyquist (when Nyquist_factor > 1);
                    # cap at the Nyquist to avoid out-of-range initialisation.
                    _bb_t = _best_band_lc._xdata_raw.sort().values
                    _bb_diffs = _bb_t[1:] - _bb_t[:-1]
                    _bb_pos = _bb_diffs[_bb_diffs > 0]
                    _bb_nyquist = (
                        float(1.0 / (2.0 * _bb_pos.min()))
                        if len(_bb_pos) > 0
                        else float("inf")
                    )
                else:
                    ls_freqs, ls_sig = self.fit_LS(num_peaks=_max_peaks)
                    _bb_nyquist = float("inf")

                # Filter peaks whose period exceeds the data span or falls
                # outside user-specified constraint-set period bounds.
                # When use_best_band_init=True also cap at the best-band
                # Nyquist to remove alias peaks that exceed the true sampling
                # limit of the best-sampled band.
                _eff_upper = min(_cs_freq_upper, _bb_nyquist)
                # Ensure that any subsequent use of the frequency upper bound
                # (e.g. for padding when num_mixtures exceeds the number of
                # LS peaks) also respects the best-band Nyquist cap.
                if use_best_band_init and self.ndim > 1:
                    _cs_freq_upper = _eff_upper
                    _freq_upper = _eff_upper
                if len(ls_freqs) > 0 and _cs_freq_lower > 0:
                    _valid = (ls_freqs >= _cs_freq_lower) & (
                        ls_freqs <= _eff_upper
                    )
                    if not _valid.all():
                        _n_filtered = int((~_valid).sum().item())
                        warnings.warn(
                            f"{_n_filtered} MLS peak(s) fell outside the "
                            f"allowed frequency range "
                            f"[{_cs_freq_lower:.4g}, {_eff_upper:.4g}] "
                            "(derived from data span"
                            + (
                                f" and constraint_set={constraint_set!r}"
                                if constraint_set is not None
                                else ""
                            )
                            + ") and were excluded from the initialisation.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        ls_freqs = ls_freqs[_valid]
                        ls_sig = ls_sig[_valid]

                if len(ls_freqs) > 0:
                    ls_sig_freqs = ls_freqs[ls_sig]
                    ls_insig_freqs = ls_freqs[~ls_sig]

                    if num_mixtures is None:
                        # Default: use only the statistically significant peaks.
                        if len(ls_sig_freqs) > 0:
                            num_mixtures = len(ls_sig_freqs)
                            _init_freqs = ls_sig_freqs
                        else:
                            # No significant peaks; fall back to the strongest one.
                            num_mixtures = 1
                            _init_freqs = ls_freqs[:1]
                    else:
                        # User specified num_mixtures: fill with significant peaks
                        # first, then non-significant ones, then pad with
                        # evenly-spaced frequencies if still not enough.
                        n_sig = len(ls_sig_freqs)
                        if num_mixtures <= n_sig:
                            _init_freqs = ls_sig_freqs[:num_mixtures]
                        else:
                            _extra = num_mixtures - n_sig
                            _available_insig = ls_insig_freqs[:_extra]
                            _init_freqs = torch.cat(
                                [ls_sig_freqs, _available_insig]
                            )
                            # Pad with additional frequencies if still short.
                            _n_pad = num_mixtures - len(_init_freqs)
                            if _n_pad > 0:
                                # Determine padding interval as the intersection of
                                # the data-based frequency range and any
                                # constraint-set bounds.
                                _pad_lower = _freq_lower
                                _pad_upper = _freq_upper
                                if _cs_freq_lower > 0:
                                    _pad_lower = max(_pad_lower, _cs_freq_lower)
                                    _pad_upper = min(_pad_upper, _cs_freq_upper)
                                if _pad_upper > _pad_lower:
                                    _msg = (
                                        f"Only {len(_init_freqs)} MLS peak(s)"
                                        f" found but {num_mixtures} were"
                                        f" requested. Padding with {_n_pad}"
                                        " evenly-spaced frequencies in"
                                        f" [{_pad_lower:.4g},"
                                        f" {_pad_upper:.4g}]."
                                    )
                                    warnings.warn(
                                        _msg,
                                        RuntimeWarning,
                                        stacklevel=2,
                                    )
                                    _pad = torch.linspace(
                                        _pad_lower,
                                        _pad_upper,
                                        _n_pad + 2,
                                        dtype=_init_freqs.dtype,
                                    )[1:-1]
                                else:
                                    _msg = (
                                        "Could not construct a valid"
                                        " frequency range for padding MLS"
                                        " initialisation; repeating the last"
                                        " available MLS frequency to reach"
                                        f" num_mixtures={num_mixtures}."
                                    )
                                    warnings.warn(
                                        _msg,
                                        RuntimeWarning,
                                        stacklevel=2,
                                    )
                                    _last_freq = _init_freqs[-1]
                                    _pad = _init_freqs.new_full(
                                        (_n_pad,), _last_freq
                                    )
                                _init_freqs = torch.cat([_init_freqs, _pad])
                else:
                    # MLS found no peaks at all; warn and fall back.

                    if num_mixtures is None:
                        num_mixtures = 4
                    # This Warning has to be raised after the if, so that the
                    # user-defined number of mixtures is used and they still see
                    # the warning if they set a value.
                    warnings.warn(
                        "MLS periodogram returned no peaks; falling back to "
                        f"num_mixtures={num_mixtures} with default initialisation.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            except Exception as exc:
                # MLS failed for any reason; fall back gracefully but warn the
                # user.  Ensure num_mixtures is set before issuing the warning.
                if num_mixtures is None:
                    inferred_count = None
                    if hasattr(self, "model") and self.model is not None:
                        inferred_count = self._infer_num_mixtures_from_model()
                    num_mixtures = (
                        inferred_count if inferred_count is not None else 4
                    )
                # Store the authoritative mixture counts now that we know the
                # fallback value.
                self._fit_num_mixtures_requested = _num_mixtures_arg
                self._fit_num_mixtures_effective = num_mixtures
                warnings.warn(
                    "MLS-based initialisation failed; falling back to "
                    f"num_mixtures={num_mixtures}. Original error was: "
                    f"{exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Final fallback when MLS init is disabled or not applicable.
        if num_mixtures is None:
            num_mixtures = 4

        if model is None and not hasattr(self, "model"):
            raise ValueError("""You must provide a model""")
        elif model is None and self.model is None:
            # The model was discarded (e.g. after band filtering). Re-create
            # it with the updated training data.
            _stored_instance = getattr(self, "_model_instance", None)
            _stored_str = getattr(self, "_model_str", None)
            # Preserve the originally configured num_mixtures when the
            # caller did not explicitly provide a value (i.e. passed None).
            # If the caller explicitly supplied num_mixtures, honour that
            # value even if it happens to equal the stored one.
            if _num_mixtures_arg is None and hasattr(self, "_model_num_mixtures"):
                stored_nm = self._model_num_mixtures
                _effective_num_mixtures = (
                    stored_nm if stored_nm is not None else num_mixtures
                )
            else:
                _effective_num_mixtures = num_mixtures
            if _stored_instance is not None:
                # User originally provided a GP instance. Re-bind it to the
                # new (filtered) training data via set_train_data() if the
                # model supports it (ExactGP), otherwise recreate via
                # set_model() which will use the same underlying class.
                self.set_likelihood(likelihood, variance=variance, **kwargs)
                if hasattr(_stored_instance, "set_train_data"):
                    _stored_instance.set_train_data(
                        inputs=self._xdata_transformed,
                        targets=self._ydata_transformed,
                        strict=False,
                    )
                    self.model = _stored_instance
                    self._make_parameter_dict()
                else:
                    # Approximate GP (e.g. SparseSpectralMixtureGPModel):
                    # cannot cheaply rebind, so fall back to raising an
                    # informative error.
                    raise ValueError(
                        "The model instance does not support set_train_data(). "
                        "Please pass model= explicitly to fit() after band "
                        "filtering, or use a string model identifier."
                    )
            elif _stored_str is not None:
                self.set_model(
                    _stored_str,
                    self.likelihood,
                    num_mixtures=_effective_num_mixtures,
                    variance=variance,
                    **kwargs,
                )
            else:
                raise ValueError("""You must provide a model""")
        elif model is not None:
            self.set_model(
                model,
                self.likelihood,
                num_mixtures=num_mixtures,
                variance=variance,
                **kwargs,
            )

        # Validate 2D setup if we have 2D data
        if self.ndim > 1:
            self._validate_2d_setup()
        if not self.__CONTRAINTS_SET:
            self.set_default_constraints(constraint_set=constraint_set)

        if not self.__CONTRAINTS_SET:
            self.set_default_constraints()

        if use_parameter_workflow:
            self._apply_parameter_workflow_estimates()

        if cuda:
            self.cuda()
        # Build the combined hyperparameter initialisation dict.
        # MLS-derived (or user-supplied) period frequencies act as the base;
        # any explicit `guess` entries take priority on top.
        _hypers_to_set = {}
        if (
            _init_freqs is not None
            and hasattr(self, "model")
            and hasattr(self.model, "covar_module")
            and hasattr(self.model.covar_module, "mixture_means")
            and getattr(self.model.covar_module, "ard_num_dims", 1) == 1
        ):
            _hypers_to_set["covar_module.mixture_means"] = _init_freqs
        elif (
            use_best_band_init
            and _init_freqs is not None
            and self.ndim > 1
            and hasattr(self, "model")
            and hasattr(self.model, "covar_module")
            and hasattr(self.model.covar_module, "mixture_means")
            and getattr(self.model.covar_module, "ard_num_dims", 1) == 2
        ):
            # For 2D SM models: initialise the temporal dimension (dim 0)
            # from the best-band 1D LS frequencies and use the minimum
            # wavelength frequency (1/wavelength_span) as a placeholder for
            # the wavelength dimension (dim 1), which encodes approximately
            # achromatic variability.  This avoids leaving all mixture means
            # at GPyTorch defaults while still seeding the most informative
            # (temporal) dimension from the best-sampled band.
            _bands_raw = self._xdata_raw[:, 1]
            _wl_span = float(_bands_raw.max() - _bands_raw.min())
            _default_wl_freq = 1.0 / _wl_span if _wl_span > 0 else 1e-6
            _n_mix = len(_init_freqs)
            # Build a [num_mixtures, 2] tensor: col 0 = temporal frequencies
            # from the best-band LS, col 1 = default wavelength frequency.
            # Using new_full preserves device and dtype of _init_freqs.
            _init_freqs_2d = torch.stack(
                [
                    _init_freqs,
                    _init_freqs.new_full((_n_mix,), _default_wl_freq),
                ],
                dim=1,  # shape: [num_mixtures, 2]
            )
            # The mixture_means constraint is derived from the temporal
            # dimension and is applied element-wise to all entries,
            # including the wavelength dimension.  The wavelength
            # frequency (1/wavelength_span) may fall below the
            # temporal-based lower bound, causing a RuntimeError.
            # Clamp to the constraint bounds only when xtransform is None
            # (i.e. raw and transformed spaces are identical).  When an
            # xtransform is active, _init_freqs_2d is still in raw units
            # but the constraint bounds are in transformed space; clamping
            # in the wrong space could create new out-of-bounds values
            # after set_hypers() applies the transform, so we skip
            # clamping and let set_hypers() handle the transform instead.
            if self.xtransform is None:
                _mixture_means_constraint = getattr(
                    self.model.covar_module,
                    "raw_mixture_means_constraint",
                    None,
                )
                if _mixture_means_constraint is not None and hasattr(
                    _mixture_means_constraint, "lower_bound"
                ):
                    _clamp_lower = float(
                        _mixture_means_constraint.lower_bound
                    )
                    _clamp_upper = (
                        float(_mixture_means_constraint.upper_bound)
                        if hasattr(_mixture_means_constraint, "upper_bound")
                        else float("inf")
                    )
                    _init_freqs_2d = _init_freqs_2d.clamp(
                        min=_clamp_lower, max=_clamp_upper
                    )
            _hypers_to_set["covar_module.mixture_means"] = _init_freqs_2d
        if guess is not None:
            _hypers_to_set.update(guess)
        if _hypers_to_set:
            self.set_hypers(_hypers_to_set)

#             if guess is not None:
#                 # self.model.initialize(**guess)
#                 self.set_hypers(guess)

        if miniter is None:
            miniter = training_iter

        if max_cg_iterations is None:
            max_cg_iterations = 10000

        # Next we probably want to report some setup info
        # later...

        # Train the model
        # self.model.train()
        # self.likelihood.train()

        # set training mode:
        self._train()

        # for param_name, param in self.model.named_parameters():
        #    print(f'Parameter name: {param_name:42} value = {param.data}')
        # self.print_parameters()

        # Now actually call the trainer!
        with gpytorch.settings.max_cg_iterations(max_cg_iterations):
            self.results = train(
                self,
                maxiter=training_iter,
                miniter=miniter,
                stop=stop,
                lr=lr,
                optim=optim,
                stopavg=stopavg,
                verbose=verbose
            )
        self.__FITTED_MAP = True
        self.is_fitted = True
        self.fit_failed = False
        self.failure_reason = None
        self.failure_diagnostics = None
        self.failure_summary = None

        return self.results

    def _consensus_fit(self, fit_strategy, **fit_kwargs):
        """Dispatch consensus fit strategies to their internal handlers."""
        try:
            if fit_strategy == "consensus":
                return self._consensus_standard_fit(**fit_kwargs)
            if fit_strategy == "consensus_multicomp":
                return self._consensus_multicomp_fit(**fit_kwargs)
            if fit_strategy == "consensus_relaxed":
                return self._consensus_relaxed_fit(**fit_kwargs)
            msg = (
                "Invalid fit_strategy. Expected None or one of: "
                "'consensus', 'consensus_multicomp', 'consensus_relaxed'. "
                f"Got {fit_strategy!r}."
            )
            raise ValueError(msg)
        except ConsensusFitError as exc:
            _failure_diagnostics = getattr(exc, "failure_diagnostics", None) or {}
            _failure_reason = _failure_diagnostics.get("reason") or "consensus_failure"
            _failure_message = str(exc)
            failure_summary = self._record_failure_state(
                reason=_failure_reason,
                message=_failure_message,
                diagnostics=_failure_diagnostics,
                clear_model_state=False,
                clear_consensus=False,
            )
            exc.failure_diagnostics = self.failure_diagnostics
            exc.failure_summary = failure_summary
            raise

    def _consensus_resolve_time_spectral_mixture_keys(self):
        """Resolve time-kernel spectral-mixture parameter keys for consensus fit."""
        if (
            not hasattr(self, "model")
            or self.model is None
            or not hasattr(self, "_model_pars")
        ):
            raise RuntimeError(
                "Model has not been set yet. Call set_model() before resolving "
                "consensus SM keys."
            )

        _available = [
            k for k in self._model_pars if isinstance(k, str)
        ]

        def _resolve_param_key(param_name):
            candidates = set()
            raw_token = "raw_"
            for key, meta in self._model_pars.items():
                if not isinstance(key, str):
                    continue
                if key.startswith(raw_token) or f".{raw_token}" in key:
                    continue
                if key.endswith(f".{param_name}"):
                    candidates.add(key)
                if key == param_name and isinstance(meta, dict):
                    resolved = (
                        meta.get("constrained_full_name")
                        or meta.get("full_name")
                    )
                    if isinstance(resolved, str):
                        candidates.add(resolved)
                if isinstance(meta, dict):
                    resolved = meta.get("constrained_full_name")
                    if (
                        isinstance(resolved, str)
                        and resolved.endswith(f".{param_name}")
                    ):
                        candidates.add(resolved)

            if not candidates:
                _avail_str = ", ".join(_available[:20]) or "(none)"
                raise RuntimeError(
                    f"Could not resolve a time-kernel '{param_name}' key from "
                    f"_model_pars. Available keys: {_avail_str}. "
                    "Ensure the model uses a spectral-mixture time kernel "
                    "(e.g. model='2D' or time_kernel_type='spectral_mixture')."
                )

            def _candidate_rank(candidate):
                if candidate == f"covar_module.{param_name}":
                    return (0, len(candidate), candidate)
                if (
                    candidate.startswith("covar_module.")
                    and ".kernels.0." in candidate
                ):
                    return (1, len(candidate), candidate)
                if candidate.startswith("covar_module."):
                    return (2, len(candidate), candidate)
                return (3, len(candidate), candidate)

            ordered = sorted(
                candidates,
                key=_candidate_rank,
            )
            return ordered[0]

        return {
            "mixture_means": _resolve_param_key("mixture_means"),
            "mixture_scales": _resolve_param_key("mixture_scales"),
        }

    def _consensus_clear_model_state(self):
        """Clear stale model-related state before a fresh consensus model build.

        Removes ``self.model``, ``self.likelihood``, and ``self._model_pars``
        and resets the likelihood-set flag and the constraints-set flag so
        that the subsequent :meth:`set_model` call always starts from a clean
        slate and default constraints are reapplied to the new model.
        Light-curve data and fit history are never touched.
        """
        for _attr in ("model", "likelihood", "_model_pars"):
            try:
                setattr(self, _attr, None)
            except Exception:
                pass
        # Reset the likelihood-set guard so set_model will call set_likelihood
        # again for the new model build.
        try:
            self.__SET_LIKELIHOOD_CALLED = False
        except Exception:
            pass
        # Reset the constraints-set flag so that the new model always has
        # fresh constraints applied (either by set_default_constraints in the
        # consensus path or by _fit_core).
        try:
            self.__CONTRAINTS_SET = False
        except Exception:
            pass

    def _consensus_validate_final_model_supports_sm_time_kernel(
        self, model_name, time_kernel_type
    ):
        """Validate that the built model exposes SM time-kernel parameters.

        Parameters
        ----------
        model_name : str or None
            The model identifier passed to the current fit call.
        time_kernel_type : str or None
            The time-kernel type passed to the current fit call.

        Raises
        ------
        ConsensusFitError
            If the model does not expose ``mixture_means`` / ``mixture_scales``
            in ``_model_pars``.
        """
        try:
            self._consensus_resolve_time_spectral_mixture_keys()
        except RuntimeError as exc:
            _model_str = repr(model_name)
            _tkt_str = repr(time_kernel_type)
            _msg = (
                "Consensus constraints require a spectral-mixture time kernel, "
                f"but the model {_model_str} built with "
                f"time_kernel_type={_tkt_str} does not expose the required "
                "mixture_means / mixture_scales parameters. "
                "Pass time_kernel_type='spectral_mixture' or use model='2D'."
            )
            _exc = ConsensusFitError(_msg)
            _exc.failure_diagnostics = {
                "reason": "model_incompatible_with_sm_constraints",
                "model": model_name,
                "time_kernel_type": time_kernel_type,
                "detail": str(exc),
            }
            raise _exc from exc

    def _consensus_validate_applied_sm_constraints(
        self, keys, consensus_frequencies, frequency_bounds
    ):
        """Verify that consensus constraints were actually applied to the model.

        Inspects the registered constraint on the ``mixture_means`` parameter
        and confirms that an ``Interval`` constraint (with finite upper bound)
        is registered.  Raises :exc:`ConsensusFitError` if the constraint is
        missing or looks like a default ``GreaterThan``/``Positive`` (infinite
        upper bound), which would indicate that default constraints overwrote
        the consensus constraints.

        Parameters
        ----------
        keys : dict
            Resolved SM parameter keys as returned by
            :meth:`_consensus_resolve_time_spectral_mixture_keys`.
        consensus_frequencies : array-like
            The consensus frequency or frequencies (used in error messages).
        frequency_bounds : tuple of (float, float) or None
            The ``(lower, upper)`` bounds that were passed to
            :meth:`set_constraint`.  If ``None``, validation is skipped.

        Raises
        ------
        ConsensusFitError
            If the constraint is not found or its upper bound is infinite
            (indicating the consensus Interval constraint was not applied).
        """
        if frequency_bounds is None:
            return
        _model_pars = getattr(self, "_model_pars", None)
        _available_keys = (
            sorted(_model_pars.keys())
            if isinstance(_model_pars, dict)
            else f"<non-dict:{type(_model_pars).__name__}>"
        )
        _mm_key = keys.get("mixture_means")
        if _mm_key is None:
            _msg = (
                "Consensus constraint validation failed: resolved keys do not "
                "contain 'mixture_means' while frequency bounds were provided. "
                f"resolved_keys={sorted(keys.keys())}, "
                f"available_model_parameter_keys={_available_keys}, "
                f"expected_frequency_bounds={frequency_bounds}."
            )
            raise ConsensusFitError(_msg)
        if not isinstance(_model_pars, dict) or _mm_key not in _model_pars:
            _msg = (
                "Consensus constraint validation failed: mixture_means key "
                f"{_mm_key!r} is missing from model parameters. "
                f"available_model_parameter_keys={_available_keys}, "
                f"expected_frequency_bounds={frequency_bounds}."
            )
            raise ConsensusFitError(_msg)
        _mm_meta = _model_pars[_mm_key]
        if not isinstance(_mm_meta, dict):
            _msg = (
                "Consensus constraint validation failed: metadata for "
                f"{_mm_key!r} must be a dict, got "
                f"{type(_mm_meta).__name__}. "
                f"available_model_parameter_keys={_available_keys}, "
                f"expected_frequency_bounds={frequency_bounds}."
            )
            raise ConsensusFitError(_msg)
        _module = _mm_meta.get("module")
        if _module is None:
            _msg = (
                "Consensus constraint validation failed: no module found in "
                f"metadata for { _mm_key!r}. metadata_keys={sorted(_mm_meta.keys())}, "
                f"available_model_parameter_keys={_available_keys}, "
                f"expected_frequency_bounds={frequency_bounds}."
            )
            raise ConsensusFitError(_msg)
        _module_name = _module.__class__.__name__
        _raw_name = (
            f"raw_{_mm_key.split('.')[-1]}"
            if "raw_" not in _mm_key
            else _mm_key.split(".")[-1]
        )
        # GPyTorch stores constraints with a "_constraint" suffix in
        # named_constraints(), so the registered name is:
        #   raw_mixture_means_constraint
        _constraint_key = _raw_name + "_constraint"
        try:
            _registered = dict(_module.named_constraints())
        except Exception as exc:
            _msg = (
                "Consensus constraint validation failed: unable to inspect "
                "registered constraints via named_constraints(). "
                f"mixture_means_key={_mm_key!r}, module_class={_module_name}, "
                f"expected_raw_constraint={_constraint_key!r}, "
                f"expected_frequency_bounds={frequency_bounds}, "
                f"detail={exc!r}."
            )
            raise ConsensusFitError(_msg) from exc
        _found = _registered.get(_constraint_key)
        if _found is None:
            _registered_names = sorted(_registered.keys())
            raise ConsensusFitError(
                "Consensus constraint validation failed: no constraint found "
                f"for raw parameter {_constraint_key!r} on module "
                f"{_module_name}. The consensus constraint was not registered. "
                "This may indicate that default constraints overwrote the "
                "consensus constraints. "
                f"registered_constraint_names={_registered_names}, "
                f"mixture_means_key={_mm_key!r}, "
                f"expected_frequency_bounds={frequency_bounds}."
            )
        # Verify the constraint is an Interval (finite upper bound), not just
        # a GreaterThan or Positive (infinite upper bound). Default constraints
        # set by set_default_constraints use GreaterThan; the consensus
        # constraint sets an Interval with finite bounds. If the upper bound
        # is infinite, the consensus constraint was overwritten.
        _freqs_arr = np.asarray(consensus_frequencies, dtype=float).ravel()
        if _freqs_arr.size == 0 or not np.all(np.isfinite(_freqs_arr)):
            raise ConsensusFitError(
                "Consensus constraint validation failed: consensus_frequencies "
                "must be non-empty and finite for validation. "
                f"got={_freqs_arr.tolist()}, mixture_means_key={_mm_key!r}, "
                f"expected_frequency_bounds={frequency_bounds}."
            )
        _target_freq = float(np.median(_freqs_arr))
        try:
            import math as _math
            _lower_raw = getattr(_found, "lower_bound", None)
            _upper_raw = getattr(_found, "upper_bound", None)
            if _lower_raw is None or _upper_raw is None:
                raise ConsensusFitError(
                    "Consensus constraint validation failed: registered "
                    "constraint does not expose usable lower/upper bounds. "
                    f"constraint_type={type(_found).__name__}, "
                    f"module_class={_module_name}, mixture_means_key={_mm_key!r}, "
                    f"expected_frequency_bounds={frequency_bounds}, "
                    f"consensus_frequency={_target_freq:.6g}."
                )
            _upper = float(_upper_raw)
            _lower = float(_lower_raw)
            if _math.isinf(_upper):
                raise ConsensusFitError(
                    "Consensus constraint validation failed: the registered "
                    f"mixture_means constraint on module {_module_name} has "
                    "an infinite upper bound, indicating it is not the expected "
                    "finite consensus Interval constraint. "
                    f"mixture_means_key={_mm_key!r}, "
                    f"expected_frequency_bounds={frequency_bounds}, "
                    f"registered_bounds=[{_lower:.6g}, {_upper:.6g}], "
                    f"consensus_frequency={_target_freq:.6g}."
                )
            if not (_lower < _upper):
                raise ConsensusFitError(
                    "Consensus constraint validation failed: the registered "
                    "mixture_means constraint has invalid bounds "
                    f"[{_lower:.6g}, {_upper:.6g}] (lower >= upper). "
                    f"mixture_means_key={_mm_key!r}, module_class={_module_name}, "
                    f"expected_frequency_bounds={frequency_bounds}, "
                    f"consensus_frequency={_target_freq:.6g}."
                )
            if not (_lower <= _target_freq <= _upper):
                raise ConsensusFitError(
                    "Consensus constraint validation failed: consensus "
                    "frequency lies outside the registered mixture_means "
                    "constraint bounds. "
                    f"mixture_means_key={_mm_key!r}, module_class={_module_name}, "
                    f"expected_frequency_bounds={frequency_bounds}, "
                    f"registered_bounds=[{_lower:.6g}, {_upper:.6g}], "
                    f"consensus_frequency={_target_freq:.6g}."
                )
        except ConsensusFitError:
            raise
        except Exception as exc:
            raise ConsensusFitError(
                "Consensus constraint validation failed: unable to parse "
                "registered constraint bounds. "
                f"mixture_means_key={_mm_key!r}, module_class={_module_name}, "
                f"constraint_type={type(_found).__name__}, "
                f"expected_frequency_bounds={frequency_bounds}, "
                f"consensus_frequency={_target_freq:.6g}, detail={exc!r}."
            ) from exc

    def _consensus_get_registered_sm_constraint_bounds(self, keys):
        """Return live mixture-means constraint bounds for consensus diagnostics."""
        _model_pars = getattr(self, "_model_pars", None)
        _mm_key = keys.get("mixture_means") if isinstance(keys, dict) else None

        if not isinstance(_model_pars, dict) or _mm_key not in _model_pars:
            return None

        _mm_meta = _model_pars[_mm_key]
        if not isinstance(_mm_meta, dict):
            return None

        _module = _mm_meta.get("module")
        if _module is None:
            return None

        _raw_name = (
            f"raw_{_mm_key.split('.')[-1]}"
            if "raw_" not in _mm_key
            else _mm_key.split(".")[-1]
        )
        _constraint = getattr(_module, f"{_raw_name}_constraint", None)
        if _constraint is None:
            return None

        _lower = getattr(_constraint, "lower_bound", None)
        _upper = getattr(_constraint, "upper_bound", None)
        if _lower is None or _upper is None:
            return None

        return [float(_lower), float(_upper)]


    def _consensus_build_spectral_mixture_initialization(
        self,
        frequencies,
        scales=None,
        dtype=None,
        device=None,
    ):
        """Convert consensus frequency estimates to spectral-mixture init tensors.

        Converts consensus frequency estimates into properly-shaped tensors for
        spectral-mixture kernel initialization.

        Parameters
        ----------
        frequencies : float or list or numpy.ndarray or torch.Tensor
            Consensus frequency estimate(s). Values are converted to a 1-D
            tensor and must be non-empty, finite, and strictly positive.
        scales : float or list or numpy.ndarray or torch.Tensor or None, optional
            Optional spectral-mixture scale value(s). If a scalar is provided,
            it is broadcast to all mixtures. If array-like, it must have the
            same number of elements as ``frequencies``.
        dtype : torch.dtype or None, optional
            Tensor dtype for returned initialization tensors. If ``None``,
            inferred from ``self.xdata`` when available; otherwise uses
            ``torch.float32``.
        device : torch.device or str or None, optional
            Device for returned initialization tensors. If ``None``, inferred
            from ``self.xdata`` when available; otherwise uses CPU.

        Returns
        -------
        dict
            Initialization dictionary containing ``mixture_means`` with shape
            ``(1, n_mixtures, 1)``, ``mixture_scales`` (same shape when scales
            are provided, otherwise ``None``), and ``num_mixtures``.

        Raises
        ------
        ValueError
            If frequencies or scales fail validation checks.
        """
        xdata_tensor = (
            self.xdata
            if hasattr(self, "xdata") and isinstance(self.xdata, torch.Tensor)
            else None
        )
        if dtype is None:
            dtype = (
                xdata_tensor.dtype
                if xdata_tensor is not None
                else torch.float32
            )
        if device is None:
            device = (
                xdata_tensor.device
                if xdata_tensor is not None
                else torch.device("cpu")
            )

        freq_tensor = torch.as_tensor(frequencies, dtype=dtype, device=device)
        freq_tensor = freq_tensor.reshape(-1)

        if freq_tensor.numel() == 0:
            raise ValueError("frequencies must not be empty.")
        if not torch.all(torch.isfinite(freq_tensor)):
            raise ValueError("frequencies must contain only finite values.")
        if not torch.all(freq_tensor > 0):
            raise ValueError("frequencies must be strictly positive.")

        n_mixtures = freq_tensor.numel()
        mixture_means = freq_tensor.reshape(1, n_mixtures, 1)
        init = {
            "mixture_means": mixture_means,
            "mixture_scales": None,
            "num_mixtures": int(n_mixtures),
        }

        if scales is not None:
            scales_tensor = torch.as_tensor(scales, dtype=dtype, device=device)
            scales_tensor = scales_tensor.reshape(-1)
            if scales_tensor.numel() == 0:
                raise ValueError("scales must not be empty when provided.")
            if not torch.all(torch.isfinite(scales_tensor)):
                raise ValueError("scales must contain only finite values.")
            if not torch.all(scales_tensor > 0):
                raise ValueError("scales must be strictly positive.")

            if scales_tensor.numel() == 1 and n_mixtures > 1:
                scales_tensor = scales_tensor.expand(n_mixtures)
            elif scales_tensor.numel() != n_mixtures:
                raise ValueError(
                    "scales must be a scalar or have the same number of elements "
                    "as frequencies."
                )

            init["mixture_scales"] = scales_tensor.reshape(1, n_mixtures, 1)

        return init

    def _consensus_build_guess(
        self,
        frequencies,
        scales=None,
        dtype=None,
        device=None,
    ):
        """Build a model-key-aware spectral-mixture init dictionary."""
        keys = self._consensus_resolve_time_spectral_mixture_keys()
        init = self._consensus_build_spectral_mixture_initialization(
            frequencies=frequencies,
            scales=scales,
            dtype=dtype,
            device=device,
        )

        expected_num_mixtures = getattr(self, "_fit_num_mixtures_effective", None)
        if (
            expected_num_mixtures is not None
            and int(expected_num_mixtures) != init["num_mixtures"]
        ):
            raise ValueError(
                "The number of consensus frequencies does not match the model's "
                f"number of mixtures ({init['num_mixtures']} != "
                f"{int(expected_num_mixtures)})."
            )

        guess = {
            keys["mixture_means"]: init["mixture_means"],
        }
        if init["mixture_scales"] is not None:
            guess[keys["mixture_scales"]] = init["mixture_scales"]

        return guess

    def _consensus_extract_initialized_parameter_vector(
        self,
        param_key,
        *,
        source_guess=None,
    ):
        """Return per-mixture values from an initialized spectral-mixture parameter.

        Parameters
        ----------
        param_key : str
            Fully-qualified constrained parameter key resolved from
            :meth:`_consensus_resolve_time_spectral_mixture_keys`.
        source_guess : dict or None, optional
            Initialization guess dictionary used as a fallback source when the
            parameter cannot be read from the current model instance.

        Returns
        -------
        numpy.ndarray
            Flattened 1D float array with one value per mixture component.
            For multidimensional tensors, the time dimension (index ``0`` of
            the last axis) is returned to stay consistent with consensus-time
            initialization semantics.
        """
        tensor_value = None
        if (
            hasattr(self, "_model_pars")
            and isinstance(self._model_pars, dict)
            and isinstance(param_key, str)
        ):
            meta = self._model_pars.get(param_key)
            if isinstance(meta, dict):
                module = meta.get("module")
                attr_name = param_key.split(".")[-1]
                if module is not None and hasattr(module, attr_name):
                    tensor_value = getattr(module, attr_name)

        if (
            tensor_value is None
            and hasattr(self, "model")
            and self.model is not None
            and isinstance(param_key, str)
        ):
            current = self.model
            for component in param_key.split("."):
                if current is None or not hasattr(current, component):
                    current = None
                    break
                current = getattr(current, component)
            tensor_value = current

        if tensor_value is None and isinstance(source_guess, dict):
            tensor_value = source_guess.get(param_key)

        if tensor_value is None:
            return np.asarray([], dtype=float)
        if torch.is_tensor(tensor_value):
            values = tensor_value.detach().cpu().numpy()
        else:
            values = np.asarray(tensor_value, dtype=float)

        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.asarray([], dtype=float)
        if values.ndim == 0:
            return np.asarray([float(values)], dtype=float)
        if values.ndim == 1:
            return values.astype(float).ravel()

        flattened = values.reshape(-1, values.shape[-1])
        return flattened[:, 0].astype(float).ravel()

    def _consensus_collect_initialization_diagnostics(
        self,
        *,
        requested_consensus_frequencies,
        requested_consensus_scales,
        consensus_guess,
    ):
        """Collect and validate multicomp initialization diagnostics.

        Parameters
        ----------
        requested_consensus_frequencies : array-like
            Consensus frequencies requested by the multicomp consensus
            aggregator.
        requested_consensus_scales : array-like
            Consensus scales requested by the multicomp consensus aggregator.
        consensus_guess : dict
            Initialization guess dictionary produced by
            :meth:`_consensus_build_guess`. This helper reapplies the guess via
            :meth:`set_hypers` before reading initialized parameters so the
            diagnostics reflect the exact post-initialization model state.

        Returns
        -------
        dict
            JSON-safe diagnostics containing requested and initialized mixture
            means/scales plus the initialization strategy label.

        Raises
        ------
        RuntimeError
            If requested frequency/scale shapes are inconsistent, if requested
            and initialized component counts are inconsistent, or if consensus
            guesses cannot be applied to the model.
        """
        requested_frequencies = np.asarray(
            requested_consensus_frequencies, dtype=float
        ).ravel()
        requested_scales = np.asarray(
            requested_consensus_scales, dtype=float
        ).ravel()
        if not np.all(np.isfinite(requested_frequencies) & (requested_frequencies > 0.0)):
            raise RuntimeError(
                "Consensus multi-component initialization failed: requested "
                "consensus frequencies must be finite and strictly positive."
            )
        if requested_scales.shape != requested_frequencies.shape:
            raise RuntimeError(
                "Consensus multi-component initialization failed: requested "
                "consensus scales must align one-to-one with requested "
                "consensus frequencies "
                f"(scales shape={requested_scales.shape}, "
                f"frequencies shape={requested_frequencies.shape})."
            )

        if (
            isinstance(consensus_guess, dict)
            and consensus_guess
            and hasattr(self, "model")
            and self.model is not None
            and callable(getattr(self.model, "initialize", None))
        ):
            try:
                self.set_hypers(dict(consensus_guess))
            except Exception as exc:
                raise RuntimeError(
                    "Consensus multi-component initialization failed while "
                    "applying consensus guesses to the model."
                ) from exc

        keys = self._consensus_resolve_time_spectral_mixture_keys()
        initialized_means = self._consensus_extract_initialized_parameter_vector(
            keys["mixture_means"]
        )
        initialized_scales = self._consensus_extract_initialized_parameter_vector(
            keys["mixture_scales"]
        )
        if initialized_means.size != requested_frequencies.size:
            raise RuntimeError(
                "Consensus multi-component initialization mismatch: "
                f"len(initialized_mixture_means)={int(initialized_means.size)} "
                "does not match "
                "len(requested_consensus_frequencies)="
                f"{int(requested_frequencies.size)}."
            )
        if initialized_scales.size != requested_scales.size:
            raise RuntimeError(
                "Consensus multi-component initialization mismatch: "
                f"len(initialized_mixture_scales)={int(initialized_scales.size)} "
                "does not match "
                f"len(requested_consensus_scales)={int(requested_scales.size)}."
            )
        if not np.all(np.isfinite(initialized_means) & (initialized_means > 0.0)):
            raise RuntimeError(
                "Consensus multi-component initialization mismatch: initialized "
                "mixture means must be finite and strictly positive frequencies."
            )

        initialized_periods = 1.0 / initialized_means
        initialized_period_widths = initialized_scales / (initialized_means**2)

        return self._consensus_make_json_safe(
            {
                "requested_consensus_frequencies": requested_frequencies.tolist(),
                "requested_consensus_scales": requested_scales.tolist(),
                "initialized_mixture_means": initialized_means.tolist(),
                "initialized_mixture_periods": initialized_periods.tolist(),
                "initialized_mixture_scales": initialized_scales.tolist(),
                "initialized_mixture_period_widths": initialized_period_widths.tolist(),
                "initialization_strategy": (
                    "per_component_consensus_initialization"
                ),
            }
        )

    def _consensus_collect_fitted_mixture_diagnostics(
        self,
        *,
        initialized_mixture_frequencies,
        initialized_mixture_scales,
    ):
        """Collect post-fit multicomp mixture diagnostics.

        Parameters
        ----------
        initialized_mixture_frequencies : array-like
            Initialized per-component mixture frequencies used as the canonical
            component ordering reference.
        initialized_mixture_scales : array-like
            Initialized per-component mixture scales aligned one-to-one with
            ``initialized_mixture_frequencies``.

        Returns
        -------
        dict
            JSON-safe diagnostics aligned to initialization component order
            containing:
            ``fitted_mixture_frequencies``,
            ``fitted_mixture_periods``,
            ``fitted_mixture_scales``,
            ``fitted_mixture_period_widths``,
            ``fitted_frequency_shift_from_initialization``,
            ``fitted_period_shift_from_initialization``,
            ``fitted_fractional_frequency_shift_from_initialization``, and
            ``fitted_fractional_period_shift_from_initialization``.
        """
        initialized_frequencies = np.asarray(
            initialized_mixture_frequencies, dtype=float
        ).ravel()
        initialized_scales = np.asarray(initialized_mixture_scales, dtype=float).ravel()
        if initialized_frequencies.shape != initialized_scales.shape:
            raise RuntimeError(
                "Consensus multi-component fitted diagnostics mismatch: "
                "initialized mixture frequencies and scales must have identical "
                "shapes "
                f"(frequencies shape={initialized_frequencies.shape}, "
                f"scales shape={initialized_scales.shape})."
            )
        if not np.all(
            np.isfinite(initialized_frequencies) & (initialized_frequencies > 0.0)
        ):
            raise RuntimeError(
                "Consensus multi-component fitted diagnostics mismatch: "
                "initialized mixture frequencies must be finite and strictly "
                "positive."
            )
        if not np.all(np.isfinite(initialized_scales) & (initialized_scales > 0.0)):
            raise RuntimeError(
                "Consensus multi-component fitted diagnostics mismatch: "
                "initialized mixture scales must be finite and strictly positive."
            )

        keys = self._consensus_resolve_time_spectral_mixture_keys()
        fitted_frequencies = self._consensus_extract_initialized_parameter_vector(
            keys["mixture_means"]
        )
        fitted_scales = self._consensus_extract_initialized_parameter_vector(
            keys["mixture_scales"]
        )
        if fitted_frequencies.size != initialized_frequencies.size:
            raise RuntimeError(
                "Consensus multi-component fitted diagnostics mismatch: "
                f"len(fitted_mixture_frequencies)={int(fitted_frequencies.size)} "
                "does not match "
                "len(initialized_mixture_means)="
                f"{int(initialized_frequencies.size)}."
            )
        if fitted_scales.size != initialized_scales.size:
            raise RuntimeError(
                "Consensus multi-component fitted diagnostics mismatch: "
                f"len(fitted_mixture_scales)={int(fitted_scales.size)} "
                "does not match "
                f"len(initialized_mixture_scales)={int(initialized_scales.size)}."
            )
        if not np.all(
            np.isfinite(fitted_frequencies) & (fitted_frequencies > 0.0)
        ):
            raise RuntimeError(
                "Consensus multi-component fitted diagnostics mismatch: fitted "
                "mixture frequencies must be finite and strictly positive."
            )
        if not np.all(np.isfinite(fitted_scales) & (fitted_scales > 0.0)):
            raise RuntimeError(
                "Consensus multi-component fitted diagnostics mismatch: fitted "
                "mixture scales must be finite and strictly positive."
            )

        initialized_periods = 1.0 / initialized_frequencies
        fitted_periods = 1.0 / fitted_frequencies
        fitted_period_widths = fitted_scales / (fitted_frequencies**2)
        frequency_shift = fitted_frequencies - initialized_frequencies
        period_shift = fitted_periods - initialized_periods
        fractional_frequency_shift = frequency_shift / initialized_frequencies
        fractional_period_shift = period_shift / initialized_periods

        return self._consensus_make_json_safe(
            {
                "fitted_mixture_frequencies": fitted_frequencies.tolist(),
                "fitted_mixture_periods": fitted_periods.tolist(),
                "fitted_mixture_scales": fitted_scales.tolist(),
                "fitted_mixture_period_widths": fitted_period_widths.tolist(),
                "fitted_frequency_shift_from_initialization": frequency_shift.tolist(),
                "fitted_period_shift_from_initialization": period_shift.tolist(),
                "fitted_fractional_frequency_shift_from_initialization": (
                    fractional_frequency_shift.tolist()
                ),
                "fitted_fractional_period_shift_from_initialization": (
                    fractional_period_shift.tolist()
                ),
            }
        )

    @staticmethod
    def _consensus_compute_component_identity_diagnostics(
        *,
        initialized_periods,
        fitted_periods,
        initialized_frequencies=None,
        fitted_frequencies=None,
    ):
        """Compute per-component identity diagnostics for multicomp fits.

        Period-space fractional distance is the primary identity metric.
        Ties are resolved deterministically via ``np.argmin`` which returns
        the lowest index among equal minima.
        """
        initialized_periods = np.asarray(initialized_periods, dtype=float).ravel()
        fitted_periods = np.asarray(fitted_periods, dtype=float).ravel()
        if initialized_periods.shape != fitted_periods.shape:
            raise RuntimeError(
                "Consensus multi-component identity diagnostics mismatch: "
                "initialized and fitted period vectors must share identical shapes."
            )
        if initialized_periods.size == 0:
            raise RuntimeError(
                "Consensus multi-component identity diagnostics mismatch: "
                "component vectors must be non-empty."
            )
        if not np.all(np.isfinite(initialized_periods) & (initialized_periods > 0.0)):
            raise RuntimeError(
                "Consensus multi-component identity diagnostics mismatch: "
                "initialized periods must be finite and strictly positive."
            )
        if not np.all(np.isfinite(fitted_periods) & (fitted_periods > 0.0)):
            raise RuntimeError(
                "Consensus multi-component identity diagnostics mismatch: "
                "fitted periods must be finite and strictly positive."
            )

        period_distance_matrix = (
            np.abs(fitted_periods[:, None] - initialized_periods[None, :])
            / initialized_periods[None, :]
        )
        if not np.all(np.isfinite(period_distance_matrix)):
            raise RuntimeError(
                "Consensus multi-component identity diagnostics mismatch: "
                "period distance matrix contains non-finite values."
            )
        nearest_index = np.argmin(period_distance_matrix, axis=1).astype(int)
        nearest_period_distance = period_distance_matrix[
            np.arange(initialized_periods.size), nearest_index
        ]
        component_indices = np.arange(initialized_periods.size, dtype=int)
        identity_preserved = nearest_index == component_indices
        possible_swaps = np.flatnonzero(~identity_preserved).astype(int)

        diagnostics = {
            "nearest_initialized_component_index": nearest_index.tolist(),
            "nearest_initialized_component_fractional_period_distance": (
                nearest_period_distance.tolist()
            ),
            "component_identity_preserved": identity_preserved.tolist(),
            "all_component_identities_preserved": bool(np.all(identity_preserved)),
            "possible_component_swaps": possible_swaps.tolist(),
            "nearest_initialized_component_fractional_frequency_distance": [],
        }

        if initialized_frequencies is not None and fitted_frequencies is not None:
            initialized_frequencies = np.asarray(
                initialized_frequencies, dtype=float
            ).ravel()
            fitted_frequencies = np.asarray(fitted_frequencies, dtype=float).ravel()
            if initialized_frequencies.shape != initialized_periods.shape:
                raise RuntimeError(
                    "Consensus multi-component identity diagnostics mismatch: "
                    "initialized frequency and period vectors must align."
                )
            if fitted_frequencies.shape != fitted_periods.shape:
                raise RuntimeError(
                    "Consensus multi-component identity diagnostics mismatch: "
                    "fitted frequency and period vectors must align."
                )
            if not np.all(
                np.isfinite(initialized_frequencies) & (initialized_frequencies > 0.0)
            ):
                raise RuntimeError(
                    "Consensus multi-component identity diagnostics mismatch: "
                    "initialized frequencies must be finite and strictly positive."
                )
            if not np.all(
                np.isfinite(fitted_frequencies) & (fitted_frequencies > 0.0)
            ):
                raise RuntimeError(
                    "Consensus multi-component identity diagnostics mismatch: "
                    "fitted frequencies must be finite and strictly positive."
                )
            frequency_distance_matrix = (
                np.abs(
                    fitted_frequencies[:, None] - initialized_frequencies[None, :]
                )
                / initialized_frequencies[None, :]
            )
            if not np.all(np.isfinite(frequency_distance_matrix)):
                raise RuntimeError(
                    "Consensus multi-component identity diagnostics mismatch: "
                    "frequency distance matrix contains non-finite values."
                )
            diagnostics[
                "nearest_initialized_component_fractional_frequency_distance"
            ] = frequency_distance_matrix[
                np.arange(initialized_periods.size), nearest_index
            ].tolist()

        return diagnostics

    def _consensus_iter_band_lightcurves(self):
        """Yield per-band 1D light curves using stored band-label metadata.

        Yields
        ------
        tuple[str, Lightcurve]
            Pairs of ``(band_label, band_lightcurve_1d)``. Each returned light
            curve contains only time (1-D xdata), flux, and optional flux
            uncertainty for that band.

        Raises
        ------
        ValueError
            If this light curve is not 2-D, or if per-row band labels are not
            available/consistent.
        """
        if self.ndim <= 1:
            raise ValueError(
                "fit_strategy='consensus' requires a 2D (multiband) Lightcurve."
            )
        if self.band is None:
            raise ValueError(
                "fit_strategy='consensus' requires per-row band labels in "
                "Lightcurve.band for multiband splitting."
            )
        if len(self.band) != len(self._xdata_raw):
            raise ValueError(
                "fit_strategy='consensus' requires one band label per "
                "observation row for 2D light curves."
            )

        band_arr = np.asarray(self.band, dtype=str)
        unique_bands = list(dict.fromkeys(band_arr.tolist()))

        for band_label in unique_bands:
            mask_np = band_arr == band_label
            mask = torch.as_tensor(
                mask_np,
                dtype=torch.bool,
                device=self._xdata_raw.device,
            )
            t = self._xdata_raw[mask, 0].clone()
            y = self._ydata_raw[mask].clone()
            yerr = (
                self._yerr_raw[mask].clone()
                if hasattr(self, "_yerr_raw") and self._yerr_raw is not None
                else None
            )
            lc_band = Lightcurve(
                t,
                y,
                yerr=yerr,
                xtransform=self.xtransform,
                ytransform=self.ytransform,
                name=self.name,
                band=np.asarray([band_label], dtype=np.str_),
            )
            yield str(band_label), lc_band

    def _consensus_resolve_controls(
        self,
        metrics_by_band,
        min_points_per_band=None,
        max_gap_fraction=None,
        min_duty_cycle=None,
        outlier_sigma=None,
        consensus_width_factor=None,
    ):
        """Resolve consensus-control defaults from per-band sampling metrics.

        Any control set explicitly by the caller is used as-is. Missing controls
        are derived conservatively from the observed per-band sampling metrics.
        """
        valid_metrics = [
            m
            for m in metrics_by_band.values()
            if isinstance(m, dict) and "error" not in m
        ]

        if valid_metrics:
            n_points_vals = np.asarray(
                [float(m.get("n_points", np.nan)) for m in valid_metrics],
                dtype=float,
            )
            gap_vals = np.asarray(
                [float(m.get("max_gap_fraction", np.nan)) for m in valid_metrics],
                dtype=float,
            )
            duty_vals = np.asarray(
                [float(m.get("duty_cycle", np.nan)) for m in valid_metrics],
                dtype=float,
            )
        else:
            n_points_vals = np.asarray([8.0], dtype=float)
            gap_vals = np.asarray([0.4], dtype=float)
            duty_vals = np.asarray([0.1], dtype=float)

        if min_points_per_band is None:
            min_points_per_band = int(
                np.clip(np.nanpercentile(n_points_vals, 25), 8, 25)
            )
        if max_gap_fraction is None:
            max_gap_fraction = float(
                np.clip(np.nanmedian(gap_vals) * 1.5, 0.25, 0.8)
            )
        if min_duty_cycle is None:
            min_duty_cycle = float(np.clip(np.nanmedian(duty_vals) * 0.5, 0.02, 0.3))
        if outlier_sigma is None:
            outlier_sigma = 3.5
        if consensus_width_factor is None:
            consensus_width_factor = 3.0

        if min_points_per_band < 2:
            raise ValueError("min_points_per_band must be >= 2.")
        if not (0 < max_gap_fraction <= 1):
            raise ValueError("max_gap_fraction must be in the interval (0, 1].")
        if not (0 <= min_duty_cycle <= 1):
            raise ValueError("min_duty_cycle must be in the interval [0, 1].")
        if not (np.isfinite(outlier_sigma) and outlier_sigma > 0):
            raise ValueError("outlier_sigma must be positive and finite.")
        if not (
            np.isfinite(consensus_width_factor) and consensus_width_factor > 0
        ):
            raise ValueError(
                "consensus_width_factor must be positive and finite."
            )

        return {
            "min_points_per_band": int(min_points_per_band),
            "max_gap_fraction": float(max_gap_fraction),
            "min_duty_cycle": float(min_duty_cycle),
            "outlier_sigma": float(outlier_sigma),
            "consensus_width_factor": float(consensus_width_factor),
        }

    @staticmethod
    def _consensus_reject_bad_bands(
        metrics,
        min_points_per_band,
        max_gap_fraction,
        min_duty_cycle,
    ):
        """Return a list of deterministic rejection reasons from sampling metrics."""
        reasons = []
        if not isinstance(metrics, dict):
            return [_CONSENSUS_REJECTION_REASON_SAMPLING_METRICS_UNAVAILABLE]
        if "error" in metrics:
            reasons.append(str(metrics["error"]))
            return reasons

        n_points = float(metrics.get("n_points", np.nan))
        if not (np.isfinite(n_points) and n_points >= min_points_per_band):
            reasons.append(
                f"{_CONSENSUS_REJECTION_REASON_PREFIX_TOO_FEW_POINTS}"
                f"{n_points:g} < {int(min_points_per_band)})"
            )

        gap_fraction = float(metrics.get("max_gap_fraction", np.nan))
        if not (np.isfinite(gap_fraction) and gap_fraction <= max_gap_fraction):
            reasons.append(
                f"{_CONSENSUS_REJECTION_REASON_PREFIX_MAX_GAP_FRACTION}"
                f"{gap_fraction:.3g} > {max_gap_fraction:.3g})"
            )

        duty_cycle = float(metrics.get("duty_cycle", np.nan))
        if not (np.isfinite(duty_cycle) and duty_cycle >= min_duty_cycle):
            reasons.append(
                f"{_CONSENSUS_REJECTION_REASON_PREFIX_DUTY_CYCLE}"
                f"{duty_cycle:.3g} < {min_duty_cycle:.3g})"
            )

        return reasons

    def _consensus_prepare_band_consensus_inputs(
        self,
        *,
        min_points_per_band=None,
        max_gap_fraction=None,
        min_duty_cycle=None,
        include_wavelengths=False,
    ):
        """Prepare shared per-band consensus inputs.

        This helper centralizes the common setup shared by the single-candidate
        and multi-component candidate extraction paths: per-band splitting,
        sampling-metric computation, conservative control resolution, and
        optional wavelength lookup from ``xdata[:, 1]``.
        """
        per_band_lc = dict(self._consensus_iter_band_lightcurves())
        metrics_by_band = {
            band: lc_band.compute_sampling_metrics()
            for band, lc_band in per_band_lc.items()
        }
        controls = self._consensus_resolve_controls(
            metrics_by_band=metrics_by_band,
            min_points_per_band=min_points_per_band,
            max_gap_fraction=max_gap_fraction,
            min_duty_cycle=min_duty_cycle,
        )

        prepared = {
            "per_band_lc": per_band_lc,
            "metrics_by_band": metrics_by_band,
            "controls": controls,
        }

        if include_wavelengths:
            prepared["band_to_wavelength"] = (
                self._consensus_build_band_wavelength_map(per_band_lc.keys())
            )

        return prepared

    def _consensus_build_band_wavelength_map(self, band_labels):
        """Map each band label to a representative wavelength value.

        Returns ``None`` for bands that cannot be mapped safely (missing rows,
        missing wavelength column, or non-numeric values).
        """
        band_to_wavelength = {str(label): None for label in band_labels}
        band_arr = np.asarray(self.band, dtype=str)

        try:
            xdata_np = self._xdata_raw.detach().cpu().numpy()
        except Exception:
            return band_to_wavelength

        if xdata_np.ndim != 2 or xdata_np.shape[1] < 2:
            return band_to_wavelength

        for band_label in band_to_wavelength:
            mask_np = band_arr == band_label
            if not np.any(mask_np):
                continue
            try:
                band_wavelengths = xdata_np[mask_np, 1]
            except Exception:
                continue
            if np.size(band_wavelengths) == 0:
                continue
            try:
                band_to_wavelength[band_label] = float(
                    np.ravel(band_wavelengths)[0]
                )
            except (TypeError, ValueError):
                band_to_wavelength[band_label] = None

        return band_to_wavelength

    @staticmethod
    def _consensus_lookup_ls_peak_metadata(
        *,
        ls_frequency,
        freq_grid,
        power_grid,
        peak_idx_to_prominence,
        rtol=1e-7,
        atol=1e-12,
    ):
        """Resolve LS peak metadata only when the frequency is grid-aligned."""
        peak_power = np.nan
        peak_prominence = np.nan

        ls_freq = float(ls_frequency)
        if not np.isfinite(ls_freq):
            return peak_power, peak_prominence

        freq_np = np.asarray(freq_grid, dtype=float)
        power_np = np.asarray(power_grid, dtype=float)
        if freq_np.size == 0 or power_np.size == 0:
            return peak_power, peak_prominence

        closest_idx = int(np.argmin(np.abs(freq_np - ls_freq)))
        if closest_idx < 0 or closest_idx >= power_np.size:
            return peak_power, peak_prominence
        if not np.isclose(freq_np[closest_idx], ls_freq, rtol=rtol, atol=atol):
            return peak_power, peak_prominence

        peak_power = float(power_np[closest_idx])
        peak_prominence = float(peak_idx_to_prominence.get(closest_idx, np.nan))
        return peak_power, peak_prominence

    @staticmethod
    def _consensus_extract_band_ls_candidates(
        lc_band,
        *,
        metrics,
        num_requested_peaks,
        max_candidates=None,
        include_peak_metadata=False,
    ):
        """Run per-band LS and return plausible candidates in raw LS order."""
        if include_peak_metadata:
            from scipy.signal import find_peaks, peak_prominences

        ls_kwargs = {"num_peaks": int(num_requested_peaks)}
        if include_peak_metadata:
            ls_kwargs["return_full"] = True

        ls_result = lc_band.fit_LS(**ls_kwargs)
        if include_peak_metadata:
            ls_freqs, ls_sig, freq_grid, power_grid = ls_result
            freq_np = np.asarray(freq_grid.detach().cpu().numpy(), dtype=float)
            power_np = np.asarray(power_grid.detach().cpu().numpy(), dtype=float)
        else:
            ls_freqs, ls_sig = ls_result
            freq_np = np.asarray([], dtype=float)
            power_np = np.asarray([], dtype=float)

        ls_freqs_np = np.asarray(ls_freqs.detach().cpu().numpy(), dtype=float)
        ls_sig_np = np.asarray(ls_sig.detach().cpu().numpy(), dtype=bool)

        baseline = float(metrics.get("baseline", np.nan))
        longest_period = float(metrics.get("longest_detectable_period", np.nan))
        if not (np.isfinite(longest_period) and longest_period > 0):
            longest_period = baseline / 2.0 if np.isfinite(baseline) else np.nan
        min_detectable_frequency = (
            float(1.0 / longest_period)
            if (np.isfinite(longest_period) and longest_period > 0)
            else 0.0
        )
        nyquist_freq = float(metrics.get("nyquist_frequency", np.inf))

        if include_peak_metadata and len(power_np) > 0:
            all_peak_idx, _ = find_peaks(power_np, distance=5)
            if len(all_peak_idx) > 0:
                prom_values, _, _ = peak_prominences(power_np, all_peak_idx)
                peak_idx_to_prominence = dict(
                    zip(all_peak_idx.tolist(), prom_values.tolist(), strict=True)
                )
            else:
                peak_idx_to_prominence = {}
        else:
            peak_idx_to_prominence = {}

        candidates = []
        for rank, ls_freq in enumerate(ls_freqs_np):
            if max_candidates is not None and len(candidates) >= max_candidates:
                break
            if not (np.isfinite(ls_freq) and ls_freq > 0):
                continue
            if np.isfinite(nyquist_freq) and ls_freq > nyquist_freq:
                continue
            if min_detectable_frequency > 0 and ls_freq < min_detectable_frequency:
                continue

            candidate = {
                "frequency": float(ls_freq),
                "period": float(1.0 / ls_freq),
                "ls_rank": rank,
                "significant": bool(rank < ls_sig_np.size and ls_sig_np[rank]),
            }

            if include_peak_metadata:
                peak_power, peak_prominence = (
                    Lightcurve._consensus_lookup_ls_peak_metadata(
                        ls_frequency=ls_freq,
                        freq_grid=freq_np,
                        power_grid=power_np,
                        peak_idx_to_prominence=peak_idx_to_prominence,
                    )
                )
                candidate["peak_power"] = peak_power
                candidate["peak_prominence"] = peak_prominence

            candidates.append(candidate)

        return {
            "ls_frequencies": ls_freqs_np,
            "ls_significant": ls_sig_np,
            "candidates": candidates,
            "min_detectable_frequency": min_detectable_frequency,
            "nyquist_frequency": nyquist_freq,
        }

    @staticmethod
    def _consensus_extract_acf_candidate(acf_result):
        """Extract the strongest non-zero-lag ACF peak as a frequency candidate.

        ACF peaks are located in lag space [days].  The dominant lag is then
        converted to a frequency [1/day] which is the primary quantity used
        by the consensus machinery.

        Convention
        ----------
        INTERNAL : frequency [1/day]  (``"frequency"`` key in returned dict)
        USER-FACING : period [day]    (``"period"`` key — display/diagnostics only)

        Parameters
        ----------
        acf_result : ACFResult or None
            Output from :meth:`acf(method="data")`. If unavailable or invalid,
            no candidate is returned.

        Returns
        -------
        dict or None
            ``{"frequency": ..., "period": ...}`` where ``"frequency"``
            [1/day] is the primary consensus quantity and ``"period"`` [day]
            is retained for backward-compatible display only.  Returns ``None``
            when no robust candidate can be derived.

        Notes
        -----
        If ``scipy.signal.find_peaks`` is unavailable, this helper degrades
        gracefully and returns ``None`` so consensus vetting can continue in
        LS-only mode.
        """
        if acf_result is None:
            return None
        lag = acf_result.lag.detach().cpu().numpy()
        acf_vals = acf_result.acf.detach().cpu().numpy()
        if lag.size < 3 or acf_vals.size < 3:
            return None

        # Drop the zero-lag bin; all positive lags remain.
        lag = lag[1:]
        acf_vals = acf_vals[1:]
        valid = np.isfinite(lag) & np.isfinite(acf_vals) & (lag > 0)
        if not np.any(valid):
            return None
        lag = lag[valid]
        acf_vals = acf_vals[valid]
        if lag.size < 3:
            return None

        if _scipy_find_peaks is None:
            peaks = np.asarray([], dtype=int)
        else:
            peaks, _ = _scipy_find_peaks(acf_vals)

        if peaks.size == 0:
            return None

        best_idx = peaks[np.argmax(acf_vals[peaks])]
        # acf_lag is the dominant ACF lag [days]; convert to frequency for
        # consensus logic.  "period" is kept only for user-facing display.
        acf_lag = float(lag[best_idx])
        if not (np.isfinite(acf_lag) and acf_lag > 0):
            return None
        candidate_frequency = float(1.0 / acf_lag)
        return {
            "frequency": candidate_frequency,
            "period": acf_lag,  # display-only; primary key is "frequency"
        }

    @staticmethod
    def _consensus_fractional_frequency_difference(f1, f2):
        """Return the standard consensus fractional frequency difference.

        Definition
        ----------
        ``abs(f1 - f2) / min(f1, f2)``
        The smaller-frequency normalisation keeps agreement checks symmetric in
        frequency space while measuring mismatch relative to the slower
        timescale represented by the pair.

        Parameters
        ----------
        f1 : float
            First positive frequency [1/day].
        f2 : float
            Second positive frequency [1/day].

        Returns
        -------
        float
            Fractional frequency difference for consensus agreement tests.

        Raises
        ------
        ValueError
            If either frequency is not finite and strictly positive.
        """
        f1_val = float(f1)
        f2_val = float(f2)
        if not (
            np.isfinite(f1_val)
            and np.isfinite(f2_val)
            and f1_val > 0
            and f2_val > 0
        ):
            raise ValueError(
                "fractional frequency difference requires finite, positive "
                "frequencies."
            )
        return abs(f1_val - f2_val) / min(f1_val, f2_val)

    @staticmethod
    def _consensus_compare_ls_acf(
        ls_frequency,
        acf_frequency,
        harmonic_tolerance=0.15,
    ):
        """Compare LS and ACF dominant frequencies for consistency checks.

        All internal comparisons are performed in frequency space [1/day].
        ACF is treated as an independent periodicity diagnostic. Bands where
        LS and ACF strongly disagree are rejected because the dominant LS peak
        is less likely to reflect the shared physical timescale.

        The ``ratio`` returned is ``larger_frequency / smaller_frequency``.
        Because both inputs represent the same physical cycle rate, a harmonic
        relationship in frequency space (f1 = n * f2) yields the same integer
        ratio as the equivalent period-space check, so harmonic detection is
        unaffected by the convention change.

        Parameters
        ----------
        ls_frequency : float
            Dominant frequency [1/day] derived from Lomb-Scargle for a band.
        acf_frequency : float
            Dominant frequency [1/day] derived from ACF for the same band.
        harmonic_tolerance : float, optional
            Relative tolerance used to classify direct or harmonic agreement.

        Returns
        -------
        dict
            Comparison summary with keys:
            ``status`` (``"agreement"``, ``"harmonic"``,
            ``"disagreement"``, or ``"unavailable"``),
            ``ratio`` (larger/smaller frequency ratio), and
            ``harmonic_order`` (integer harmonic when applicable).
        """
        ls_val = float(ls_frequency) if ls_frequency is not None else np.nan
        acf_val = float(acf_frequency) if acf_frequency is not None else np.nan

        if not (
            np.isfinite(ls_val)
            and np.isfinite(acf_val)
            and ls_val > 0
            and acf_val > 0
        ):
            return {
                "status": _ACF_STATUS_UNAVAILABLE,
                "ratio": None,
                "harmonic_order": None,
            }

        larger = max(ls_val, acf_val)
        smaller = min(ls_val, acf_val)
        ratio = larger / smaller

        if not (np.isfinite(ratio) and ratio > 0):
            return {
                "status": _ACF_STATUS_UNAVAILABLE,
                "ratio": None,
                "harmonic_order": None,
            }

        if abs(ratio - 1.0) <= harmonic_tolerance:
            return {
                "status": _ACF_STATUS_AGREEMENT,
                "ratio": float(ratio),
                "harmonic_order": 1,
            }

        for harmonic_order in (2, 3, 4):
            harmonic_target = float(harmonic_order)
            if (
                abs(ratio - harmonic_target) / harmonic_target
                <= harmonic_tolerance
            ):
                return {
                    "status": _ACF_STATUS_HARMONIC,
                    "ratio": float(ratio),
                    "harmonic_order": int(harmonic_order),
                }

        return {
            "status": _ACF_STATUS_DISAGREEMENT,
            "ratio": float(ratio),
            "harmonic_order": None,
        }

    @staticmethod
    def _consensus_make_json_safe(value):
        """Return a JSON-safe copy of a consensus diagnostic value.

        Consensus diagnostics are consumed by plotting, inspection utilities,
        and future JSON export/recovery workflows. Keeping values JSON-safe at
        creation time avoids late serialization failures caused by NaN/Inf,
        tensors, numpy objects, Interval objects, or set/tuple containers.
        """
        if value is None:
            return None
        if isinstance(value, bool | int | str):
            return value
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, np.generic):
            return Lightcurve._consensus_make_json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [
                Lightcurve._consensus_make_json_safe(v)
                for v in value.tolist()
            ]
        if torch.is_tensor(value):
            if value.numel() == 1:
                return Lightcurve._consensus_make_json_safe(value.item())
            return Lightcurve._consensus_make_json_safe(
                value.detach().cpu().tolist()
            )
        if isinstance(value, Interval):
            lower = None if value.lower_bound is None else value.lower_bound
            upper = None if value.upper_bound is None else value.upper_bound
            return {
                "lower": Lightcurve._consensus_make_json_safe(lower),
                "upper": Lightcurve._consensus_make_json_safe(upper),
            }
        if isinstance(value, dict):
            return {
                str(key): Lightcurve._consensus_make_json_safe(val)
                for key, val in value.items()
            }
        if isinstance(value, list | tuple | set):
            return [
                Lightcurve._consensus_make_json_safe(v)
                for v in list(value)
            ]
        return str(value)

    @staticmethod
    def _consensus_normalize_rejection_reasons(reasons):
        """Normalize rejection reasons into an ordered, de-duplicated list."""
        if reasons is None:
            return []
        if isinstance(reasons, str):
            items = [reasons]
        elif isinstance(reasons, list | tuple | set):
            items = list(reasons)
        else:
            items = [str(reasons)]
        normalized = []
        for reason in items:
            reason_str = str(reason).strip()
            if reason_str and reason_str not in normalized:
                normalized.append(reason_str)
        return normalized

    @staticmethod
    def _consensus_set_gp_validation_status(
        record,
        status,
        *,
        reason=None,
    ):
        """Set a validated GP-validation status on a band record.

        Parameters
        ----------
        record : dict
            Per-band consensus diagnostic record.
        status : str
            GP-validation status to store.  Must be one of the values defined
            by the ``_CONSENSUS_GP_VALIDATION_STATUS_*`` constants (currently
            ``_CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES``).
        reason : str or None, optional
            Optional GP-validation reason string stored in
            ``record["gp_validation_reason"]``. If ``None``, existing reason
            values are preserved and no new key is created. This argument is
            keyword-only.

        Raises
        ------
        ValueError
            If ``status`` is not an allowed GP-validation status.
        """
        if status not in _CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES:
            raise ValueError(
                f"Invalid gp_validation_status {status!r}. "
                "Allowed values are: "
                f"{_CONSENSUS_ALLOWED_GP_VALIDATION_STATUSES_SORTED}"
            )
        record["gp_validation_status"] = status
        if reason is not None:
            record["gp_validation_reason"] = reason
        elif status in _CONSENSUS_GP_VALIDATION_STATUSES_CLEAR_REASON:
            # Clear any stale reason that may have been set by an earlier
            # status transition so these terminal/clean statuses never carry
            # leftover failure/rejection/skip reason strings.
            record["gp_validation_reason"] = None

    @staticmethod
    def _consensus_set_band_status(
        record,
        status,
        rejection_reasons=None,
    ):
        """Set a validated band-level consensus status and rejection payload.

        The helper is the canonical state transition surface for
        ``record["status"]``, ``record["rejection_reasons"]``, and
        ``record["rejection_reason"]``.

        Invariants enforced by this helper:

        - ``status`` must be in ``_CONSENSUS_ALLOWED_BAND_STATUSES``.
        - ``status == "accepted"``:
          ``rejection_reasons`` must be empty and ``rejection_reason`` is
          forced to ``None``.
        - ``status == "rejected"``:
          ``rejection_reasons`` must be non-empty.
        - ``status == "pending"``:
          no rejection reasons are allowed.
        - ``rejection_reasons`` are normalized to ``list[str]`` and
          de-duplicated while preserving input order.
        - every rejection reason must satisfy
          :meth:`_consensus_is_allowed_rejection_reason`.
        - ``rejection_reason`` is always synchronized to the first entry in
          ``rejection_reasons`` (or ``None`` when empty), so stale values are
          never retained.

        Parameters
        ----------
        record : dict
            Per-band consensus diagnostics record.
        status : str
            New band status.
        rejection_reasons : sequence[str] or str or None, optional
            Rejection reasons to attach for rejected status.

        Raises
        ------
        ValueError
            If status is unknown, any reason is invalid, or the status/reason
            combination violates the invariants listed above.
        """
        if status not in _CONSENSUS_ALLOWED_BAND_STATUSES:
            raise ValueError(
                f"Invalid band status {status!r}. Allowed values are: "
                f"{_CONSENSUS_ALLOWED_BAND_STATUSES_SORTED}"
            )

        normalized_reasons = Lightcurve._consensus_normalize_rejection_reasons(
            rejection_reasons
        )
        for reason in normalized_reasons:
            if not Lightcurve._consensus_is_allowed_rejection_reason(reason):
                raise ValueError(
                    f"Unknown rejection reason {reason!r}. Expected one of "
                    f"{_CONSENSUS_ALLOWED_REJECTION_REASONS_SORTED} or a known "
                    "sampling-metrics reason prefix."
                )

        if status == _CONSENSUS_BAND_STATUS_REJECTED:
            if not normalized_reasons:
                raise ValueError(
                    "Band status 'rejected' requires at least one "
                    "rejection reason."
                )
        else:
            if normalized_reasons:
                raise ValueError(
                    f"Band status {status!r} cannot carry rejection reasons "
                    f"{normalized_reasons!r}."
                )

        record["status"] = status
        record["rejection_reasons"] = normalized_reasons
        record["rejection_reason"] = (
            normalized_reasons[0] if normalized_reasons else None
        )

    @staticmethod
    def _consensus_set_acf_comparison_status(
        record,
        status,
        *,
        period_ratio=None,
        harmonic_order=None,
    ):
        """Set validated ACF-vs-LS comparison status and metadata.

        This helper is the canonical state transition surface for
        ``acf_comparison_status``, ``acf_period_ratio``, and
        ``acf_harmonic_order``.

        State-machine semantics:

        - ``status is None``: no ACF comparison metadata is allowed.
        - ``status == "agreement"``: ``period_ratio`` is required;
          ``harmonic_order`` is optional.
        - ``status == "harmonic"``: both ``period_ratio`` and
          ``harmonic_order`` are required.
        - ``status == "disagreement"``: ``period_ratio`` is required;
          ``harmonic_order`` is optional.
        - ``status == "unavailable"``: no comparison metadata is allowed.

        For statuses where metadata is disallowed, stale values are always
        cleared to ``None``.

        Parameters
        ----------
        record : dict
            Per-band consensus diagnostics record.
        status : str or None
            ACF comparison status.
        period_ratio : float, optional
            Frequency-ratio style comparison statistic.
        harmonic_order : int, optional
            Harmonic order metadata when available.

        Raises
        ------
        ValueError
            If status is unknown or metadata does not satisfy allowed
            combinations.
        """
        if (
            status is not None
            and status not in _CONSENSUS_ALLOWED_ACF_COMPARISON_STATUSES
        ):
            raise ValueError(
                f"Invalid acf_comparison_status {status!r}. Allowed values are: "
                f"{_CONSENSUS_ALLOWED_ACF_COMPARISON_STATUSES_SORTED} or None."
            )

        ratio_val = None
        if period_ratio is not None:
            ratio_val = float(period_ratio)
            if not (np.isfinite(ratio_val) and ratio_val > 0):
                raise ValueError(
                    "acf_period_ratio must be finite and strictly positive "
                    "when provided."
                )

        order_val = None
        if harmonic_order is not None:
            try:
                order_val = int(harmonic_order)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "acf_harmonic_order must be an integer when provided."
                ) from exc
            if order_val <= 0:
                raise ValueError(
                    "acf_harmonic_order must be a positive integer when "
                    "provided."
                )

        if status is None:
            if ratio_val is not None or order_val is not None:
                raise ValueError(
                    "acf_comparison_status=None does not permit "
                    "acf_period_ratio/acf_harmonic_order metadata."
                )
            ratio_val = None
            order_val = None
        elif status in (_ACF_STATUS_AGREEMENT, _ACF_STATUS_DISAGREEMENT):
            if ratio_val is None:
                raise ValueError(
                    f"acf_comparison_status={status!r} requires "
                    "'acf_period_ratio'."
                )
        elif status == _ACF_STATUS_HARMONIC:
            if ratio_val is None:
                raise ValueError(
                    "acf_comparison_status='harmonic' requires "
                    "'acf_period_ratio'."
                )
            if order_val is None:
                raise ValueError(
                    "acf_comparison_status='harmonic' requires "
                    "'acf_harmonic_order'."
                )
        elif status == _ACF_STATUS_UNAVAILABLE:
            if ratio_val is not None or order_val is not None:
                raise ValueError(
                    "acf_comparison_status='unavailable' does not permit "
                    "acf_period_ratio/acf_harmonic_order metadata."
                )
            ratio_val = None
            order_val = None

        record["acf_comparison_status"] = status
        record["acf_period_ratio"] = ratio_val
        record["acf_harmonic_order"] = order_val

    def _consensus_initialize_band_record(
        self,
        band_label,
        *,
        metrics=None,
        gp_validation_requested=False,
    ):
        """Create a canonical per-band consensus diagnostic record.

        A stable schema is required so every downstream consumer can assume all
        keys exist for every band, including early-return failure paths.
        Initial ``gp_validation_status`` is always ``"not_requested"`` and is
        changed later only when the GP-validation stage runs.
        """
        # "not_requested" means GP validation was not invoked for this record.
        # Records are upgraded to "skipped"/"failed"/"rejected"/"success" later
        # only when the GP-validation stage is actually executed.
        record = _consensus_schema_default_record(_CONSENSUS_BAND_SCHEMA_FIELDS)
        record["band"] = str(band_label)
        record["metrics"] = self._consensus_make_json_safe(metrics)
        self._consensus_set_gp_validation_status(
            record, _CONSENSUS_GP_VALIDATION_STATUS_NOT_REQUESTED
        )
        return record

    @staticmethod
    def _consensus_is_allowed_rejection_reason(reason):
        """Return True if a rejection reason is canonical or known free-form."""
        if reason in _CONSENSUS_ALLOWED_REJECTION_REASONS:
            return True
        return any(
            reason.startswith(prefix)
            for prefix in _CONSENSUS_ALLOWED_REJECTION_REASON_PREFIXES
        )

    def _consensus_add_rejection_reasons(self, record, reasons):
        """Append rejection reasons to a band record without duplicates."""
        merged = self._consensus_normalize_rejection_reasons(
            record.get("rejection_reasons", [])
        )
        for reason in self._consensus_normalize_rejection_reasons(reasons):
            if reason not in merged:
                merged.append(reason)
        if merged:
            self._consensus_set_band_status(
                record,
                _CONSENSUS_BAND_STATUS_REJECTED,
                merged,
            )

    def _consensus_set_top_level_rejection_reasons(
        self,
        diagnostics,
        rejection_reasons,
    ):
        """Canonicalize top-level consensus rejection reasons bookkeeping.

        This helper centralizes canonicalization for
        ``diagnostics["rejection_reasons"]`` and enforces stable structure:

        - input is normalized to a plain ``dict`` (non-dict inputs become ``{}``)
        - each key must be a canonical rejection reason or a known
          sampling-metrics reason prefix variant
        - each value is normalized to a list of string band labels
        - duplicate band labels are removed while preserving order

        Parameters
        ----------
        diagnostics : dict
            Top-level consensus diagnostics dictionary to mutate.
        rejection_reasons : dict or object
            Candidate reason-to-band mapping. Expected shape is
            ``{reason_key: [band_label, ...], ...}``; non-dict inputs are
            normalized to an empty mapping.

        Raises
        ------
        ValueError
            If any rejection-reason key is not recognized.
        """
        canonical = {}
        if isinstance(rejection_reasons, dict):
            source = rejection_reasons
        else:
            source = {}

        for reason, bands in source.items():
            reason_str = str(reason).strip()
            if not reason_str:
                raise ValueError(
                    "Top-level rejection_reasons contains an empty reason key."
                )
            if not self._consensus_is_allowed_rejection_reason(reason_str):
                raise ValueError(
                    f"Unknown top-level rejection reason {reason_str!r}; expected "
                    f"one of {_CONSENSUS_ALLOWED_REJECTION_REASONS_SORTED} or a "
                    "known sampling-metrics reason prefix."
                )

            if bands is None:
                band_items = []
            elif isinstance(bands, list | tuple | set):
                band_items = list(bands)
            else:
                band_items = [bands]

            normalized_bands = []
            for band in band_items:
                band_str = str(band).strip()
                if band_str and band_str not in normalized_bands:
                    normalized_bands.append(band_str)
            canonical[reason_str] = normalized_bands

        diagnostics["rejection_reasons"] = canonical

    @staticmethod
    def _consensus_initialize_result_structure(*, fit_strategy="consensus"):
        """Create a canonical top-level consensus diagnostics schema.

        A stable schema is required so downstream consumers can reliably inspect
        consensus outputs regardless of whether the workflow succeeds, fails
        early, or fails after partial candidate construction.

        This consistency is important for:

        - JSON serialization/export pipelines (fixed key presence),
        - notebooks and plotting/reporting code (reduced key-guard logic),
        - regression tests (deterministic structure assertions),
        - future consensus-recovery/fallback workflows (portable diagnostics).
        """
        result = _consensus_schema_default_record(_CONSENSUS_TOP_LEVEL_SCHEMA_FIELDS)
        result["fit_strategy"] = fit_strategy
        return result

    def _consensus_build_rejection_summary(
        self,
        *,
        per_band_diagnostics,
        rejected_bands,
        rejection_reasons=None,
    ):
        """Build deterministic reason->bands rejection summary."""
        summary = {}
        per_band_diagnostics = per_band_diagnostics or {}
        rejection_reasons = rejection_reasons or {}
        for band in rejected_bands or []:
            reasons = []
            band_record = per_band_diagnostics.get(band, {})
            if isinstance(band_record, dict):
                reasons.extend(
                    self._consensus_normalize_rejection_reasons(
                        band_record.get("rejection_reasons")
                    )
                )
                reasons.extend(
                    self._consensus_normalize_rejection_reasons(
                        band_record.get("rejection_reason")
                    )
                )
            reasons.extend(
                self._consensus_normalize_rejection_reasons(
                    rejection_reasons.get(band)
                )
            )
            normalized = self._consensus_normalize_rejection_reasons(reasons)
            if not normalized:
                normalized = ["unspecified_rejection"]
            for reason in normalized:
                summary.setdefault(str(reason), set()).add(str(band))

        return {
            reason: sorted(bands)
            for reason, bands in sorted(summary.items(), key=lambda kv: kv[0])
        }

    def _consensus_finalize_result_structure(self, diagnostics, *, validate=True):
        """Normalize and finalize top-level consensus diagnostics."""
        canonical = self._consensus_initialize_result_structure(
            fit_strategy=(diagnostics or {}).get("fit_strategy", "consensus")
        )
        if diagnostics:
            canonical.update(dict(diagnostics))

        per_band = canonical.get("per_band_diagnostics") or {}
        if not isinstance(per_band, dict):
            per_band = {}
        canonical["per_band_diagnostics"] = per_band

        accepted = sorted({str(b) for b in (canonical.get("accepted_bands") or [])})
        rejected = sorted({str(b) for b in (canonical.get("rejected_bands") or [])})
        accepted_set = set(accepted)
        rejected_set = set(rejected)
        overlap = accepted_set & rejected_set
        if overlap:
            # Rejections take precedence: if a band appears in both lists due to
            # upstream partial bookkeeping, keep it only in rejected_bands.
            accepted_set -= overlap
            accepted = sorted(accepted_set)
        canonical["accepted_bands"] = accepted
        canonical["rejected_bands"] = rejected

        # Counts are anchored to mutually exclusive accepted/rejected sets.
        canonical["n_accepted_bands"] = len(accepted)
        canonical["n_rejected_bands"] = len(rejected)
        canonical["n_total_bands"] = (
            canonical["n_accepted_bands"] + canonical["n_rejected_bands"]
        )

        rejection_reasons_input = canonical.get("rejection_reasons")
        rejection_summary_input = canonical.get("rejection_summary")
        canonicalized_rejection_reasons = {}

        # Canonicalize top-level rejection bookkeeping only after accepted/
        # rejected membership has been normalized so reason->band mappings are
        # interpreted against stable, finalized band labels.
        if rejection_reasons_input:
            canonical_source = {}
            self._consensus_set_top_level_rejection_reasons(
                canonical_source, rejection_reasons_input
            )
            canonicalized_rejection_reasons = canonical_source["rejection_reasons"]

        canonicalized_rejection_summary = {}
        if rejection_summary_input:
            deprecated_source = {}
            self._consensus_set_top_level_rejection_reasons(
                deprecated_source, rejection_summary_input
            )
            canonicalized_rejection_summary = deprecated_source["rejection_reasons"]

        if canonicalized_rejection_reasons and canonicalized_rejection_summary:
            if canonicalized_rejection_reasons != canonicalized_rejection_summary:
                raise ValueError(
                    "Top-level 'rejection_reasons' and deprecated "
                    "'rejection_summary' disagree after canonicalization."
                )
        elif canonicalized_rejection_summary:
            canonicalized_rejection_reasons = canonicalized_rejection_summary

        canonical["rejection_reasons"] = canonicalized_rejection_reasons
        # 'rejection_summary' is a deprecated compatibility alias for the
        # canonical top-level 'rejection_reasons' mapping.
        canonical["rejection_summary"] = canonicalized_rejection_reasons.copy()

        # Keep canonical frequency-first keys and synchronize legacy aliases.
        consensus_frequency = canonical.get("consensus_frequency")
        if consensus_frequency is None:
            consensus_frequency = canonical.get("final_consensus_frequency")
        if consensus_frequency is not None:
            consensus_frequency = float(consensus_frequency)
        canonical["consensus_frequency"] = consensus_frequency
        canonical["final_consensus_frequency"] = consensus_frequency

        consensus_period = canonical.get("consensus_period")
        if consensus_period is None and consensus_frequency is not None:
            consensus_period = float(1.0 / consensus_frequency)
        canonical["consensus_period"] = consensus_period
        canonical["final_consensus_period"] = consensus_period

        frequency_scatter = canonical.get("consensus_frequency_scatter")
        if frequency_scatter is None:
            frequency_scatter = canonical.get("mad_frequency_scatter")
        canonical["consensus_frequency_scatter"] = frequency_scatter
        canonical["mad_frequency_scatter"] = frequency_scatter

        frequency_width = canonical.get("consensus_frequency_width")
        if frequency_width is None:
            frequency_width = canonical.get("robust_frequency_width")
        canonical["consensus_frequency_width"] = frequency_width
        canonical["robust_frequency_width"] = frequency_width

        # Ensure bool fields are stable booleans.
        for key in (
            "consensus_success",
            "use_acf_validation",
            "use_gp_validation",
            "gp_validation_requested",
            "gp_validation_performed",
        ):
            canonical[key] = bool(canonical.get(key))

        result = self._consensus_make_json_safe(canonical)
        if validate:
            self._consensus_validate_result_structure(result)
        return result

    def _consensus_validate_result_structure(self, diagnostics):
        """Validate internal consistency of a finalized consensus diagnostics dict.

        This method enforces structural invariants that must hold for every
        finalized consensus diagnostics structure.  It is intended to catch
        corruption introduced by bugs in the consensus machinery before the
        structure is attached to ``self.consensus_diagnostics``.

        Validation only — this method never silently repairs a corrupted
        structure.  Raise ``RuntimeError`` or ``ValueError`` with an
        informative message on any violation.

        Parameters
        ----------
        diagnostics : dict
            A finalized consensus diagnostics dict, typically the return value
            of :meth:`_consensus_finalize_result_structure`.

        Raises
        ------
        RuntimeError
            If a required key is absent, count fields are inconsistent, or
            ``consensus_success`` semantics are violated.
        ValueError
            If ``accepted_bands``/``rejected_bands`` overlap, the
            top-level rejection mappings disagree, an unknown GP status is
            found, or ``rejection_reasons`` entries are not lists.
        """
        if not isinstance(diagnostics, dict):
            raise RuntimeError(
                "Consensus diagnostics must be a dict; "
                f"got {type(diagnostics).__name__!r}."
            )

        # --- 1. Required top-level keys must all be present ---
        missing_keys = _CONSENSUS_REQUIRED_RESULT_KEYS - diagnostics.keys()
        if missing_keys:
            raise RuntimeError(
                "Consensus diagnostics is missing required top-level key(s): "
                + ", ".join(f"'{k}'" for k in sorted(missing_keys))
                + "."
            )

        top_schema_fields = _CONSENSUS_TOP_LEVEL_SCHEMA["fields"]
        accepted_bands = diagnostics["accepted_bands"]
        rejected_bands = diagnostics["rejected_bands"]

        # --- 2. Top-level container fields must match canonical schema ---
        for field_name in ("accepted_bands", "rejected_bands"):
            expected_container = top_schema_fields[field_name]["container_type"]
            if expected_container == "list":
                expected_type = list
            elif expected_container == "dict":
                expected_type = dict
            else:
                continue
            field_value = diagnostics[field_name]
            if not isinstance(field_value, expected_type):
                raise RuntimeError(
                    f"'{field_name}' must be a {expected_container}; got "
                    f"{type(field_value).__name__!r}."
                )

        # --- 3. No band may appear in both lists ---
        accepted_set = {str(b) for b in accepted_bands}
        rejected_set = {str(b) for b in rejected_bands}
        overlap = accepted_set & rejected_set
        if overlap:
            raise ValueError(
                "Band(s) appear in both 'accepted_bands' and 'rejected_bands': "
                + ", ".join(f"'{b}'" for b in sorted(overlap))
                + "."
            )

        # --- 4. Count fields must be consistent with the band lists ---
        n_accepted = diagnostics["n_accepted_bands"]
        n_rejected = diagnostics["n_rejected_bands"]
        n_total = diagnostics["n_total_bands"]
        if n_accepted != len(accepted_set):
            raise RuntimeError(
                f"'n_accepted_bands' ({n_accepted}) does not match "
                f"len(accepted_bands) ({len(accepted_set)})."
            )
        if n_rejected != len(rejected_set):
            raise RuntimeError(
                f"'n_rejected_bands' ({n_rejected}) does not match "
                f"len(rejected_bands) ({len(rejected_set)})."
            )
        if n_total != n_accepted + n_rejected:
            raise RuntimeError(
                f"'n_total_bands' ({n_total}) != "
                f"n_accepted_bands ({n_accepted}) + "
                f"n_rejected_bands ({n_rejected})."
            )

        # --- 5. rejection_reasons is canonical; rejection_summary is alias ---
        canonical_rejection_source = {}
        self._consensus_set_top_level_rejection_reasons(
            canonical_rejection_source, diagnostics["rejection_reasons"]
        )
        rejection_reasons = canonical_rejection_source["rejection_reasons"]

        for reason, bands in rejection_reasons.items():
            if not isinstance(bands, list):
                raise ValueError(
                    f"rejection_reasons[{reason!r}] must be a list; "
                    f"got {type(bands).__name__!r}."
                )
            band_labels = [str(b) for b in bands]
            duplicate_bands = sorted(
                {b for b in band_labels if band_labels.count(b) > 1}
            )
            if duplicate_bands:
                raise ValueError(
                    f"rejection_reasons[{reason!r}] contains duplicate "
                    "band label(s): "
                    + ", ".join(f"{b!r}" for b in duplicate_bands)
                    + "."
                )
            for band in bands:
                band_str = str(band)
                if band_str not in rejected_set:
                    raise ValueError(
                        f"Band {band_str!r} in rejection_reasons[{reason!r}] "
                        "is not present in 'rejected_bands'."
                    )
                if band_str in accepted_set:
                    raise ValueError(
                        f"Accepted band {band_str!r} appears in "
                        f"rejection_reasons[{reason!r}]."
                    )

        alias_rejection_source = {}
        self._consensus_set_top_level_rejection_reasons(
            alias_rejection_source, diagnostics["rejection_summary"]
        )
        rejection_summary = alias_rejection_source["rejection_reasons"]
        if rejection_summary != rejection_reasons:
            raise ValueError(
                "Top-level 'rejection_summary' must match canonical "
                "'rejection_reasons'."
            )

        # --- 6. per_band_diagnostics must be a dict ---
        per_band = diagnostics["per_band_diagnostics"]
        expected_per_band_container = top_schema_fields["per_band_diagnostics"][
            "container_type"
        ]
        if not isinstance(per_band, dict):
            raise RuntimeError(
                f"'per_band_diagnostics' must be a "
                f"{expected_per_band_container}; "
                f"got {type(per_band).__name__!r}."
            )
        referenced_bands = accepted_set | rejected_set
        per_band_labels = {str(label) for label in per_band}
        missing_per_band = sorted(referenced_bands - per_band_labels)
        if missing_per_band:
            raise RuntimeError(
                "per_band_diagnostics is missing entries for referenced band(s): "
                + ", ".join(f"'{b}'" for b in missing_per_band)
                + "."
            )
        extra_per_band = sorted(per_band_labels - referenced_bands)
        if extra_per_band:
            raise RuntimeError(
                "per_band_diagnostics contains entries for unknown band(s): "
                + ", ".join(f"'{b}'" for b in extra_per_band)
                + "."
            )

        # --- 7-9. Per-band record invariants ---
        band_schema_fields = _CONSENSUS_BAND_SCHEMA["fields"]
        for band_label, record in per_band.items():
            _b = str(band_label)
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"per_band_diagnostics[{_b!r}] must be a dict; "
                    f"got {type(record).__name__!r}."
                )
            missing_band = _CONSENSUS_REQUIRED_BAND_KEYS - record.keys()
            if missing_band:
                raise RuntimeError(
                    f"Band {_b!r} is missing required key(s): "
                    + ", ".join(f"'{k}'" for k in sorted(missing_band))
                    + "."
                )

            band_status = record["status"]
            band_status_allowed = band_schema_fields["status"]["allowed_values"]
            if band_status not in band_status_allowed:
                raise ValueError(
                    f"Band {_b!r} has unknown 'status' {band_status!r}; "
                    "expected one of "
                    f"{band_status_allowed}."
                )

            # 8. gp_validation_status must be a known value ---
            gp_status = record["gp_validation_status"]
            gp_status_allowed = band_schema_fields["gp_validation_status"][
                "allowed_values"
            ]
            if gp_status not in gp_status_allowed:
                raise ValueError(
                    f"Band {_b!r} has unknown 'gp_validation_status' "
                    f"{gp_status!r}; expected one of "
                    f"{gp_status_allowed}."
                )

            acf_status = record.get("acf_comparison_status")
            acf_nullable = band_schema_fields["acf_comparison_status"]["nullable"]
            acf_allowed = band_schema_fields["acf_comparison_status"]["allowed_values"]
            if (
                (acf_status is not None or not acf_nullable)
                and acf_status not in acf_allowed
            ):
                raise ValueError(
                    f"Band {_b!r} has unknown 'acf_comparison_status' "
                    f"{acf_status!r}; expected one of "
                    f"{acf_allowed} or "
                    "None."
                )
            acf_ratio = record.get("acf_period_ratio")
            if acf_ratio is not None:
                try:
                    acf_ratio = float(acf_ratio)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Band {_b!r}: 'acf_period_ratio' must be a float or "
                        f"None; got {record.get('acf_period_ratio')!r}."
                    ) from exc
                if not (np.isfinite(acf_ratio) and acf_ratio > 0):
                    raise ValueError(
                        f"Band {_b!r}: 'acf_period_ratio' must be finite and "
                        "strictly positive when present."
                    )
            acf_order = record.get("acf_harmonic_order")
            if acf_order is not None:
                try:
                    acf_order = int(acf_order)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Band {_b!r}: 'acf_harmonic_order' must be an integer "
                        f"or None; got {record.get('acf_harmonic_order')!r}."
                    ) from exc
                if acf_order <= 0:
                    raise ValueError(
                        f"Band {_b!r}: 'acf_harmonic_order' must be positive "
                        "when present."
                    )

            # 9. rejection_reasons must be a list ---
            rr = record["rejection_reasons"]
            if not isinstance(rr, list):
                raise ValueError(
                    f"Band {_b!r}: 'rejection_reasons' must be a list; "
                    f"got {type(rr).__name__!r}."
                )
            normalized_rr = []
            for reason in rr:
                if not isinstance(reason, str):
                    raise ValueError(
                        f"Band {_b!r}: rejection reason {reason!r} must be a "
                        "string."
                    )
                reason_str = reason.strip()
                if not reason_str:
                    raise ValueError(
                        f"Band {_b!r}: rejection reason {reason!r} is empty."
                    )
                if not self._consensus_is_allowed_rejection_reason(reason_str):
                    raise ValueError(
                        f"Band {_b!r} has unknown rejection reason "
                        f"{reason_str!r}; expected one of "
                        f"{_CONSENSUS_ALLOWED_REJECTION_REASONS_SORTED} "
                        "or a known sampling-metrics reason prefix."
                    )
                normalized_rr.append(reason_str)

            rejection_reason = record.get("rejection_reason")
            if rejection_reason is not None:
                if not isinstance(rejection_reason, str):
                    raise ValueError(
                        f"Band {_b!r}: 'rejection_reason' must be a string or "
                        f"None; got {type(rejection_reason).__name__!r}."
                    )
                rejection_reason = rejection_reason.strip()
                if not rejection_reason:
                    raise ValueError(
                        f"Band {_b!r}: 'rejection_reason' must not be empty."
                    )
                if not self._consensus_is_allowed_rejection_reason(rejection_reason):
                    raise ValueError(
                        f"Band {_b!r} has unknown 'rejection_reason' "
                        f"{rejection_reason!r}; expected one of "
                        f"{_CONSENSUS_ALLOWED_REJECTION_REASONS_SORTED} "
                        "or a known sampling-metrics reason prefix."
                    )
            if normalized_rr and rejection_reason != normalized_rr[0]:
                raise ValueError(
                    f"Band {_b!r} has 'rejection_reason'={rejection_reason!r} "
                    "but first entry in 'rejection_reasons' is "
                    f"{normalized_rr[0]!r}."
                )
            if not normalized_rr and rejection_reason is not None:
                raise ValueError(
                    f"Band {_b!r} has 'rejection_reason'={rejection_reason!r} "
                    "but no rejection reasons."
                )

            if (
                band_status == _CONSENSUS_BAND_STATUS_ACCEPTED
                and (normalized_rr or rejection_reason is not None)
            ):
                raise ValueError(
                    f"Band {_b!r} has status='accepted' but carries rejection "
                    f"payload (rejection_reasons={normalized_rr!r}, "
                    f"rejection_reason={rejection_reason!r})."
                )
            if (
                band_status == _CONSENSUS_BAND_STATUS_REJECTED
                and not normalized_rr
            ):
                raise ValueError(
                    f"Band {_b!r} has status='rejected' but no rejection reasons."
                )
            if (
                band_status == _CONSENSUS_BAND_STATUS_PENDING
                and (normalized_rr or rejection_reason is not None)
            ):
                raise ValueError(
                    f"Band {_b!r} has status='pending' but carries rejection "
                    f"payload (rejection_reasons={normalized_rr!r}, "
                    f"rejection_reason={rejection_reason!r})."
                )
            if (
                _b in accepted_set
                and band_status == _CONSENSUS_BAND_STATUS_REJECTED
            ):
                raise ValueError(
                    f"Band {_b!r} is in 'accepted_bands' but has status "
                    "'rejected'."
                )
            if (
                _b in rejected_set
                and band_status == _CONSENSUS_BAND_STATUS_ACCEPTED
            ):
                raise ValueError(
                    f"Band {_b!r} is in 'rejected_bands' but has status "
                    "'accepted'."
                )
            if (
                _b in accepted_set
                and band_status == _CONSENSUS_BAND_STATUS_PENDING
            ):
                raise ValueError(
                    f"Band {_b!r} is in 'accepted_bands' but has status "
                    "'pending'."
                )
            if (
                _b in rejected_set
                and band_status == _CONSENSUS_BAND_STATUS_PENDING
            ):
                raise ValueError(
                    f"Band {_b!r} is in 'rejected_bands' but has status "
                    "'pending'."
                )

            if acf_status is None:
                if acf_ratio is not None or acf_order is not None:
                    raise ValueError(
                        f"Band {_b!r} has acf_comparison_status=None but "
                        "acf_period_ratio/acf_harmonic_order is set."
                    )
            elif acf_status in (_ACF_STATUS_AGREEMENT, _ACF_STATUS_DISAGREEMENT):
                if acf_ratio is None:
                    raise ValueError(
                        f"Band {_b!r} has acf_comparison_status={acf_status!r} "
                        "but missing required acf_period_ratio."
                    )
            elif acf_status == _ACF_STATUS_HARMONIC:
                if acf_ratio is None:
                    raise ValueError(
                        f"Band {_b!r} has acf_comparison_status='harmonic' but "
                        "missing required acf_period_ratio."
                    )
                if acf_order is None:
                    raise ValueError(
                        f"Band {_b!r} has acf_comparison_status='harmonic' but "
                        "missing required acf_harmonic_order."
                    )
            elif acf_status == _ACF_STATUS_UNAVAILABLE:
                if acf_ratio is not None or acf_order is not None:
                    raise ValueError(
                        f"Band {_b!r} has acf_comparison_status='unavailable' "
                        "but carries acf_period_ratio/acf_harmonic_order data."
                    )

            # 10. gp_validation_reason invariant per status ---
            gp_reason = record.get("gp_validation_reason")
            gp_reason_allowed = band_schema_fields["gp_validation_reason"][
                "allowed_values"
            ]
            if (
                gp_reason is not None
                and gp_reason not in gp_reason_allowed
            ):
                raise ValueError(
                    f"Band {_b!r} has unknown 'gp_validation_reason' "
                    f"{gp_reason!r}; expected one of {gp_reason_allowed} "
                    "or None."
                )
            allowed_reasons = _CONSENSUS_ALLOWED_GP_VALIDATION_REASONS_BY_STATUS.get(
                gp_status
            )
            if allowed_reasons is not None and gp_reason not in allowed_reasons:
                allowed_sorted = (
                    _CONSENSUS_ALLOWED_GP_VALIDATION_REASONS_BY_STATUS_SORTED[
                        gp_status
                    ]
                )
                raise ValueError(
                    f"Band {_b!r} has gp_validation_status={gp_status!r} "
                    f"but gp_validation_reason={gp_reason!r} is not allowed "
                    f"for this status (allowed: {allowed_sorted} and/or None)."
                )

        # --- 10. consensus_success semantics ---
        consensus_success = diagnostics["consensus_success"]
        if consensus_success:
            consensus_frequency = diagnostics.get("consensus_frequency")
            fit_strategy = str(diagnostics.get("fit_strategy") or "")
            consensus_frequencies = diagnostics.get("consensus_frequencies")
            consensus_periods = diagnostics.get("consensus_periods")
            n_accepted_bands = diagnostics["n_accepted_bands"]
            trusted_candidate_count = diagnostics.get("trusted_candidate_count")
            has_scalar_consensus_frequency = consensus_frequency is not None
            has_vector_consensus_frequencies = False
            has_vector_consensus_periods = False
            consensus_frequencies_arr = np.asarray([], dtype=float)
            consensus_periods_arr = np.asarray([], dtype=float)
            if consensus_frequencies is not None:
                try:
                    consensus_frequencies_arr = np.asarray(
                        consensus_frequencies, dtype=float
                    ).ravel()
                except Exception:
                    consensus_frequencies_arr = np.asarray([], dtype=float)
                has_vector_consensus_frequencies = bool(
                    consensus_frequencies_arr.size > 0
                    and np.all(
                        np.isfinite(consensus_frequencies_arr)
                        & (consensus_frequencies_arr > 0)
                    )
                )
            if consensus_periods is not None:
                try:
                    consensus_periods_arr = np.asarray(consensus_periods, dtype=float).ravel()
                except Exception:
                    consensus_periods_arr = np.asarray([], dtype=float)
                has_vector_consensus_periods = bool(
                    consensus_periods_arr.size > 0
                    and np.all(np.isfinite(consensus_periods_arr) & (consensus_periods_arr > 0))
                )
            if fit_strategy == "consensus_multicomp":
                if not has_vector_consensus_frequencies:
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'consensus_frequencies' is missing or invalid."
                    )
                if not has_vector_consensus_periods:
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'consensus_periods' is missing or invalid."
                    )
                if consensus_frequencies_arr.size != consensus_periods_arr.size:
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'consensus_frequencies' and 'consensus_periods' have "
                        "different lengths."
                    )
                fitted_frequencies_arr = np.asarray(
                    diagnostics.get("fitted_mixture_frequencies", []), dtype=float
                ).ravel()
                fitted_periods_arr = np.asarray(
                    diagnostics.get("fitted_mixture_periods", []), dtype=float
                ).ravel()
                fitted_scales_arr = np.asarray(
                    diagnostics.get("fitted_mixture_scales", []), dtype=float
                ).ravel()
                fitted_period_widths_arr = np.asarray(
                    diagnostics.get("fitted_mixture_period_widths", []), dtype=float
                ).ravel()
                if fitted_frequencies_arr.size == 0:
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'fitted_mixture_frequencies' is missing."
                    )
                if not np.all(
                    np.isfinite(fitted_frequencies_arr) & (fitted_frequencies_arr > 0)
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'fitted_mixture_frequencies' is invalid."
                    )
                if not np.all(
                    np.isfinite(fitted_periods_arr) & (fitted_periods_arr > 0)
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'fitted_mixture_periods' is missing or invalid."
                    )
                if not np.all(
                    np.isfinite(fitted_scales_arr) & (fitted_scales_arr > 0)
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'fitted_mixture_scales' is missing or invalid."
                    )
                if not np.all(np.isfinite(fitted_period_widths_arr)):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'fitted_mixture_period_widths' is missing or invalid."
                    )
                fitted_vector_lengths = [
                    fitted_frequencies_arr.size,
                    fitted_periods_arr.size,
                    fitted_scales_arr.size,
                    fitted_period_widths_arr.size,
                    np.asarray(
                        diagnostics.get(
                            "fitted_frequency_shift_from_initialization", []
                        ),
                        dtype=float,
                    ).ravel().size,
                    np.asarray(
                        diagnostics.get(
                            "fitted_period_shift_from_initialization", []
                        ),
                        dtype=float,
                    ).ravel().size,
                    np.asarray(
                        diagnostics.get(
                            "fitted_fractional_frequency_shift_from_initialization", []
                        ),
                        dtype=float,
                    ).ravel().size,
                    np.asarray(
                        diagnostics.get(
                            "fitted_fractional_period_shift_from_initialization", []
                        ),
                        dtype=float,
                    ).ravel().size,
                ]
                if any(
                    length != fitted_frequencies_arr.size
                    for length in fitted_vector_lengths
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "fitted diagnostics vectors do not share one-to-one "
                        "component lengths."
                    )
                initialized_periods_arr = np.asarray(
                    diagnostics.get("initialized_mixture_periods", []), dtype=float
                ).ravel()
                initialized_frequencies_arr = np.asarray(
                    diagnostics.get("initialized_mixture_means", []), dtype=float
                ).ravel()
                if not np.all(
                    np.isfinite(initialized_periods_arr)
                    & (initialized_periods_arr > 0)
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'initialized_mixture_periods' is missing or invalid."
                    )
                if not np.all(
                    np.isfinite(initialized_frequencies_arr)
                    & (initialized_frequencies_arr > 0)
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'initialized_mixture_means' is missing or invalid."
                    )
                if (
                    initialized_periods_arr.size != fitted_frequencies_arr.size
                    or initialized_frequencies_arr.size != fitted_frequencies_arr.size
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "initialized diagnostics vectors do not match fitted "
                        "component lengths."
                    )
                drift_warning_fraction = diagnostics.get("drift_warning_fraction")
                if drift_warning_fraction is None:
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'drift_warning_fraction' is missing or invalid."
                    )
                drift_warning_fraction = float(drift_warning_fraction)
                if not (
                    np.isfinite(drift_warning_fraction)
                    and drift_warning_fraction >= 0
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'drift_warning_fraction' is missing or invalid."
                    )
                abs_period_shift = np.abs(
                    np.asarray(
                        diagnostics.get(
                            "fitted_fractional_period_shift_from_initialization", []
                        ),
                        dtype=float,
                    ).ravel()
                )
                abs_frequency_shift = np.abs(
                    np.asarray(
                        diagnostics.get(
                            "fitted_fractional_frequency_shift_from_initialization", []
                        ),
                        dtype=float,
                    ).ravel()
                )
                max_abs_period_shift = diagnostics.get(
                    "max_abs_fractional_period_shift_from_initialization"
                )
                max_abs_frequency_shift = diagnostics.get(
                    "max_abs_fractional_frequency_shift_from_initialization"
                )
                if not (
                    max_abs_period_shift is not None
                    and np.isfinite(float(max_abs_period_shift))
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'max_abs_fractional_period_shift_from_initialization' "
                        "is missing or invalid."
                    )
                if not (
                    max_abs_frequency_shift is not None
                    and np.isfinite(float(max_abs_frequency_shift))
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'max_abs_fractional_frequency_shift_from_initialization' "
                        "is missing or invalid."
                    )
                if not np.isclose(
                    float(max_abs_period_shift),
                    float(np.max(abs_period_shift)),
                    rtol=1e-8,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'max_abs_fractional_period_shift_from_initialization' "
                        "does not match fitted period drift diagnostics."
                    )
                if not np.isclose(
                    float(max_abs_frequency_shift),
                    float(np.max(abs_frequency_shift)),
                    rtol=1e-8,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'max_abs_fractional_frequency_shift_from_initialization' "
                        "does not match fitted frequency drift diagnostics."
                    )
                expected_period_flags = (
                    abs_period_shift >= drift_warning_fraction
                ).tolist()
                expected_frequency_flags = (
                    abs_frequency_shift >= drift_warning_fraction
                ).tolist()
                component_fit_drift_flags = diagnostics.get("component_fit_drift_flags")
                if not (
                    isinstance(component_fit_drift_flags, list)
                    and len(component_fit_drift_flags) == fitted_frequencies_arr.size
                    and all(
                        isinstance(flag, (bool, np.bool_))
                        for flag in component_fit_drift_flags
                    )
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'component_fit_drift_flags' is missing or invalid."
                    )
                observed_period_flags = [
                    bool(flag) for flag in component_fit_drift_flags
                ]
                if observed_period_flags != expected_period_flags:
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'component_fit_drift_flags' does not match fitted period "
                        "drift diagnostics."
                    )

                def _validate_component_index_list(key, expected_indices):
                    observed = diagnostics.get(key)
                    if not isinstance(observed, list):
                        raise RuntimeError(
                            "consensus_success is True for 'consensus_multicomp' but "
                            f"'{key}' is missing or invalid."
                        )
                    coerced = []
                    for value in observed:
                        if isinstance(value, (bool, np.bool_)):
                            raise RuntimeError(
                                "consensus_success is True for 'consensus_multicomp' "
                                f"but '{key}' contains invalid component indices."
                            )
                        if isinstance(value, (int, np.integer)):
                            idx = int(value)
                        else:
                            raise RuntimeError(
                                "consensus_success is True for 'consensus_multicomp' "
                                f"but '{key}' contains invalid component indices."
                            )
                        if idx < 0 or idx >= int(fitted_frequencies_arr.size):
                            raise RuntimeError(
                                "consensus_success is True for 'consensus_multicomp' "
                                f"but '{key}' contains out-of-range indices."
                            )
                        coerced.append(idx)
                    if coerced != list(expected_indices):
                        raise RuntimeError(
                            "consensus_success is True for 'consensus_multicomp' but "
                            f"'{key}' does not match fitted drift diagnostics."
                        )

                _validate_component_index_list(
                    "components_with_large_period_drift",
                    np.flatnonzero(expected_period_flags).tolist(),
                )
                _validate_component_index_list(
                    "components_with_large_frequency_drift",
                    np.flatnonzero(expected_frequency_flags).tolist(),
                )
                nearest_initialized_component_index = diagnostics.get(
                    "nearest_initialized_component_index"
                )
                nearest_initialized_component_fractional_period_distance = (
                    diagnostics.get(
                        "nearest_initialized_component_fractional_period_distance"
                    )
                )
                component_identity_preserved = diagnostics.get(
                    "component_identity_preserved"
                )
                all_component_identities_preserved = diagnostics.get(
                    "all_component_identities_preserved"
                )
                if not (
                    isinstance(nearest_initialized_component_index, list)
                    and len(nearest_initialized_component_index)
                    == fitted_frequencies_arr.size
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'nearest_initialized_component_index' is missing or invalid."
                    )
                coerced_nearest_idx = []
                for value in nearest_initialized_component_index:
                    if isinstance(value, (bool, np.bool_)):
                        raise RuntimeError(
                            "consensus_success is True for 'consensus_multicomp' but "
                            "'nearest_initialized_component_index' contains invalid "
                            "component indices."
                        )
                    if not isinstance(value, (int, np.integer)):
                        raise RuntimeError(
                            "consensus_success is True for 'consensus_multicomp' but "
                            "'nearest_initialized_component_index' contains invalid "
                            "component indices."
                        )
                    idx = int(value)
                    if idx < 0 or idx >= int(fitted_frequencies_arr.size):
                        raise RuntimeError(
                            "consensus_success is True for 'consensus_multicomp' but "
                            "'nearest_initialized_component_index' contains "
                            "out-of-range indices."
                        )
                    coerced_nearest_idx.append(idx)
                if not (
                    isinstance(
                        nearest_initialized_component_fractional_period_distance, list
                    )
                    and len(nearest_initialized_component_fractional_period_distance)
                    == fitted_frequencies_arr.size
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'nearest_initialized_component_fractional_period_distance' "
                        "is missing or invalid."
                    )
                try:
                    coerced_nearest_period_distance = np.asarray(
                        nearest_initialized_component_fractional_period_distance,
                        dtype=float,
                    ).ravel()
                except Exception as exc:
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'nearest_initialized_component_fractional_period_distance' "
                        "is missing or invalid."
                    ) from exc
                if not np.all(
                    np.isfinite(coerced_nearest_period_distance)
                    & (coerced_nearest_period_distance >= 0)
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'nearest_initialized_component_fractional_period_distance' "
                        "is missing or invalid."
                    )
                if not (
                    isinstance(component_identity_preserved, list)
                    and len(component_identity_preserved) == fitted_frequencies_arr.size
                    and all(
                        isinstance(flag, (bool, np.bool_))
                        for flag in component_identity_preserved
                    )
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'component_identity_preserved' is missing or invalid."
                    )
                coerced_identity_flags = [
                    bool(flag) for flag in component_identity_preserved
                ]
                expected_identity_flags = [
                    idx == component_idx
                    for component_idx, idx in enumerate(coerced_nearest_idx)
                ]
                if coerced_identity_flags != expected_identity_flags:
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'component_identity_preserved' does not match nearest "
                        "initialized-component diagnostics."
                    )
                if not isinstance(all_component_identities_preserved, (bool, np.bool_)):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'all_component_identities_preserved' is missing or invalid."
                    )
                if bool(all_component_identities_preserved) != bool(
                    all(coerced_identity_flags)
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'all_component_identities_preserved' does not match per-"
                        "component identity diagnostics."
                    )
                expected_possible_swaps = [
                    idx for idx, flag in enumerate(coerced_identity_flags) if not flag
                ]
                _validate_component_index_list(
                    "possible_component_swaps", expected_possible_swaps
                )
                expected_identity = (
                    self._consensus_compute_component_identity_diagnostics(
                        initialized_periods=initialized_periods_arr,
                        fitted_periods=fitted_periods_arr,
                        initialized_frequencies=initialized_frequencies_arr,
                        fitted_frequencies=fitted_frequencies_arr,
                    )
                )
                if (
                    expected_identity["nearest_initialized_component_index"]
                    != coerced_nearest_idx
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'nearest_initialized_component_index' does not match fitted "
                        "period identity diagnostics."
                    )
                expected_nearest_period_distance = np.asarray(
                    expected_identity[
                        "nearest_initialized_component_fractional_period_distance"
                    ],
                    dtype=float,
                ).ravel()
                if not np.allclose(
                    coerced_nearest_period_distance,
                    expected_nearest_period_distance,
                    rtol=1e-8,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "consensus_success is True for 'consensus_multicomp' but "
                        "'nearest_initialized_component_fractional_period_distance' "
                        "does not match fitted period identity diagnostics."
                    )
                nearest_initialized_component_fractional_frequency_distance = (
                    diagnostics.get(
                        "nearest_initialized_component_fractional_frequency_distance"
                    )
                )
                if (
                    nearest_initialized_component_fractional_frequency_distance
                    is not None
                ):
                    if not isinstance(
                        nearest_initialized_component_fractional_frequency_distance,
                        list,
                    ):
                        raise RuntimeError(
                            "consensus_success is True for 'consensus_multicomp' but "
                            "'nearest_initialized_component_fractional_frequency_"
                            "distance' is invalid."
                        )
                    if nearest_initialized_component_fractional_frequency_distance:
                        coerced_freq_distance = np.asarray(
                            nearest_initialized_component_fractional_frequency_distance,
                            dtype=float,
                        ).ravel()
                        if (
                            coerced_freq_distance.size != fitted_frequencies_arr.size
                            or not np.all(
                                np.isfinite(coerced_freq_distance)
                                & (coerced_freq_distance >= 0)
                            )
                        ):
                            raise RuntimeError(
                                "consensus_success is True for "
                                "'consensus_multicomp' but "
                                "'nearest_initialized_component_fractional_frequency_"
                                "distance' is invalid."
                            )
                        expected_nearest_frequency_distance = np.asarray(
                            expected_identity[
                                "nearest_initialized_component_fractional_frequency_distance"
                            ],
                            dtype=float,
                        ).ravel()
                        if not np.allclose(
                            coerced_freq_distance,
                            expected_nearest_frequency_distance,
                            rtol=1e-8,
                            atol=1e-12,
                        ):
                            raise RuntimeError(
                                "consensus_success is True for "
                                "'consensus_multicomp' but "
                                "'nearest_initialized_component_fractional_frequency_"
                                "distance' "
                                "does not match fitted frequency identity "
                                "diagnostics."
                            )
            elif not (
                has_scalar_consensus_frequency or has_vector_consensus_frequencies
            ):
                raise RuntimeError(
                    "consensus_success is True but neither a scalar "
                    "'consensus_frequency' nor valid positive "
                    "'consensus_frequencies' were provided."
                )
            if n_accepted_bands <= 0:
                raise RuntimeError(
                    "consensus_success is True but 'n_accepted_bands' is "
                    f"{n_accepted_bands} (must be > 0)."
                )
            if (
                trusted_candidate_count is None
                or int(trusted_candidate_count) <= 0
            ):
                raise RuntimeError(
                    "consensus_success is True but 'trusted_candidate_count' "
                    f"is {trusted_candidate_count!r} (must be > 0)."
                )

    def _consensus_debug_checkpoint(self, result_diagnostics, label):
        """Validate a partial diagnostics snapshot at a consensus checkpoint.

        Only active when :data:`_CONSENSUS_DEBUG_VALIDATE` is ``True``.
        Finalizes a shallow copy of *result_diagnostics* (without mutating
        the original) and passes the result to
        :meth:`_consensus_validate_result_structure`.  Any
        ``RuntimeError`` or ``ValueError`` raised by validation is
        re-raised as ``AssertionError`` with the checkpoint label prepended.

        This method is a no-op (fast path) when
        ``_CONSENSUS_DEBUG_VALIDATE`` is ``False``.

        Parameters
        ----------
        result_diagnostics : dict
            Intermediate consensus diagnostics dict to snapshot and check.
        label : str
            Short human-readable identifier for this checkpoint (used in
            the ``AssertionError`` message).

        Raises
        ------
        AssertionError
            If the finalized snapshot fails structural validation.
        """
        if not _CONSENSUS_DEBUG_VALIDATE:
            return
        try:
            self._consensus_finalize_result_structure(
                dict(result_diagnostics), validate=True
            )
        except (RuntimeError, ValueError) as exc:
            raise AssertionError(
                f"[consensus debug] Checkpoint {label!r} detected a "
                f"structural inconsistency: {exc}"
            ) from exc

    def _consensus_debug_checkpoint_from_candidate_state(
        self,
        *,
        label,
        controls,
        band_records,
        accepted_bands,
        rejected_bands,
        rejection_reasons,
        use_acf,
        use_gp_validation,
        gp_validation_requested,
        gp_validation_performed,
    ):
        """Build and validate a staged diagnostics snapshot from candidate state."""
        staged_diag = self._consensus_initialize_result_structure(
            fit_strategy="consensus"
        )
        _band_records = dict(band_records or {})
        _accepted = list(accepted_bands or [])
        _rejected = list(rejected_bands or [])
        _band_reasons = dict(rejection_reasons or {})
        staged_diag.update({
            "controls": self._consensus_make_json_safe(dict(controls or {})),
            "per_band_diagnostics": self._consensus_make_json_safe(_band_records),
            "accepted_bands": self._consensus_make_json_safe(_accepted),
            "rejected_bands": self._consensus_make_json_safe(_rejected),
            "rejection_reasons": self._consensus_build_rejection_summary(
                per_band_diagnostics=_band_records,
                rejected_bands=_rejected,
                rejection_reasons=_band_reasons,
            ),
            "use_acf_validation": bool(use_acf),
            "use_gp_validation": bool(use_gp_validation),
            "gp_validation_requested": bool(gp_validation_requested),
            "gp_validation_performed": bool(gp_validation_performed),
        })
        self._consensus_debug_checkpoint(staged_diag, label)

    def _consensus_collect_band_candidates(
        self,
        *,
        min_points_per_band=None,
        max_gap_fraction=None,
        min_duty_cycle=None,
        use_acf=False,
        gp_validation_requested=False,
        verbose=False,
    ):
        """Collect one dominant LS frequency candidate per accepted band.

        This function computes per-band sampling metrics, applies conservative
        pre-fit band rejection, runs per-band LS (and optional ACF diagnostics),
        and retains at most one dominant physically plausible LS candidate per
        band. When ACF is enabled, LS-vs-ACF disagreement triggers rejection,
        while direct or harmonic agreement is preserved as diagnostic support.

        Parameters
        ----------
        min_points_per_band : int or None, optional
            Minimum number of points required to accept a band prior to LS.
            If ``None``, derived from per-band sampling metrics.
        max_gap_fraction : float or None, optional
            Maximum allowed largest-gap fraction per band. If ``None``, derived
            from per-band sampling metrics.
        min_duty_cycle : float or None, optional
            Minimum allowed duty-cycle estimate per band. If ``None``, derived
            from per-band sampling metrics.
        use_acf : bool, optional
            If ``True``, compute data-driven ACF diagnostics per accepted band.
        gp_validation_requested : bool, optional
            If ``True``, this stage records intent only; records remain
            ``gp_validation_status="not_requested"`` until the GP-validation
            stage runs and assigns ``skipped``/``failed``/``rejected``/
            ``success`` per band.
        verbose : bool, optional
            If ``True``, print per-band acceptance/rejection and dominant LS
            values.

        Returns
        -------
        dict
            Dictionary with control values, per-band records, accepted and
            rejected bands, and rejection reasons.
        """
        prepared = self._consensus_prepare_band_consensus_inputs(
            min_points_per_band=min_points_per_band,
            max_gap_fraction=max_gap_fraction,
            min_duty_cycle=min_duty_cycle,
        )
        per_band_lc = prepared["per_band_lc"]
        metrics_by_band = prepared["metrics_by_band"]
        controls = prepared["controls"]

        band_records = {}
        accepted_bands = []
        rejected_bands = []
        rejection_reasons = {}

        def _reject_band(record, reasons):
            self._consensus_add_rejection_reasons(record, reasons)
            band = record["band"]
            if band not in rejected_bands:
                rejected_bands.append(band)
            if band in accepted_bands:
                accepted_bands.remove(band)
            rejection_reasons[band] = list(record["rejection_reasons"])

        for band_label, lc_band in per_band_lc.items():
            metrics = metrics_by_band[band_label]
            reasons = self._consensus_reject_bad_bands(
                metrics=metrics,
                min_points_per_band=controls["min_points_per_band"],
                max_gap_fraction=controls["max_gap_fraction"],
                min_duty_cycle=controls["min_duty_cycle"],
            )
            record = self._consensus_initialize_band_record(
                band_label,
                metrics=metrics,
                gp_validation_requested=gp_validation_requested,
            )

            if reasons:
                _reject_band(record, reasons)
                band_records[band_label] = record
                continue

            ls_candidates = self._consensus_extract_band_ls_candidates(
                lc_band,
                metrics=metrics,
                num_requested_peaks=5,
            )
            if ls_candidates["ls_frequencies"].size == 0:
                _reject_band(record, [_CONSENSUS_REJECTION_REASON_NO_LS_PEAKS])
                band_records[band_label] = record
                continue

            plausible_candidates = ls_candidates["candidates"]
            if not plausible_candidates:
                _reject_band(
                    record, [_CONSENSUS_REJECTION_REASON_NO_PLAUSIBLE_LS_PEAK]
                )
                band_records[band_label] = record
                continue

            best_candidate = next(
                (
                    candidate
                    for candidate in plausible_candidates
                    if candidate["significant"]
                ),
                plausible_candidates[0],
            )
            dominant_freq = float(best_candidate["frequency"])

            # Final plausibility guard: frequency must meet the minimum
            # detectable frequency threshold.
            if (
                ls_candidates["min_detectable_frequency"] > 0
                and dominant_freq < ls_candidates["min_detectable_frequency"]
            ):
                _reject_band(
                    record,
                    [_CONSENSUS_REJECTION_REASON_CANDIDATE_FREQUENCY_TOO_LOW],
                )
                band_records[band_label] = record
                continue

            if use_acf:
                # ACF is optional in consensus vetting. Sparse/irregular bands
                # may fail ACF estimation, so failure falls back to LS-only
                # candidate vetting while preserving diagnostics.
                try:
                    _acf_result = lc_band.acf(method="data", normalize=True)
                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                    AttributeError,
                ) as exc:
                    _acf_result = None
                    record["acf_error"] = f"{type(exc).__name__}: {exc}"
                    record["acf_supported"] = False

                acf_candidate = self._consensus_extract_acf_candidate(_acf_result)
                if acf_candidate is not None:
                    record["acf_frequency"] = acf_candidate["frequency"]
                    record["acf_period"] = acf_candidate["period"]

                # Consistency check is performed in frequency space; period
                # fields in the record are presentation-only derivations.
                acf_compare = self._consensus_compare_ls_acf(
                    ls_frequency=dominant_freq,
                    acf_frequency=record["acf_frequency"],
                )
                self._consensus_set_acf_comparison_status(
                    record,
                    acf_compare["status"],
                    period_ratio=acf_compare["ratio"],
                    harmonic_order=acf_compare["harmonic_order"],
                )
                record["acf_supported"] = bool(
                    acf_compare["status"] in (
                        _ACF_STATUS_AGREEMENT,
                        _ACF_STATUS_HARMONIC,
                    )
                )

                # ACF is a direct time-domain periodicity diagnostic. Strong
                # LS-vs-ACF disagreement is treated conservatively as likely
                # LS window/alias contamination for shared-timescale consensus.
                if acf_compare["status"] == _ACF_STATUS_DISAGREEMENT:
                    _reject_band(
                        record, [_CONSENSUS_REJECTION_REASON_LS_ACF_DISAGREEMENT]
                    )
                    band_records[band_label] = record
                    continue

            # Consensus logic operates in frequency space internally. Period is
            # derived only for user-facing diagnostics.
            dominant_period = float(1.0 / dominant_freq)
            record["dominant_frequency"] = dominant_freq
            record["dominant_period"] = dominant_period
            record["ls_significant"] = bool(best_candidate["significant"])
            record["selected_from"] = "ls_primary_peak"
            self._consensus_set_band_status(
                record,
                _CONSENSUS_BAND_STATUS_ACCEPTED,
                [],
            )
            band_records[band_label] = record
            accepted_bands.append(band_label)

        if verbose:
            print("[consensus] accepted bands:", accepted_bands)
            print("[consensus] rejected bands:", rejected_bands)
            for _band, _record in band_records.items():
                _freq = _record["dominant_frequency"]
                _period = _record["dominant_period"]
                if _freq is not None:
                    _msg = (
                        f"[consensus] band={_band} dominant_period={_period:.6g} "
                        f"dominant_frequency={_freq:.6g}"
                    )
                    if use_acf:
                        _msg += (
                            f" acf_status={_record.get('acf_comparison_status')}"
                            f" harmonic_order={_record.get('acf_harmonic_order')}"
                        )
                    print(_msg)
                elif _band in rejection_reasons:
                    print(
                        f"[consensus] band={_band} rejected: "
                        f"{', '.join(rejection_reasons[_band])}"
                    )

        # Final guard: records define accepted/rejected membership.
        for _band, _record in band_records.items():
            _reasons = self._consensus_normalize_rejection_reasons(
                _record.get("rejection_reasons")
            )
            if _reasons:
                self._consensus_set_band_status(
                    _record,
                    _CONSENSUS_BAND_STATUS_REJECTED,
                    _reasons,
                )
                if _band in accepted_bands:
                    accepted_bands.remove(_band)
                if _band not in rejected_bands:
                    rejected_bands.append(_band)
                rejection_reasons[_band] = list(_reasons)
            else:
                _status = _record.get("status")
                if _status == _CONSENSUS_BAND_STATUS_REJECTED:
                    _reasons = self._consensus_normalize_rejection_reasons(
                        rejection_reasons.get(_band)
                    )
                    self._consensus_set_band_status(
                        _record,
                        _CONSENSUS_BAND_STATUS_REJECTED,
                        _reasons,
                    )
                    if _band not in rejected_bands:
                        rejected_bands.append(_band)
                    rejection_reasons[_band] = list(
                        _record.get("rejection_reasons", [])
                    )
                else:
                    self._consensus_set_band_status(
                        _record,
                        _CONSENSUS_BAND_STATUS_ACCEPTED,
                        [],
                    )
                    rejection_reasons.pop(_band, None)
            band_records[_band] = self._consensus_make_json_safe(_record)

        accepted_bands = [b for b in accepted_bands if b not in set(rejected_bands)]

        self._consensus_debug_checkpoint_from_candidate_state(
            label="after_per_band_ls_candidate_extraction",
            controls=controls,
            band_records=band_records,
            accepted_bands=accepted_bands,
            rejected_bands=rejected_bands,
            rejection_reasons=rejection_reasons,
            use_acf=False,
            use_gp_validation=bool(gp_validation_requested),
            gp_validation_requested=bool(gp_validation_requested),
            gp_validation_performed=False,
        )
        if use_acf:
            self._consensus_debug_checkpoint_from_candidate_state(
                label="after_acf_validation_comparison",
                controls=controls,
                band_records=band_records,
                accepted_bands=accepted_bands,
                rejected_bands=rejected_bands,
                rejection_reasons=rejection_reasons,
                use_acf=True,
                use_gp_validation=bool(gp_validation_requested),
                gp_validation_requested=bool(gp_validation_requested),
                gp_validation_performed=False,
            )

        return {
            "controls": self._consensus_make_json_safe(controls),
            "band_records": band_records,
            "accepted_bands": accepted_bands,
            "rejected_bands": rejected_bands,
            "rejection_reasons": self._consensus_make_json_safe(rejection_reasons),
        }

    def _consensus_collect_band_component_candidates(
        self,
        *,
        max_components_per_band=3,
        min_points_per_band=None,
        max_gap_fraction=None,
        min_duty_cycle=None,
        verbose=False,
    ):
        """Collect multiple LS frequency candidates per accepted band.

        Per-band multi-component candidate extraction for future
        ``fit_strategy='consensus_multicomp'`` support.  This helper runs the
        same Lomb-Scargle machinery as
        :meth:`_consensus_collect_band_candidates` but retains up to
        ``max_components_per_band`` plausible peaks per band rather than only
        the single dominant peak.

        .. note::
           This helper is **not** called by the existing
           ``fit_strategy='consensus'`` path.  It is a dedicated building
           block for future per-band multi-component candidate extraction.
           Cross-band association and consensus aggregation of matched
           components are **not** implemented here; they will be added in
           subsequent PRs.

        Parameters
        ----------
        max_components_per_band : int, optional
            Maximum number of candidate components to collect per accepted
            band.  Only physically plausible peaks (finite positive frequency,
            within Nyquist and above the minimum detectable frequency) are
            included.  Default is 3.
        min_points_per_band : int or None, optional
            Minimum number of points required to accept a band prior to LS.
            If ``None``, derived from per-band sampling metrics.
        max_gap_fraction : float or None, optional
            Maximum allowed largest-gap fraction per band.  If ``None``,
            derived from per-band sampling metrics.
        min_duty_cycle : float or None, optional
            Minimum allowed duty-cycle estimate per band.  If ``None``,
            derived from per-band sampling metrics.
        verbose : bool, optional
            If ``True``, print per-band acceptance/rejection summaries and
            candidate details.

        Returns
        -------
        list of dict
            One entry per accepted band.  Each entry contains:

            ``band_name`` (str)
                Band label.
            ``wavelength`` (float or None)
                Numeric wavelength value from ``xdata[:, 1]``, or ``None``
                when the band mask resolves to no rows.
            ``component_candidates`` (list of dict)
                Up to ``max_components_per_band`` candidates in LS-rank
                order (highest power first).  Preservation of this order is
                critical — cross-band matching and canonical ordering are
                deferred to future PRs.  Each candidate dict contains:

                ``frequency`` (float)
                    LS peak frequency in the same units as ``xdata``.
                ``period`` (float)
                    ``1 / frequency``.
                ``ls_rank`` (int)
                    Zero-based rank by descending LS peak power
                    (0 = highest power).
                ``peak_power`` (float)
                    LS power at this peak.
                ``peak_prominence`` (float)
                    Prominence of this peak in the LS power spectrum
                    (``scipy.signal.peak_prominences``).  ``nan`` when
                    the peak index cannot be resolved.
                ``significant`` (bool)
                    ``True`` when the peak passes Benjamini-Hochberg FDR
                    correction (from :meth:`fit_LS`).
        """
        prepared = self._consensus_prepare_band_consensus_inputs(
            min_points_per_band=min_points_per_band,
            max_gap_fraction=max_gap_fraction,
            min_duty_cycle=min_duty_cycle,
            include_wavelengths=True,
        )
        per_band_lc = prepared["per_band_lc"]
        metrics_by_band = prepared["metrics_by_band"]
        controls = prepared["controls"]
        band_to_wavelength = prepared["band_to_wavelength"]

        results = []

        for band_label, lc_band in per_band_lc.items():
            metrics = metrics_by_band[band_label]
            reasons = self._consensus_reject_bad_bands(
                metrics=metrics,
                min_points_per_band=controls["min_points_per_band"],
                max_gap_fraction=controls["max_gap_fraction"],
                min_duty_cycle=controls["min_duty_cycle"],
            )
            if reasons:
                if verbose:
                    print(
                        f"[multicomp] band={band_label} rejected (quality): "
                        f"{', '.join(reasons)}"
                    )
                continue

            # Request more peaks than max_components_per_band to account
            # for implausible alias peaks (above Nyquist) that will be
            # filtered out.  See _CONSENSUS_MULTICOMP_ALIAS_PEAK_BUFFER
            # for rationale; mirrors the existing
            # _consensus_collect_band_candidates which requests num_peaks=5
            # to reliably find 1 plausible candidate when several aliases
            # dominate the periodogram at above-Nyquist frequencies.
            ls_candidates = self._consensus_extract_band_ls_candidates(
                lc_band,
                metrics=metrics,
                num_requested_peaks=(
                    max_components_per_band
                    + _CONSENSUS_MULTICOMP_ALIAS_PEAK_BUFFER
                ),
                max_candidates=max_components_per_band,
                include_peak_metadata=True,
            )
            if ls_candidates["ls_frequencies"].size == 0:
                if verbose:
                    print(f"[multicomp] band={band_label} rejected (no LS peaks)")
                continue

            component_candidates = ls_candidates["candidates"]
            if not component_candidates:
                if verbose:
                    print(
                        f"[multicomp] band={band_label} rejected "
                        "(no plausible candidates after filtering)"
                    )
                continue

            wavelength = band_to_wavelength.get(band_label)
            results.append(
                {
                    "band_name": str(band_label),
                    "wavelength": wavelength,
                    "component_candidates": component_candidates,
                }
            )

            if verbose:
                print(
                    f"[multicomp] band={band_label} "
                    f"wavelength={wavelength} "
                    f"n_candidates={len(component_candidates)}"
                )
                for cand in component_candidates:
                    print(
                        f"  rank={cand['ls_rank']} "
                        f"freq={cand['frequency']:.6g} "
                        f"period={cand['period']:.6g} "
                        f"power={cand['peak_power']:.4g} "
                        f"prom={cand['peak_prominence']:.4g} "
                        f"sig={cand['significant']}"
                    )

        return results

    def _consensus_cluster_component_candidates(
        self,
        band_component_candidates,
        *,
        cluster_frequency_rtol=0.10,
        min_bands_per_component=2,
    ):
        """Cluster per-band component candidates in log-frequency space.

        This helper performs cross-band association for the staged
        ``fit_strategy='consensus_multicomp'`` implementation. It does not
        perform final consensus aggregation or GP fitting.

        Parameters
        ----------
        band_component_candidates : list of dict
            Per-band candidate structure returned by
            :meth:`_consensus_collect_band_component_candidates`.
        cluster_frequency_rtol : float, optional
            Relative frequency tolerance used in log-frequency space.
            Candidates with ``abs(log(f1) - log(f2)) <= log1p(rtol)`` are
            considered nearby for greedy clustering.
        min_bands_per_component : int, optional
            Minimum number of unique bands required for a cluster to be marked
            as accepted.

        Returns
        -------
        list of dict
            Cluster diagnostics (both accepted and rejected) with fields:
            ``cluster_id``, ``accepted``, ``rejection_reasons``,
            ``member_bands``, ``n_member_bands``, ``center_frequency``,
            ``center_period``, ``log_center_frequency``,
            ``frequency_scatter``, ``log_frequency_scatter``, ``members``, and
            ``duplicate_band_candidates``.
        """

        def _as_float_or_none(value):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric):
                return None
            return numeric

        def _as_int_or_default(value, default):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _candidate_priority_key(member):
            peak_power = _as_float_or_none(member.get("peak_power"))
            if peak_power is None:
                peak_power = float("-inf")
            ls_rank = _as_int_or_default(member.get("ls_rank"), sys.maxsize)
            return (
                0 if bool(member.get("significant")) else 1,
                -peak_power,
                ls_rank,
                float(member["frequency"]),
                str(member["band_name"]),
            )

        def _build_member(band_name, wavelength, candidate):
            frequency = _as_float_or_none(candidate.get("frequency"))
            if frequency is None or frequency <= 0.0:
                return None
            return {
                "band_name": str(band_name),
                "wavelength": wavelength,
                "frequency": frequency,
                "period": candidate.get("period"),
                "ls_rank": candidate.get("ls_rank"),
                "peak_power": candidate.get("peak_power"),
                "peak_prominence": candidate.get("peak_prominence"),
                "significant": bool(candidate.get("significant")),
            }

        cluster_frequency_rtol = _as_float_or_none(cluster_frequency_rtol)
        if cluster_frequency_rtol is None or cluster_frequency_rtol < 0.0:
            cluster_frequency_rtol = 0.10
        min_bands_per_component = _as_int_or_default(min_bands_per_component, 2)
        if min_bands_per_component < 1:
            min_bands_per_component = 1

        # log1p computes log(1 + rtol) with better stability than log(1 + x)
        # when rtol is very small (avoids cancellation near zero).
        log_tol = float(np.log1p(cluster_frequency_rtol))
        flattened_members = []

        for band_entry in band_component_candidates or []:
            if not isinstance(band_entry, dict):
                continue
            band_name = band_entry.get("band_name")
            if band_name is None:
                continue
            wavelength = band_entry.get("wavelength")
            candidates = band_entry.get("component_candidates") or []
            if not isinstance(candidates, (list, tuple)):
                continue
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                member = _build_member(band_name, wavelength, candidate)
                if member is not None:
                    flattened_members.append(member)

        if not flattened_members:
            return []

        log_frequencies = np.array(
            [np.log(member["frequency"]) for member in flattened_members],
            dtype=float,
        )
        seed_order = sorted(
            range(len(flattened_members)),
            key=lambda idx: _candidate_priority_key(flattened_members[idx]),
        )
        unused = set(range(len(flattened_members)))
        tentative_clusters = []

        while unused:
            seed_idx = next(idx for idx in seed_order if idx in unused)
            cluster_members = {seed_idx}
            center_log_frequency = float(log_frequencies[seed_idx])

            while True:
                addable = {
                    idx
                    for idx in unused
                    if idx not in cluster_members
                    and abs(log_frequencies[idx] - center_log_frequency) <= log_tol
                }
                if not addable:
                    break
                updated_members = cluster_members | addable
                updated_center = float(
                    np.median([log_frequencies[idx] for idx in updated_members])
                )
                cluster_members = updated_members
                center_log_frequency = updated_center

            tentative_clusters.append(sorted(cluster_members))
            unused.difference_update(cluster_members)

        clusters = []
        for candidate_indices in tentative_clusters:
            tentative_members = [flattened_members[idx] for idx in candidate_indices]
            members_by_band = {}
            duplicate_members = []

            for member in sorted(tentative_members, key=_candidate_priority_key):
                band_name = member["band_name"]
                if band_name not in members_by_band:
                    members_by_band[band_name] = member
                else:
                    duplicate_members.append(member)

            retained_members = sorted(
                members_by_band.values(),
                key=lambda m: (float(m["frequency"]), str(m["band_name"])),
            )
            retained_log_freqs = np.array(
                [np.log(member["frequency"]) for member in retained_members],
                dtype=float,
            )
            retained_freqs = np.array(
                [member["frequency"] for member in retained_members],
                dtype=float,
            )
            center_log_frequency = float(np.median(retained_log_freqs))
            center_frequency = float(np.exp(center_log_frequency))
            center_period = float(1.0 / center_frequency)
            log_frequency_scatter = (
                float(np.std(retained_log_freqs, ddof=0))
                if retained_log_freqs.size > 1
                else 0.0
            )
            frequency_scatter = (
                float(np.std(retained_freqs, ddof=0))
                if retained_freqs.size > 1
                else 0.0
            )
            member_bands = sorted({member["band_name"] for member in retained_members})
            n_member_bands = len(member_bands)
            accepted = bool(n_member_bands >= min_bands_per_component)
            rejection_reasons = []
            if not accepted:
                rejection_reasons.append(
                    f"insufficient_bands ({n_member_bands} < {min_bands_per_component})"
                )

            clusters.append(
                {
                    "cluster_id": None,
                    "accepted": accepted,
                    "rejection_reasons": rejection_reasons,
                    "member_bands": member_bands,
                    "n_member_bands": n_member_bands,
                    "center_frequency": center_frequency,
                    "center_period": center_period,
                    "log_center_frequency": center_log_frequency,
                    "frequency_scatter": frequency_scatter,
                    "log_frequency_scatter": log_frequency_scatter,
                    "members": retained_members,
                    "duplicate_band_candidates": sorted(
                        duplicate_members,
                        key=_candidate_priority_key,
                    ),
                }
            )

        clusters.sort(
            key=lambda cluster: (
                float(cluster["center_frequency"]),
                float(cluster["log_center_frequency"]),
            )
        )
        for cluster_id, cluster in enumerate(clusters):
            cluster["cluster_id"] = cluster_id

        return clusters

    def _consensus_build_multicomponent_frequency_consensus(
        self,
        component_clusters,
        *,
        min_width_fraction=0.05,
    ):
        """Build per-component frequency consensus from cross-band clusters.

        This helper is the aggregation stage for future
        ``fit_strategy='consensus_multicomp'`` support.  It consumes cluster
        diagnostics from :meth:`_consensus_cluster_component_candidates` and
        computes one consensus frequency/width/scale triplet per accepted
        cluster.
        """

        def _as_positive_float_or_none(value):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric) or numeric <= 0.0:
                return None
            return numeric

        def _member_base_weight(member):
            peak_power = _as_positive_float_or_none(member.get("peak_power"))
            if peak_power is None:
                peak_power = 1.0
            significance_factor = 1.0 if bool(member.get("significant")) else 0.5
            return peak_power * significance_factor

        min_width_fraction = _as_positive_float_or_none(min_width_fraction)
        if min_width_fraction is None:
            min_width_fraction = 0.05

        accepted_clusters = []
        rejected_clusters = []
        for cluster in component_clusters or []:
            if not isinstance(cluster, dict):
                continue
            if bool(cluster.get("accepted")):
                accepted_clusters.append(cluster)
            else:
                rejected_clusters.append(cluster)

        def _cluster_id_sort_key(cluster):
            cluster_id = cluster.get("cluster_id")
            try:
                cluster_id = int(cluster_id)
            except (TypeError, ValueError):
                return (1, float("inf"))
            return (0, cluster_id)

        accepted_clusters.sort(key=_cluster_id_sort_key)

        component_payloads = []
        for cluster in accepted_clusters:
            members = cluster.get("members") or []
            valid_members = []
            valid_log_frequencies = []
            raw_weights = []
            fallback_weights = []

            for member in members:
                if not isinstance(member, dict):
                    continue
                frequency = _as_positive_float_or_none(member.get("frequency"))
                if frequency is None:
                    continue
                valid_members.append(member)
                valid_log_frequencies.append(float(np.log(frequency)))
                raw_weight = _member_base_weight(member)
                raw_weights.append(raw_weight)
                fallback_weights.append(
                    1.0 if bool(member.get("significant")) else 0.5
                )

            if not valid_members:
                continue

            log_freqs = np.asarray(valid_log_frequencies, dtype=float)
            candidate_weights = np.asarray(raw_weights, dtype=float)
            usable_weight_mask = np.isfinite(candidate_weights) & (
                candidate_weights > 0.0
            )

            if np.any(usable_weight_mask):
                used_weights = candidate_weights[usable_weight_mask]
                used_log_freqs = log_freqs[usable_weight_mask]
                weight_sum = float(np.sum(used_weights))
                if np.isfinite(weight_sum) and weight_sum > 0.0:
                    consensus_method = "weighted_log_frequency_center"
                    consensus_log_frequency = float(
                        np.sum(used_weights * used_log_freqs) / weight_sum
                    )
                    if used_log_freqs.size > 1:
                        centered = used_log_freqs - consensus_log_frequency
                        measured_log_scatter = float(
                            np.sqrt(
                                np.sum(used_weights * centered * centered) / weight_sum
                            )
                        )
                    else:
                        measured_log_scatter = 0.0
                    consensus_scale = weight_sum
                else:
                    usable_weight_mask[:] = False

            if not np.any(usable_weight_mask):
                consensus_method = "median_log_frequency_fallback"
                consensus_log_frequency = float(np.median(log_freqs))
                measured_log_scatter = (
                    float(np.std(log_freqs, ddof=0)) if log_freqs.size > 1 else 0.0
                )
                fallback_weight_values = np.asarray(fallback_weights, dtype=float)
                fallback_weight_values = fallback_weight_values[
                    np.isfinite(fallback_weight_values) & (fallback_weight_values > 0.0)
                ]
                if fallback_weight_values.size == 0:
                    consensus_scale = float(log_freqs.size)
                else:
                    consensus_scale = float(np.sum(fallback_weight_values))

            consensus_frequency = float(np.exp(consensus_log_frequency))
            consensus_period = float(1.0 / consensus_frequency)
            measured_frequency_width = float(consensus_frequency * measured_log_scatter)
            minimum_frequency_width = float(min_width_fraction * consensus_frequency)
            consensus_frequency_width = float(
                max(measured_frequency_width, minimum_frequency_width)
            )

            component_payloads.append(
                {
                    "source_cluster_id": cluster.get("cluster_id"),
                    "consensus_frequency": consensus_frequency,
                    "consensus_period": consensus_period,
                    "consensus_log_frequency": consensus_log_frequency,
                    "consensus_frequency_width": consensus_frequency_width,
                    "consensus_scale": consensus_scale,
                    "consensus_method": consensus_method,
                    "member_bands": cluster.get("member_bands"),
                    "n_member_bands": cluster.get("n_member_bands"),
                    "frequency_scatter": cluster.get("frequency_scatter"),
                    "log_frequency_scatter": cluster.get("log_frequency_scatter"),
                    "members": list(cluster.get("members") or []),
                }
            )

        component_summaries = []
        consensus_frequencies = []
        consensus_frequency_widths = []
        consensus_scales = []
        for component_index, payload in enumerate(component_payloads):
            summary = dict(payload)
            summary["component_index"] = component_index
            component_summaries.append(summary)
            consensus_frequencies.append(float(summary["consensus_frequency"]))
            consensus_frequency_widths.append(float(summary["consensus_frequency_width"]))
            consensus_scales.append(float(summary["consensus_scale"]))

        return self._consensus_make_json_safe(
            {
                "consensus_frequencies": consensus_frequencies,
                "consensus_frequency_widths": consensus_frequency_widths,
                "consensus_scales": consensus_scales,
                "accepted_clusters": accepted_clusters,
                "rejected_clusters": rejected_clusters,
                "component_summaries": component_summaries,
            }
        )

    @staticmethod
    def _consensus_prepare_gp_validation_fit_kwargs(gp_validation_kwargs=None):
        """Build a safe, isolated kwarg dict for nested 1D GP validation fits.

        All consensus logic operates in frequency space [1/day]. Periods are
        derived from frequencies only for user-facing diagnostics.

        This helper:

        1. Starts from conservative defaults (``model="1D"``,
           ``num_mixtures=1``, ``use_mls_init=True``,
           ``training_iter=100``).
        2. Merges non-blocked user overrides from ``gp_validation_kwargs``.
        3. Always forces ``fit_strategy=None`` after merging.

        Why ``fit_strategy`` is forced to ``None``
        ------------------------------------------
        Nested validation fits run on 1D per-band ``Lightcurve`` objects
        created by ``select_bands``. These objects have no multi-band
        structure and cannot support ``fit_strategy="consensus"``.
        Propagating the outer ``fit_strategy`` would cause infinite
        recursion or a misleading error.

        Why consensus-only kwargs are stripped
        --------------------------------------
        Keys like ``consensus_frequencies``, ``use_gp_validation``, and
        ``outlier_sigma`` are meaningful only at the 2D consensus level.
        Forwarding them into nested 1D fits would either be silently
        ignored or cause unexpected errors.

        Parameters
        ----------
        gp_validation_kwargs : dict or None, optional
            User-supplied overrides. The reserved nested key
            ``period_summary_kwargs`` is stripped here and must NOT be
            forwarded to ``fit()``; callers must pass it separately to
            ``get_period_summary``.

        Returns
        -------
        dict
            Sanitized kwargs safe for a nested 1D ``Lightcurve.fit()``
            call. The ``fit_strategy`` key is always ``None``.
        """
        # Keys that must NOT propagate into nested GP validation fits.
        # Consensus-only kwargs are meaningless or harmful for 1D band fits.
        # period_summary_kwargs is forwarded separately to get_period_summary.
        _BLOCKED_KEYS = frozenset({
            "fit_strategy",
            "consensus_frequencies",
            "consensus_scales",
            "consensus_frequency_width",
            "consensus_frequency_k",
            "consensus_scale_max_factor",
            "apply_consensus_constraints",
            "constrain_consensus",
            "use_gp_validation",
            "gp_validation_kwargs",
            "gp_frequency_tolerance_factor",
            "outlier_sigma",
            "consensus_width_factor",
            "consensus_dedup_rtol",
            "min_points_per_band",
            "max_gap_fraction",
            "min_duty_cycle",
            "use_acf",
            "period_summary_kwargs",
        })

        # Conservative defaults: keep validation lightweight and reproducible.
        # Users can override non-blocked keys via gp_validation_kwargs.
        defaults = {
            "model": "1D",
            "num_mixtures": 1,
            "use_mls_init": True,
            "training_iter": 100,
        }

        merged = dict(defaults)
        if gp_validation_kwargs:
            for key, val in gp_validation_kwargs.items():
                if key not in _BLOCKED_KEYS:
                    merged[key] = val

        # Force fit_strategy=None last, regardless of any user override.
        # A recursive consensus fit on a 1D band lightcurve is always wrong.
        merged["fit_strategy"] = None
        return merged

    def _consensus_validate_candidates_with_1d_gp(
        self,
        candidate_diag,
        gp_validation_kwargs=None,
        gp_frequency_tolerance_factor=3.0,
        verbose=False,
    ):
        """Validate LS/ACF-vetted band candidates against per-band 1D GP PSD.

        All comparison logic operates in frequency space [1/day].  Period
        values stored in ``band_records`` (``gp_dominant_period`` etc.) are
        derived from the validated GP frequency solely for user-facing display
        and are not used in any acceptance/rejection decision.
        """
        if not isinstance(candidate_diag, dict):
            raise ValueError("candidate_diag must be a dictionary.")
        if gp_validation_kwargs is None:
            gp_validation_kwargs = {}
        elif not isinstance(gp_validation_kwargs, dict):
            raise ValueError("gp_validation_kwargs must be None or a dictionary.")

        gp_frequency_tolerance_factor = float(gp_frequency_tolerance_factor)
        if (
            not np.isfinite(gp_frequency_tolerance_factor)
            or gp_frequency_tolerance_factor <= 0
        ):
            raise ValueError(
                "gp_frequency_tolerance_factor must be a finite, strictly "
                "positive float."
            )

        period_summary_kwargs = gp_validation_kwargs.get("period_summary_kwargs")
        if period_summary_kwargs is None:
            period_summary_kwargs = {}
        elif not isinstance(period_summary_kwargs, dict):
            raise ValueError(
                "gp_validation_kwargs['period_summary_kwargs'] must be a "
                "dictionary when provided."
            )

        controls = dict(candidate_diag.get("controls", {}))
        band_records = {}
        for band, record in candidate_diag.get("band_records", {}).items():
            canonical = self._consensus_initialize_band_record(
                band,
                metrics=(record or {}).get("metrics"),
                gp_validation_requested=True,
            )
            canonical.update(dict(record))
            canonical_reasons = self._consensus_normalize_rejection_reasons(
                canonical.get("rejection_reasons")
            )
            canonical_status = (
                _CONSENSUS_BAND_STATUS_REJECTED
                if canonical_reasons
                else canonical.get("status", _CONSENSUS_BAND_STATUS_PENDING)
            )
            if canonical_status == _CONSENSUS_BAND_STATUS_REJECTED:
                self._consensus_set_band_status(
                    canonical,
                    _CONSENSUS_BAND_STATUS_REJECTED,
                    canonical_reasons,
                )
            elif canonical_status == _CONSENSUS_BAND_STATUS_ACCEPTED:
                self._consensus_set_band_status(
                    canonical,
                    _CONSENSUS_BAND_STATUS_ACCEPTED,
                    [],
                )
            else:
                self._consensus_set_band_status(
                    canonical,
                    _CONSENSUS_BAND_STATUS_PENDING,
                    [],
                )
            band_records[band] = canonical
        accepted_bands = list(candidate_diag.get("accepted_bands", []))
        rejected_bands = list(candidate_diag.get("rejected_bands", []))
        rejection_reasons = {}
        for band, reasons in candidate_diag.get("rejection_reasons", {}).items():
            normalized = self._consensus_normalize_rejection_reasons(reasons)
            if normalized:
                rejection_reasons[band] = normalized

        for record in band_records.values():
            record.setdefault("gp_validation_used", False)
            record.setdefault("gp_dominant_frequency", None)
            record.setdefault("gp_dominant_period", None)
            record.setdefault("gp_frequency_difference", None)
            record.setdefault("gp_fractional_frequency_difference", None)
            record.setdefault("gp_frequency_tolerance", None)
            if "gp_validation_status" not in record:
                self._consensus_set_gp_validation_status(
                    record,
                    _CONSENSUS_GP_VALIDATION_STATUS_NOT_REQUESTED,
                )
            record.setdefault("gp_validation_error", None)

        # GP validation is requested for this function call. Bands not in
        # accepted_bands are intentionally bypassed by pre-GP filtering.
        for band, record in band_records.items():
            if (
                band not in accepted_bands
                and not record.get("gp_validation_used", False)
            ):
                self._consensus_set_gp_validation_status(
                    record,
                    _CONSENSUS_GP_VALIDATION_STATUS_SKIPPED,
                    reason=_CONSENSUS_GP_VALIDATION_REASON_BAND_NOT_ACCEPTED,
                )

        default_gp_fit_kwargs = (
            self._consensus_prepare_gp_validation_fit_kwargs(gp_validation_kwargs)
        )
        gp_ls_tolerance_base_factor = 0.1

        accepted_after_gp = []
        for band_label in accepted_bands:
            record = band_records.get(
                band_label,
                self._consensus_initialize_band_record(
                    band_label, gp_validation_requested=True
                ),
            )
            band_records[band_label] = record

            record["gp_validation_used"] = True
            # The pre-attempt state is "failed"; it remains "failed" if any
            # exception is raised during the GP fit.
            self._consensus_set_gp_validation_status(
                record, _CONSENSUS_GP_VALIDATION_STATUS_FAILED
            )
            record["gp_validation_error"] = None

            _candidate_frequency_raw = record.get("dominant_frequency", np.nan)
            candidate_frequency = (
                float(_candidate_frequency_raw)
                if _candidate_frequency_raw is not None
                else np.nan
            )
            # Period is derived for verbose display only; all GP validation
            # decisions are made in frequency space.
            gp_dominant_frequency = None
            gp_dominant_period = None
            reason = None

            try:
                if not (
                    np.isfinite(candidate_frequency) and candidate_frequency > 0
                ):
                    raise ValueError(
                        "dominant_frequency is missing or invalid for GP validation."
                    )

                # Validation is performed on a separate 1D Lightcurve returned
                # by select_bands. The per-band lc_band.fit() call mutates
                # only lc_band — it does NOT affect self.model, self.likelihood,
                # self.guess, or self.consensus_diagnostics on this instance.
                # (self.consensus_diagnostics is assigned by _consensus_standard_fit
                # after all per-band validation has finished, not here.)
                lc_band = self.select_bands([str(band_label)])
                lc_band.fit(**default_gp_fit_kwargs)
                summary = lc_band.get_period_summary(**period_summary_kwargs)

                gp_dominant_frequency = getattr(summary, "dominant_frequency", None)
                if gp_dominant_frequency is None and hasattr(summary, "get"):
                    gp_dominant_frequency = summary.get("dominant_frequency")

                gp_dominant_frequency = float(gp_dominant_frequency)
                if not (
                    np.isfinite(gp_dominant_frequency)
                    and gp_dominant_frequency > 0
                ):
                    raise ValueError("GP dominant frequency is not finite/positive.")

                # Consensus logic operates in frequency space internally.
                # Period is derived from frequency only for display/diagnostics.
                gp_dominant_period = float(1.0 / gp_dominant_frequency)

                frequency_tolerance = max(
                    gp_frequency_tolerance_factor
                    * gp_ls_tolerance_base_factor
                    * min(candidate_frequency, gp_dominant_frequency),
                    1.0e-8,
                )
                frequency_difference = abs(
                    gp_dominant_frequency - candidate_frequency
                )
                fractional_frequency_difference = (
                    self._consensus_fractional_frequency_difference(
                        candidate_frequency, gp_dominant_frequency
                    )
                )

                record["gp_dominant_frequency"] = gp_dominant_frequency
                record["gp_dominant_period"] = gp_dominant_period
                record["gp_frequency_difference"] = float(frequency_difference)
                record["gp_fractional_frequency_difference"] = float(
                    fractional_frequency_difference
                )
                record["gp_frequency_tolerance"] = float(frequency_tolerance)

                if frequency_difference <= frequency_tolerance:
                    # GP fit succeeded and the band is accepted.
                    self._consensus_set_gp_validation_status(
                        record, _CONSENSUS_GP_VALIDATION_STATUS_SUCCESS
                    )
                    self._consensus_set_band_status(
                        record,
                        _CONSENSUS_BAND_STATUS_ACCEPTED,
                        [],
                    )
                    accepted_after_gp.append(band_label)
                else:
                    # GP fit succeeded but disagreed with the LS candidate.
                    self._consensus_set_gp_validation_status(
                        record,
                        _CONSENSUS_GP_VALIDATION_STATUS_REJECTED,
                        reason=_CONSENSUS_GP_VALIDATION_REASON_DIAGNOSTICS_FAILED,
                    )
                    reason = _CONSENSUS_REJECTION_REASON_GP_LS_DISAGREEMENT

            except Exception as exc:
                # GP fit attempt raised/failed.
                self._consensus_set_gp_validation_status(
                    record,
                    _CONSENSUS_GP_VALIDATION_STATUS_FAILED,
                    reason=_CONSENSUS_GP_VALIDATION_REASON_EXCEPTION,
                )
                record["gp_validation_error"] = (
                    f"band={band_label}: {type(exc).__name__}: {exc}"
                )
                reason = _CONSENSUS_REJECTION_REASON_GP_VALIDATION_FAILED

            if reason is not None:
                self._consensus_add_rejection_reasons(record, [reason])
                if band_label not in rejected_bands:
                    rejected_bands.append(band_label)
                rejection_reasons[band_label] = list(record["rejection_reasons"])

            if verbose:
                _status = record.get("gp_validation_status")
                # Period shown here is derived from frequency for display only.
                _ls_period_display = (
                    float(1.0 / candidate_frequency)
                    if (
                        np.isfinite(candidate_frequency)
                        and candidate_frequency > 0
                    )
                    else None
                )
                _msg = (
                    f"[consensus][gp] band={band_label} "
                    f"ls_period_display={_ls_period_display} "
                    f"ls_frequency={record.get('dominant_frequency')} "
                    f"gp_period={record.get('gp_dominant_period')} "
                    f"gp_frequency={record.get('gp_dominant_frequency')} "
                    f"status={_status}"
                )
                if reason is not None:
                    _msg += f" rejection_reason={reason}"
                print(_msg)

        # Guard: ensure no band can appear in both accepted and rejected.
        # Overlap is unlikely in normal operation but could occur if a band
        # was recorded in rejected_bands before GP validation ran (e.g. for
        # a missing LS frequency) and was somehow also added to accepted_after_gp.
        for band in rejected_bands:
            if band in band_records:
                band_reasons = self._consensus_normalize_rejection_reasons(
                    band_records[band].get("rejection_reasons")
                )
                if band_reasons:
                    self._consensus_set_band_status(
                        band_records[band],
                        _CONSENSUS_BAND_STATUS_REJECTED,
                        band_reasons,
                    )

        _rejected_set = set(rejected_bands)
        final_accepted = [b for b in accepted_after_gp if b not in _rejected_set]
        for band in final_accepted:
            if band in band_records:
                self._consensus_set_band_status(
                    band_records[band],
                    _CONSENSUS_BAND_STATUS_ACCEPTED,
                    [],
                )
                rejection_reasons.pop(band, None)

        for band, record in band_records.items():
            reasons = self._consensus_normalize_rejection_reasons(
                record.get("rejection_reasons")
            )
            if reasons:
                self._consensus_set_band_status(
                    record,
                    _CONSENSUS_BAND_STATUS_REJECTED,
                    reasons,
                )
                rejection_reasons[band] = list(reasons)
            elif band in set(rejected_bands):
                reasons = self._consensus_normalize_rejection_reasons(
                    rejection_reasons.get(band)
                )
                if reasons:
                    self._consensus_set_band_status(
                        record,
                        _CONSENSUS_BAND_STATUS_REJECTED,
                        reasons,
                    )
                    rejection_reasons[band] = list(reasons)
                else:
                    self._consensus_set_band_status(
                        record,
                        _CONSENSUS_BAND_STATUS_ACCEPTED,
                        [],
                    )
            else:
                self._consensus_set_band_status(
                    record,
                    _CONSENSUS_BAND_STATUS_ACCEPTED,
                    [],
                )
            band_records[band] = self._consensus_make_json_safe(record)

        return {
            "controls": self._consensus_make_json_safe(controls),
            "band_records": band_records,
            "accepted_bands": final_accepted,
            "rejected_bands": rejected_bands,
            "rejection_reasons": self._consensus_make_json_safe(rejection_reasons),
        }

    def _deduplicate_frequency_candidates(
        self,
        candidates,
        rtol=0.01,
    ):
        """Collapse near-identical frequency candidates before ranking.

        Near-duplicate frequencies can split support across effectively
        identical peaks due to floating-point jitter. Deduplicating first keeps
        only the strongest representative in each cluster before downstream
        ranking and trusted-candidate selection.

        Notes
        -----
        ``rtol=0`` disables deduplication while still validating candidate
        structure and numeric values.
        """
        if candidates is None:
            return []

        _rtol = float(rtol)
        # rtol=0 is allowed and effectively disables deduplication.
        if not np.isfinite(_rtol) or _rtol < 0:
            raise ValueError("rtol must be a finite, non-negative float.")

        normalized = []
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError("Each candidate must be a dict.")
            if "frequency" not in candidate or "score" not in candidate:
                raise ValueError(
                    "Each candidate must contain 'frequency' and 'score'."
                )
            freq = float(candidate["frequency"])
            score = float(candidate["score"])
            if not (np.isfinite(freq) and freq > 0):
                raise ValueError(
                    "Candidate frequencies must be finite and positive."
                )
            if not np.isfinite(score):
                raise ValueError("Candidate scores must be finite.")
            candidate_copy = dict(candidate)
            candidate_copy["frequency"] = freq
            candidate_copy["score"] = score
            candidate_copy["_dedup_index"] = idx
            normalized.append(candidate_copy)

        if not normalized:
            return []

        parent = list(range(len(normalized)))

        def _find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def _union(i, j):
            ri = _find(i)
            rj = _find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(len(normalized)):
            f1 = float(normalized[i]["frequency"])
            for j in range(i + 1, len(normalized)):
                f2 = float(normalized[j]["frequency"])
                rel_diff = self._consensus_fractional_frequency_difference(f1, f2)
                if rel_diff < _rtol:
                    _union(i, j)

        clusters = {}
        for idx, candidate in enumerate(normalized):
            root = _find(idx)
            clusters.setdefault(root, []).append(candidate)

        deduped = []
        for cluster in clusters.values():
            best = max(
                cluster,
                key=lambda c: (float(c["score"]), -int(c["_dedup_index"])),
            )
            deduped.append(best)

        deduped.sort(key=lambda c: (-float(c["score"]), int(c["_dedup_index"])))

        output = []
        for candidate in deduped:
            candidate_copy = dict(candidate)
            candidate_copy.pop("_dedup_index", None)
            output.append(candidate_copy)
        return output

    def _consensus_build_frequency_consensus(
        self,
        band_records,
        accepted_bands,
        outlier_sigma=3.5,
        dedup_rtol=0.01,
        min_consensus_inliers=2,
        verbose=False,
    ):
        """Build a robust cross-band consensus frequency from dominant candidates.

        Near-duplicate frequencies are collapsed before robust aggregation so
        equivalent peaks do not receive duplicate weight during ranking.

        Parameters
        ----------
        band_records : dict
            Per-band candidate records containing at least
            ``dominant_frequency`` keys.
        accepted_bands : list[str]
            Bands to include in the robust aggregation stage.
        outlier_sigma : float, optional
            Robust sigma threshold used with MAD-based dispersion for
            catastrophic outlier rejection.
        dedup_rtol : float, optional
            Relative tolerance used to cluster near-identical frequency
            candidates before consensus ranking.
        min_consensus_inliers : int, optional
            Minimum number of bands that must survive outlier rejection and
            cluster around a common frequency for the consensus to be
            considered valid.  When fewer inliers survive, the method returns
            a diagnostics dict with ``insufficient_inliers=True`` and
            ``final_consensus_frequency=nan`` so the caller can finalize
            diagnostics and raise an informative ``RuntimeError``.  Defaults
            to ``2``, meaning at least two bands must agree.
        verbose : bool, optional
            If ``True``, print outlier decisions and final consensus values.

        Returns
        -------
        dict
            Aggregation diagnostics including median frequency, MAD scatter,
            inlier/outlier bands, and final consensus frequency.  When the
            inlier count is below ``min_consensus_inliers``, the dict
            contains ``insufficient_inliers=True`` and
            ``final_consensus_frequency=nan``.

        Raises
        ------
        ValueError
            If no valid per-band dominant frequencies are available.
        """
        freq_pairs = []
        for band in accepted_bands:
            freq = band_records[band].get("dominant_frequency")
            if freq is None:
                continue
            if np.isfinite(freq) and freq > 0:
                freq_pairs.append((band, float(freq)))

        if not freq_pairs:
            raise ValueError(
                "Consensus fit failed: no valid dominant per-band frequencies "
                "were available after quality gating."
            )

        # Deduplicate near-identical frequencies to prevent floating-point
        # jitter from counting equivalent peaks multiple times.
        _significant_ls_score = 2.0
        _default_ls_score = 1.0
        candidates = []
        for band, freq in freq_pairs:
            record = band_records.get(band, {})
            ls_significant = bool(record.get("ls_significant"))
            candidates.append(
                {
                    "frequency": float(freq),
                    "score": (
                        _significant_ls_score
                        if ls_significant
                        else _default_ls_score
                    ),
                    "band": band,
                }
            )

        n_before = len(candidates)
        candidates = self._deduplicate_frequency_candidates(
            candidates, rtol=dedup_rtol
        )
        n_after = len(candidates)
        if verbose:
            print(f"[Consensus] Deduplicated {n_before} -> {n_after} candidates")

        if not candidates:
            raise ValueError(
                "Consensus fit failed: no valid dominant per-band frequencies "
                "were available after deduplication."
            )

        freq_arr = np.asarray([cand["frequency"] for cand in candidates], dtype=float)
        band_arr = np.asarray([cand["band"] for cand in candidates], dtype=object)

        median_freq = float(np.median(freq_arr))
        mad_freq = float(np.median(np.abs(freq_arr - median_freq)))
        robust_sigma = 1.4826 * mad_freq

        if len(freq_arr) >= 3 and robust_sigma > 0 and np.isfinite(robust_sigma):
            abs_dev = np.abs(freq_arr - median_freq)
            inlier_mask = abs_dev <= float(outlier_sigma) * robust_sigma
        else:
            inlier_mask = np.ones_like(freq_arr, dtype=bool)

        if not np.any(inlier_mask):
            inlier_mask = np.ones_like(freq_arr, dtype=bool)

        outlier_bands = band_arr[~inlier_mask].tolist()
        inlier_freqs = freq_arr[inlier_mask]
        inlier_bands = band_arr[inlier_mask].tolist()

        # Count how many ORIGINAL bands (pre-deduplication) lie within the
        # inlier frequency window.  Near-identical frequencies from different
        # bands may have been collapsed into one representative candidate by
        # the deduplication step, so the deduped inlier list can be shorter
        # than the true number of bands that agree on a common frequency.
        # Using the original freq_pairs count correctly handles both the
        # "all bands agree" case (robust_sigma ≈ 0 → all original bands are
        # inliers) and the "mutually inconsistent" case (robust_sigma > 0 →
        # only original bands within the inlier window are counted).
        if len(freq_arr) >= 3 and robust_sigma > 0 and np.isfinite(robust_sigma):
            _inlier_tol = float(outlier_sigma) * robust_sigma
            n_original_inlier_bands = sum(
                1 for _, f in freq_pairs if abs(f - median_freq) <= _inlier_tol
            )
        else:
            # Robust scatter is zero or undefined: all original bands are
            # treated as inliers (no sigma-clipping is possible).
            n_original_inlier_bands = len(freq_pairs)

        # Require a minimum number of original bands in the inlier cluster.
        # If too few original bands agree, no scientifically defensible
        # consensus exists and the caller should report failure.
        if n_original_inlier_bands < int(min_consensus_inliers):
            return {
                "frequencies_all": freq_arr.tolist(),
                "bands_all": band_arr.tolist(),
                "median_frequency": median_freq,
                "mad_frequency_scatter": mad_freq,
                "inlier_bands": inlier_bands,
                "outlier_bands": outlier_bands,
                "final_consensus_frequency": float("nan"),
                "final_mad_frequency_scatter": float("nan"),
                "insufficient_inliers": True,
                "insufficient_inliers_count": n_original_inlier_bands,
                "required_min_consensus_inliers": int(min_consensus_inliers),
            }

        final_consensus_frequency = float(np.median(inlier_freqs))
        mad_scatter = float(np.median(np.abs(inlier_freqs - final_consensus_frequency)))

        if verbose:
            if outlier_bands:
                print(
                    "[consensus] outlier rejection removed bands:",
                    outlier_bands,
                )
            print(
                "[consensus] median frequency:",
                f"{median_freq:.6g}",
                "MAD:",
                f"{mad_freq:.6g}",
            )
            print(
                "[consensus] final consensus frequency:",
                f"{final_consensus_frequency:.6g}",
            )

        return {
            "frequencies_all": freq_arr.tolist(),
            "bands_all": band_arr.tolist(),
            "median_frequency": median_freq,
            "mad_frequency_scatter": mad_freq,
            "inlier_bands": inlier_bands,
            "outlier_bands": outlier_bands,
            "final_consensus_frequency": final_consensus_frequency,
            "final_mad_frequency_scatter": mad_scatter,
        }

    @staticmethod
    def _consensus_build_initialization_from_frequency(
        final_frequency,
        scatter,
        *,
        num_mixtures=1,
        consensus_width_factor=3.0,
    ):
        """Convert a scalar consensus frequency into init guesses and bounds.

        Parameters
        ----------
        final_frequency : float
            Final robust consensus frequency.
        scatter : float
            Robust frequency scatter estimate (MAD-based).
        num_mixtures : int, optional
            Number of spectral-mixture components for initialization vectors.
        consensus_width_factor : float, optional
            Multiplier applied to robust scatter to form constraint width.

        Returns
        -------
        dict
            Initialization payload containing ``consensus_frequencies``,
            ``consensus_scales``, ``consensus_frequency_width``, and
            ``consensus_constraint_bounds``.

        Raises
        ------
        ValueError
            If inputs are invalid (non-positive frequency or mixture count).
        """
        if not (np.isfinite(final_frequency) and final_frequency > 0):
            raise ValueError(
                "final consensus frequency must be positive and finite."
            )
        if not isinstance(num_mixtures, int) or num_mixtures < 1:
            raise ValueError("num_mixtures must be a positive integer.")

        floor_width = max(final_frequency * 0.01, 1.0e-8)
        if np.isfinite(scatter) and scatter > 0:
            width = float(consensus_width_factor) * float(scatter)
            width = max(width, floor_width)
        else:
            width = floor_width

        lower = max(final_frequency - width, _CONSENSUS_MIN_FREQUENCY_BOUND)
        upper = final_frequency + width
        scale_guess = max(
            float(scatter),
            final_frequency * 0.05,
            _CONSENSUS_MIN_SCALE_BOUND,
        )

        return {
            "consensus_frequencies": np.full(
                num_mixtures, final_frequency, dtype=float
            ),
            "consensus_scales": np.full(num_mixtures, scale_guess, dtype=float),
            "consensus_frequency_width": np.full(num_mixtures, width, dtype=float),
            "consensus_constraint_bounds": (float(lower), float(upper)),
        }

    def _consensus_build_multicomponent_initialization(
        self,
        multicomponent_consensus,
    ):
        """Convert multicomponent consensus outputs into fit-init arrays.

        This helper supports ``fit_strategy='consensus_multicomp'`` by turning
        the accepted-cluster consensus payload into the frequency, width, and
        scale arrays expected by the existing spectral-mixture initialization
        machinery. ``consensus_scales`` from the consensus stage are retained
        as component-strength diagnostics, while
        ``consensus_frequency_widths`` are used to initialize spectral-mixture
        ``mixture_scales``. The final fit still uses the current global
        constraint system; component-specific constraint intervals remain
        future work.
        """
        if not isinstance(multicomponent_consensus, dict):
            raise ValueError("multicomponent_consensus must be a dictionary.")

        consensus_frequencies = np.asarray(
            multicomponent_consensus.get("consensus_frequencies", []),
            dtype=float,
        ).ravel()
        if consensus_frequencies.size == 0:
            raise RuntimeError(
                "Consensus multi-component fit failed: no accepted "
                "multicomponent consensus clusters were available."
            )
        if not np.all(
            np.isfinite(consensus_frequencies) & (consensus_frequencies > 0)
        ):
            raise ValueError(
                "consensus_frequencies must contain finite, strictly positive "
                "values."
            )

        consensus_frequency_width = np.asarray(
            multicomponent_consensus.get("consensus_frequency_widths", []),
            dtype=float,
        ).ravel()
        if consensus_frequency_width.shape != consensus_frequencies.shape:
            raise ValueError(
                "consensus_frequency_widths must contain one finite positive "
                "entry per consensus frequency."
            )
        if not np.all(
            np.isfinite(consensus_frequency_width) & (consensus_frequency_width > 0)
        ):
            raise ValueError(
                "consensus_frequency_widths must contain finite, strictly "
                "positive values."
            )

        consensus_scales = np.asarray(
            multicomponent_consensus.get("consensus_scales", []),
            dtype=float,
        ).ravel()
        if consensus_scales.shape != consensus_frequencies.shape:
            raise ValueError(
                "consensus_scales must contain one finite positive entry per "
                "consensus frequency."
            )
        if not np.all(np.isfinite(consensus_scales) & (consensus_scales > 0)):
            raise ValueError(
                "consensus_scales must contain finite, strictly positive values."
            )

        return {
            "n_components": int(consensus_frequencies.size),
            "consensus_frequencies": consensus_frequencies,
            "consensus_scales": consensus_scales,
            "consensus_frequency_width": consensus_frequency_width,
        }

    @staticmethod
    def _consensus_multicomp_reconcile_to_n_components(
        accepted_component_summaries,
        requested_num_mixtures,
        rejected_clusters,
        band_component_candidates,
        min_width_fraction=0.05,
    ):
        """Reconcile accepted consensus components to the user-requested count.

        Given M accepted consensus components and a user-requested count N,
        this helper returns exactly N component descriptors and a diagnostics
        dict describing what was done.

        Parameters
        ----------
        accepted_component_summaries : list of dict
            Component summaries from
            :meth:`_consensus_build_multicomponent_frequency_consensus`.
            Each entry must contain ``consensus_frequency``,
            ``consensus_frequency_width``, ``consensus_scale``,
            ``n_member_bands``, and ``source_cluster_id``.
        requested_num_mixtures : int or None
            User-requested mixture count N.  When ``None`` the accepted
            count M is used unchanged.
        rejected_clusters : list of dict
            Rejected cluster dicts (fallback source a).
        band_component_candidates : list of dict
            Per-band component candidate dicts (fallback source b).
        min_width_fraction : float
            Minimum frequency width as a fraction of the centre frequency
            used when constructing fallback component widths.

        Returns
        -------
        tuple of (list of dict, dict)
            *reconciled_summaries* – exactly N component dicts, each
            containing the ``component_source`` provenance key.
            *reconciliation_diagnostics* – diagnostic fields to be merged
            into ``consensus_diagnostics``.
        """
        M = len(accepted_component_summaries)
        if requested_num_mixtures is None:
            N = M
        else:
            if isinstance(requested_num_mixtures, bool) or not isinstance(
                requested_num_mixtures, int
            ):
                raise TypeError(
                    "`num_mixtures` must be a positive integer or None, "
                    f"got {requested_num_mixtures!r} of type "
                    f"{type(requested_num_mixtures)!r}."
                )
            if requested_num_mixtures < 1:
                raise ValueError(
                    "`num_mixtures` must be a positive integer or None, "
                    f"got {requested_num_mixtures}."
                )
            N = requested_num_mixtures
        min_wf = (
            float(min_width_fraction)
            if (min_width_fraction and min_width_fraction > 0)
            else 0.05
        )

        # Tag accepted summaries with provenance
        accepted_with_source = [
            dict(s, component_source="accepted_consensus")
            for s in accepted_component_summaries
        ]

        reconciliation_diagnostics = {
            "requested_num_mixtures": (
                int(requested_num_mixtures)
                if requested_num_mixtures is not None
                else None
            ),
            "accepted_consensus_component_count": M,
            "initialization_component_count": None,
            "fitted_num_mixtures": None,
            "component_count_reconciliation_strategy": None,
            "dropped_consensus_components": [],
            "fallback_initialization_components": [],
        }

        if M == N:
            reconciled = list(accepted_with_source)
            reconciliation_diagnostics[
                "component_count_reconciliation_strategy"
            ] = "exact_match"

        elif M > N:
            # Rank: higher n_member_bands first, higher consensus_scale second,
            # lower source_cluster_id as deterministic tie-breaker.
            def _rank_key(s):
                n_mb = int(s.get("n_member_bands") or 0)
                scale = float(s.get("consensus_scale") or 0.0)
                cid = s.get("source_cluster_id")
                try:
                    cid_int = int(cid)
                except (TypeError, ValueError):
                    cid_int = 999999
                return (-n_mb, -scale, cid_int)

            sorted_summaries = sorted(accepted_with_source, key=_rank_key)
            kept = sorted_summaries[:N]
            dropped = sorted_summaries[N:]

            # Re-sort kept by source_cluster_id for deterministic ordering.
            kept.sort(
                key=lambda s: (
                    999999
                    if s.get("source_cluster_id") is None
                    else int(s["source_cluster_id"])
                )
            )

            reconciled = kept
            reconciliation_diagnostics["dropped_consensus_components"] = [
                {
                    "original_component_index": s.get("component_index"),
                    "source_cluster_id": s.get("source_cluster_id"),
                    "consensus_frequency": float(s["consensus_frequency"]),
                    "n_member_bands": s.get("n_member_bands"),
                    "consensus_scale": float(s.get("consensus_scale") or 0.0),
                }
                for s in dropped
            ]
            reconciliation_diagnostics[
                "component_count_reconciliation_strategy"
            ] = "drop_weakest"

        else:
            # M < N: pad with fallback components.
            reconciled = list(accepted_with_source)
            fallback_list = []
            needed = N - M

            accepted_freqs = np.array(
                [
                    float(s["consensus_frequency"])
                    for s in accepted_with_source
                    if s.get("consensus_frequency") is not None
                ],
                dtype=float,
            )
            all_used_freqs = list(map(float, accepted_freqs))

            def _freq_too_close(freq, used, rtol=0.1):
                if not used:
                    return False
                arr = np.array(used, dtype=float)
                return bool(np.any(np.abs(freq - arr) / arr < rtol))

            # Fallback a: rejected consensus clusters.
            if needed > 0 and rejected_clusters:
                sorted_rejected = sorted(
                    [c for c in rejected_clusters if isinstance(c, dict)],
                    key=lambda c: (
                        -int(c.get("n_member_bands") or 0),
                        int(c.get("cluster_id") or 0),
                    ),
                )
                for cluster in sorted_rejected:
                    if needed <= 0:
                        break
                    freq_raw = cluster.get("center_frequency")
                    try:
                        freq = float(freq_raw)
                    except (TypeError, ValueError):
                        continue
                    if not (np.isfinite(freq) and freq > 0):
                        continue
                    if _freq_too_close(freq, all_used_freqs):
                        continue
                    try:
                        scatter = float(cluster.get("frequency_scatter") or 0.0)
                    except (TypeError, ValueError):
                        scatter = 0.0
                    freq_width = max(scatter, min_wf * freq)
                    fallback_list.append({
                        "source_cluster_id": cluster.get("cluster_id"),
                        "consensus_frequency": freq,
                        "consensus_frequency_width": freq_width,
                        "consensus_scale": float(cluster.get("n_member_bands") or 1),
                        "n_member_bands": cluster.get("n_member_bands"),
                        "member_bands": list(cluster.get("member_bands") or []),
                        "component_source": "rejected_cluster_fallback",
                    })
                    all_used_freqs.append(freq)
                    needed -= 1

            # Fallback b: unused per-band candidate frequencies.
            if needed > 0 and band_component_candidates:
                all_cands = []
                for entry in band_component_candidates:
                    if not isinstance(entry, dict):
                        continue
                    band_name = entry.get("band_name")
                    for cand in entry.get("component_candidates") or []:
                        if not isinstance(cand, dict):
                            continue
                        try:
                            f = float(cand.get("frequency"))
                        except (TypeError, ValueError):
                            continue
                        if not (np.isfinite(f) and f > 0):
                            continue
                        all_cands.append({
                            "frequency": f,
                            "peak_power": float(cand.get("peak_power") or 0.0),
                            "band_name": band_name,
                        })
                all_cands.sort(key=lambda c: (-c["peak_power"], c["frequency"]))
                for cand in all_cands:
                    if needed <= 0:
                        break
                    freq = cand["frequency"]
                    if _freq_too_close(freq, all_used_freqs):
                        continue
                    freq_width = min_wf * freq
                    fallback_list.append({
                        "source_cluster_id": None,
                        "consensus_frequency": freq,
                        "consensus_frequency_width": freq_width,
                        "consensus_scale": 1.0,
                        "n_member_bands": 1,
                        "member_bands": (
                            [cand["band_name"]]
                            if cand["band_name"] is not None
                            else []
                        ),
                        "component_source": "per_band_candidate_fallback",
                    })
                    all_used_freqs.append(freq)
                    needed -= 1

            # Fallback c: broad fallback within the global frequency range.
            if needed > 0:
                if len(all_used_freqs) >= 2:
                    log_min = float(np.log(np.min(all_used_freqs)))
                    log_max = float(np.log(np.max(all_used_freqs)))
                elif len(all_used_freqs) == 1:
                    log_base = float(np.log(all_used_freqs[0]))
                    log_min = log_base - 1.0
                    log_max = log_base + 1.0
                else:
                    log_min = float(np.log(0.1))
                    log_max = float(np.log(10.0))
                for i in range(needed):
                    frac = (i + 0.5) / needed
                    log_f = log_min + frac * (log_max - log_min)
                    freq = float(np.exp(log_f))
                    # Up to 20 nudge attempts (1.05× per step) to avoid
                    # collisions with already-placed components.
                    for _ in range(20):
                        if not _freq_too_close(freq, all_used_freqs):
                            break
                        freq *= 1.05
                    freq_width = min_wf * freq
                    fallback_list.append({
                        "source_cluster_id": None,
                        "consensus_frequency": freq,
                        "consensus_frequency_width": freq_width,
                        "consensus_scale": 1.0,
                        "n_member_bands": 0,
                        "member_bands": [],
                        "component_source": "broad_fallback",
                    })
                    all_used_freqs.append(freq)

            reconciled = reconciled + fallback_list
            reconciliation_diagnostics["fallback_initialization_components"] = [
                {
                    "component_source": s["component_source"],
                    "consensus_frequency": float(s["consensus_frequency"]),
                    "n_member_bands": s.get("n_member_bands"),
                    "source_cluster_id": s.get("source_cluster_id"),
                }
                for s in fallback_list
            ]
            reconciliation_diagnostics[
                "component_count_reconciliation_strategy"
            ] = "pad_with_fallback"

        reconciliation_diagnostics["initialization_component_count"] = len(reconciled)
        return reconciled, reconciliation_diagnostics

    def _consensus_standard_fit(self, **fit_kwargs):
        """Run the conservative consensus fit workflow.

        For ``fit_strategy="consensus"``, this method:
        1) uses caller-provided consensus frequencies directly when supplied,
        2) collects one dominant LS candidate per band after quality gating,
           using LS with optional ACF consistency support diagnostics,
        3) optionally validates each LS/ACF-accepted band against the dominant
           frequency from a per-band 1D GP PSD fit when
           ``use_gp_validation=True`` — all comparisons and aggregation are
           performed in frequency space; user-facing diagnostics may report
           periods,
        4) builds a robust cross-band consensus using median and MAD in
           frequency space,
        5) forwards consensus outputs into existing initial-guess and
           constraint plumbing,
        6) dispatches to the standard fit path with merged guesses.

        Manual ``consensus_frequencies`` can be supplied directly and bypass
        automatic candidate collection. Automatic LS/ACF consensus construction
        currently requires a 2D light curve.

        Parameters
        ----------
        **fit_kwargs : dict
            Standard :meth:`fit` kwargs plus consensus-specific controls:
            ``min_points_per_band``, ``max_gap_fraction``, ``min_duty_cycle``,
            ``outlier_sigma``, ``min_consensus_inliers``, ``use_acf``,
            ``constrain_consensus``, ``consensus_width_factor``,
            ``consensus_dedup_rtol``, ``use_gp_validation``,
            ``gp_validation_kwargs``, and ``gp_frequency_tolerance_factor``.

            Manual overrides are also accepted via:

            - ``consensus_frequencies`` : array-like of float
              Consensus frequency values used directly when provided (automatic
              cross-band construction is skipped).
            - ``consensus_scales`` : array-like of float or scalar
              Optional spectral-mixture scale initialization values.
            - ``consensus_frequency_width`` : array-like of float or scalar
              Optional width values for consensus-frequency constraints.
            - ``consensus_frequency_k`` : float, default ``3.0``
              Multiplier applied to ``consensus_frequency_width`` when building
              mixture-mean constraint bounds.
            - ``consensus_scale_max_factor`` : float, default ``0.2``
              Multiplier on median consensus frequency to derive an upper bound
              for mixture-scale constraints when enabled.
            - ``consensus_dedup_rtol`` : float, default ``0.01``
              Relative tolerance used to cluster near-identical frequency
              candidates before consensus ranking. Must be finite and strictly
              positive.
            - ``min_consensus_inliers`` : int, default ``2``
              Minimum number of original (pre-deduplication) photometric bands
              that must lie within the inlier frequency window for the
              consensus to be accepted.  When fewer bands agree, the fit
              raises :class:`ConsensusFitError` and
              ``consensus_success`` is ``False``.
              This guards against spurious consensus frequencies when all
              accepted bands have mutually inconsistent periods.
            - ``use_gp_validation`` : bool, default ``False``
              If ``True``, run optional per-band 1D GP frequency validation on
              LS/ACF-vetted candidates before final consensus aggregation.
            - ``gp_validation_kwargs`` : dict or None, default ``None``
              Extra kwargs for per-band 1D GP validation fit and period summary.
              Reserved nested key: ``period_summary_kwargs`` (dict), forwarded
              only to :meth:`get_period_summary`.
            - ``gp_frequency_tolerance_factor`` : float, default ``3.0``
              Positive scale factor controlling the LS-vs-GP frequency
              consistency tolerance.

        Returns
        -------
        dict
            The result object returned by the underlying :meth:`fit` call.

        Raises
        ------
        ValueError
            If automatic consensus construction is requested for a non-2D light
            curve, or if consensus inputs fail validation.
        ConsensusFitError
            If the consensus pipeline determines that the data do not support a
            coherent shared period.  The exception carries a
            ``failure_diagnostics`` attribute with structured diagnostics.
            Possible reasons: all bands fail quality gating
            (``"no_accepted_bands"``); too few inlier bands after outlier
            rejection (``"insufficient_consensus_inliers"``); frequency
            aggregation error (``"frequency_aggregation_error"``); or an
            invalid aggregated frequency (``"invalid_consensus_frequency"``).
            In all cases ``lc.consensus_diagnostics`` is populated before the
            exception is raised.
        """
        consensus_frequencies = fit_kwargs.pop("consensus_frequencies", None)
        consensus_scales = fit_kwargs.pop("consensus_scales", None)
        user_guess = fit_kwargs.pop("guess", None)
        consensus_frequency_width = fit_kwargs.pop("consensus_frequency_width", None)
        consensus_frequency_k = fit_kwargs.pop("consensus_frequency_k", 3.0)
        consensus_scale_max_factor = fit_kwargs.pop("consensus_scale_max_factor", 0.2)
        legacy_apply_constraints = fit_kwargs.pop("apply_consensus_constraints", None)
        constrain_consensus = fit_kwargs.pop("constrain_consensus", None)
        if constrain_consensus is None:
            apply_consensus_constraints = (
                True
                if legacy_apply_constraints is None
                else bool(legacy_apply_constraints)
            )
        else:
            apply_consensus_constraints = bool(constrain_consensus)

        min_points_per_band = fit_kwargs.pop("min_points_per_band", None)
        max_gap_fraction = fit_kwargs.pop("max_gap_fraction", None)
        min_duty_cycle = fit_kwargs.pop("min_duty_cycle", None)
        outlier_sigma = fit_kwargs.pop("outlier_sigma", None)
        min_consensus_inliers = fit_kwargs.pop("min_consensus_inliers", 2)
        use_acf = fit_kwargs.pop("use_acf", False)
        consensus_width_factor = fit_kwargs.pop("consensus_width_factor", None)
        consensus_dedup_rtol = fit_kwargs.pop("consensus_dedup_rtol", 0.01)
        use_gp_validation = fit_kwargs.pop("use_gp_validation", False)
        gp_validation_kwargs = fit_kwargs.pop("gp_validation_kwargs", None)
        gp_frequency_tolerance_factor = fit_kwargs.pop(
            "gp_frequency_tolerance_factor", 3.0
        )
        verbose = fit_kwargs.get("verbose", False)
        _allow_existing = fit_kwargs.pop("_allow_existing_model_for_consensus", False)
        consensus_dedup_rtol = float(consensus_dedup_rtol)
        if not np.isfinite(consensus_dedup_rtol) or consensus_dedup_rtol <= 0:
            raise ValueError(
                "consensus_dedup_rtol must be a finite, strictly positive "
                "float."
            )
        if not isinstance(use_gp_validation, bool):
            raise ValueError("use_gp_validation must be a boolean.")
        if gp_validation_kwargs is not None and not isinstance(
            gp_validation_kwargs, dict
        ):
            raise ValueError("gp_validation_kwargs must be None or a dictionary.")
        gp_frequency_tolerance_factor = float(gp_frequency_tolerance_factor)
        if (
            not np.isfinite(gp_frequency_tolerance_factor)
            or gp_frequency_tolerance_factor <= 0
        ):
            raise ValueError(
                "gp_frequency_tolerance_factor must be a finite, strictly "
                "positive float."
            )

        result_diagnostics = self._consensus_initialize_result_structure(
            fit_strategy="consensus"
        )
        result_diagnostics.update({
            "use_acf_validation": bool(use_acf),
            "use_gp_validation": bool(use_gp_validation),
            "gp_validation_requested": bool(use_gp_validation),
            "gp_validation_performed": False,
            "consensus_generation_method": (
                "auto_consensus"
                if consensus_frequencies is None
                else "manual_consensus_frequencies"
            ),
        })
        self.consensus_diagnostics = self._consensus_finalize_result_structure(
            result_diagnostics
        )

        auto_constraint_bounds = None
        auto_controls = None
        if consensus_frequencies is None:
            if self.ndim != 2:
                raise ValueError(
                    "Automatic consensus frequency construction requires a 2D "
                    "light curve."
                )
            candidate_diag = self._consensus_collect_band_candidates(
                min_points_per_band=min_points_per_band,
                max_gap_fraction=max_gap_fraction,
                min_duty_cycle=min_duty_cycle,
                use_acf=use_acf,
                gp_validation_requested=use_gp_validation,
                verbose=verbose,
            )
            if use_gp_validation:
                validated_diag = self._consensus_validate_candidates_with_1d_gp(
                    candidate_diag=candidate_diag,
                    gp_validation_kwargs=gp_validation_kwargs,
                    gp_frequency_tolerance_factor=gp_frequency_tolerance_factor,
                    verbose=verbose,
                )
                candidate_diag = validated_diag
                result_diagnostics["gp_validation_performed"] = any(
                    bool(rec.get("gp_validation_used", False))
                    for rec in candidate_diag.get("band_records", {}).values()
                )
            auto_controls = candidate_diag["controls"]
            _band_records_snap = dict(candidate_diag.get("band_records", {}))
            _rejected_bands_snap = list(candidate_diag.get("rejected_bands", []))
            # Convert per-band {band: [reasons]} accumulator to canonical
            # top-level {reason: [bands]} format via the rejection-summary
            # builder before storing in result_diagnostics.
            _top_level_rr = self._consensus_build_rejection_summary(
                per_band_diagnostics=_band_records_snap,
                rejected_bands=_rejected_bands_snap,
                rejection_reasons=dict(candidate_diag.get("rejection_reasons", {})),
            )
            result_diagnostics.update({
                "accepted_bands": list(candidate_diag.get("accepted_bands", [])),
                "rejected_bands": _rejected_bands_snap,
                "rejection_reasons": _top_level_rr,
                "per_band_diagnostics": _band_records_snap,
            })
            if use_gp_validation:
                self._consensus_debug_checkpoint(
                    result_diagnostics, "after_gp_validation"
                )
            accepted_bands = candidate_diag.get("accepted_bands", [])
            if not accepted_bands:
                rejection_reasons = candidate_diag.get("rejection_reasons", {})
                self.consensus_diagnostics = self._consensus_finalize_result_structure(
                    result_diagnostics
                )
                _rr_summary = {
                    reason: list(bands)
                    for reason, bands in rejection_reasons.items()
                }
                raise ConsensusFitError(
                    "Consensus fit failed: the bands do not support a common "
                    "periodicity. Every band was rejected before LS frequency "
                    "extraction (e.g. too few points, excessive gaps, or no "
                    "reliable LS peaks). Check per-band sampling quality. "
                    f"Rejection reasons: {_rr_summary!r}.",
                    failure_diagnostics={
                        "status": "failed",
                        "reason": "no_accepted_bands",
                        "rejection_reasons": _rr_summary,
                    },
                )
            self._consensus_debug_checkpoint(
                result_diagnostics, "before_consensus_frequency_generation"
            )
            try:
                consensus_diag = self._consensus_build_frequency_consensus(
                    band_records=candidate_diag["band_records"],
                    accepted_bands=accepted_bands,
                    outlier_sigma=(
                        auto_controls["outlier_sigma"]
                        if outlier_sigma is None
                        else float(outlier_sigma)
                    ),
                    dedup_rtol=consensus_dedup_rtol,
                    min_consensus_inliers=int(min_consensus_inliers),
                    verbose=verbose,
                )
            except Exception as exc:
                self.consensus_diagnostics = self._consensus_finalize_result_structure(
                    result_diagnostics
                )
                raise ConsensusFitError(
                    "Consensus fit failed during robust frequency aggregation. "
                    "This may occur if accepted per-band frequencies are "
                    "invalid, all identical, or too sparse to compute a "
                    "reliable median. Check per-band dominant frequencies in "
                    f"consensus_diagnostics. Details: {exc}",
                    failure_diagnostics={
                        "status": "failed",
                        "reason": "frequency_aggregation_error",
                        "detail": str(exc),
                    },
                ) from exc

            # Insufficient inliers: the accepted bands do not cluster around
            # a common frequency.  Populate partial diagnostics for inspection
            # before raising a RuntimeError.
            if consensus_diag.get("insufficient_inliers"):
                n_found = consensus_diag.get("insufficient_inliers_count", 0)
                n_req = consensus_diag.get(
                    "required_min_consensus_inliers", int(min_consensus_inliers)
                )
                _resolved_outlier_sigma = (
                    auto_controls["outlier_sigma"]
                    if outlier_sigma is None
                    else float(outlier_sigma)
                )
                _resolved_width_factor = (
                    auto_controls["consensus_width_factor"]
                    if consensus_width_factor is None
                    else float(consensus_width_factor)
                )
                result_diagnostics.update({
                    "median_frequency": consensus_diag["median_frequency"],
                    "mad_frequency_scatter": consensus_diag["mad_frequency_scatter"],
                    "consensus_inlier_bands": consensus_diag["inlier_bands"],
                    "consensus_outlier_bands": consensus_diag["outlier_bands"],
                    "candidate_count": len(
                        consensus_diag.get("frequencies_all", [])
                    ),
                    "trusted_candidate_count": n_found,
                    "per_band_dominant_periods": {
                        band: rec["dominant_period"]
                        for band, rec in candidate_diag["band_records"].items()
                        if rec.get("dominant_period") is not None
                    },
                    "per_band_dominant_frequencies": {
                        band: rec["dominant_frequency"]
                        for band, rec in candidate_diag["band_records"].items()
                        if rec.get("dominant_frequency") is not None
                    },
                    "controls": {
                        **auto_controls,
                        "outlier_sigma": _resolved_outlier_sigma,
                        "use_acf": bool(use_acf),
                        "constrain_consensus": bool(apply_consensus_constraints),
                        "consensus_width_factor": _resolved_width_factor,
                        "consensus_dedup_rtol": float(consensus_dedup_rtol),
                        "use_gp_validation": bool(use_gp_validation),
                        "gp_frequency_tolerance_factor": float(
                            gp_frequency_tolerance_factor
                        ),
                    },
                })
                self.consensus_diagnostics = (
                    self._consensus_finalize_result_structure(result_diagnostics)
                )
                _cand_periods = [
                    (float(1.0 / f) if f and f > 0 else None)
                    for f in consensus_diag.get("frequencies_all", [])
                ]
                raise ConsensusFitError(
                    f"Consensus fit failed: the inferred periods are mutually "
                    f"inconsistent across bands. Only {n_found} band(s) "
                    f"clustered around a common frequency after outlier "
                    f"rejection, but {n_req} are required. The bands do not "
                    "support a coherent shared period — this is a data-quality "
                    "issue, not a software error.",
                    failure_diagnostics={
                        "status": "failed",
                        "reason": "insufficient_consensus_inliers",
                        "n_inlier_bands": n_found,
                        "required_inliers": n_req,
                        "n_candidate_bands": len(
                            consensus_diag.get("frequencies_all", [])
                        ),
                        "candidate_periods": _cand_periods,
                    },
                )

            final_consensus_frequency = float(
                consensus_diag.get("final_consensus_frequency", np.nan)
            )
            if not (
                np.isfinite(final_consensus_frequency)
                and final_consensus_frequency > 0
            ):
                self.consensus_diagnostics = self._consensus_finalize_result_structure(
                    result_diagnostics
                )
                raise ConsensusFitError(
                    "Consensus fit failed: the aggregated consensus frequency "
                    "is not finite or not strictly positive. This may indicate "
                    "that the accepted band frequencies are dominated by noise "
                    "or contain invalid (NaN/Inf) values.",
                    failure_diagnostics={
                        "status": "failed",
                        "reason": "invalid_consensus_frequency",
                        "frequency_value": (
                            None
                            if not math.isfinite(final_consensus_frequency)
                            else float(final_consensus_frequency)
                        ),
                    },
                )
            robust_width = float(
                consensus_diag.get("final_mad_frequency_scatter", np.nan)
            )
            if not (np.isfinite(robust_width) and robust_width > 0):
                robust_width = None

            if consensus_width_factor is None:
                consensus_width_factor = auto_controls["consensus_width_factor"]
            if outlier_sigma is None:
                outlier_sigma = auto_controls["outlier_sigma"]

            requested_num_mixtures = fit_kwargs.get("num_mixtures")
            if requested_num_mixtures is None:
                requested_num_mixtures = 1
                fit_kwargs["num_mixtures"] = 1

            # Automatic consensus strategy uses a single robust cross-band
            # frequency (LS primary + optional ACF support checks).
            consensus_frequencies = np.asarray(
                [final_consensus_frequency], dtype=float
            )
            if consensus_frequency_width is None and robust_width is not None:
                consensus_frequency_width = np.asarray([robust_width], dtype=float)
            auto_constraint_bounds = None

            result_diagnostics.update({
                "accepted_bands": candidate_diag["accepted_bands"],
                "rejected_bands": candidate_diag["rejected_bands"],
                # Convert per-band {band: [reasons]} accumulator to canonical
                # top-level {reason: [bands]} format.
                "rejection_reasons": self._consensus_build_rejection_summary(
                    per_band_diagnostics=candidate_diag["band_records"],
                    rejected_bands=candidate_diag["rejected_bands"],
                    rejection_reasons=dict(
                        candidate_diag.get("rejection_reasons", {})
                    ),
                ),
                "per_band_diagnostics": candidate_diag["band_records"],
                "per_band_dominant_periods": {
                    band: rec["dominant_period"]
                    for band, rec in candidate_diag["band_records"].items()
                    if rec["dominant_period"] is not None
                },
                "per_band_dominant_frequencies": {
                    band: rec["dominant_frequency"]
                    for band, rec in candidate_diag["band_records"].items()
                    if rec["dominant_frequency"] is not None
                },
                "candidate_count": len(consensus_diag.get("frequencies_all", [])),
                "trusted_candidate_count": len(consensus_diag.get("inlier_bands", [])),
                "consensus_frequency": final_consensus_frequency,
                "consensus_period": float(1.0 / final_consensus_frequency),
                "consensus_frequency_width": robust_width,
                "consensus_frequency_scatter": consensus_diag["mad_frequency_scatter"],
                "median_frequency": consensus_diag["median_frequency"],
                "mad_frequency_scatter": consensus_diag["mad_frequency_scatter"],
                "consensus_inlier_bands": consensus_diag["inlier_bands"],
                "consensus_outlier_bands": consensus_diag["outlier_bands"],
                "final_consensus_frequency": final_consensus_frequency,
                "final_consensus_period": float(1.0 / final_consensus_frequency),
                "robust_frequency_width": robust_width,
                "final_constraint_bounds": None,
                "controls": {
                    **auto_controls,
                    "outlier_sigma": float(outlier_sigma),
                    "use_acf": bool(use_acf),
                    "constrain_consensus": bool(apply_consensus_constraints),
                    "consensus_width_factor": float(consensus_width_factor),
                    "consensus_dedup_rtol": float(consensus_dedup_rtol),
                    "use_gp_validation": bool(use_gp_validation),
                    "gp_frequency_tolerance_factor": float(
                        gp_frequency_tolerance_factor
                    ),
                },
            })
            result_diagnostics["consensus_generation_method"] = (
                "auto_ls_acf_gp" if use_gp_validation else "auto_ls_acf"
            )
            self._consensus_debug_checkpoint(
                result_diagnostics, "before_finalization"
            )
            self.consensus_diagnostics = self._consensus_finalize_result_structure(
                result_diagnostics
            )
            if verbose:
                print("[consensus] accepted bands:", candidate_diag["accepted_bands"])
                print("[consensus] rejected bands:", candidate_diag["rejected_bands"])
                print(
                    "[consensus] per-band dominant periods:",
                    self.consensus_diagnostics["per_band_dominant_periods"],
                )
                print(
                    "[consensus] final consensus frequency:",
                    f"{final_consensus_frequency:.6g}",
                )
                print(
                    "[consensus] final consensus period:",
                    f"{(1.0 / final_consensus_frequency):.6g}",
                )
                print(
                    "[consensus] robust frequency width:",
                    "None"
                    if robust_width is None
                    else f"{float(robust_width):.6g}",
                )
        else:
            consensus_frequencies = np.asarray(
                consensus_frequencies, dtype=float
            ).ravel()
            if consensus_frequencies.size == 0:
                raise ValueError("consensus_frequencies must not be empty.")
            if not np.all(
                np.isfinite(consensus_frequencies) & (consensus_frequencies > 0)
            ):
                raise ValueError(
                    "consensus_frequencies must contain finite, strictly "
                    "positive values."
                )
            # Apply near-duplicate suppression before downstream ranking and
            # aggregation, using the configured consensus clustering tolerance.
            manual_candidates = [
                {"frequency": float(freq), "score": 1.0, "index": idx}
                for idx, freq in enumerate(consensus_frequencies.tolist())
            ]
            n_before = len(manual_candidates)
            manual_candidates = self._deduplicate_frequency_candidates(
                manual_candidates, rtol=consensus_dedup_rtol
            )
            n_after = len(manual_candidates)
            if verbose:
                print(f"[Consensus] Deduplicated {n_before} -> {n_after} candidates")

            consensus_frequencies = np.asarray(
                [cand["frequency"] for cand in manual_candidates], dtype=float
            )
            if consensus_frequencies.size == 0:
                raise ValueError("consensus_frequencies must not be empty.")

            if consensus_frequency_width is not None:
                _manual_widths = np.asarray(
                    consensus_frequency_width, dtype=float
                ).ravel()
                if _manual_widths.size > 1 and _manual_widths.size == n_before:
                    _selected_idx = np.asarray(
                        [int(cand["index"]) for cand in manual_candidates],
                        dtype=int,
                    )
                    consensus_frequency_width = _manual_widths[_selected_idx]
            if consensus_frequency_width is None:
                floor_width = np.maximum(consensus_frequencies * 0.01, 1.0e-8)
                consensus_frequency_width = floor_width
            _mad_frequency_scatter = float(
                np.median(
                    np.abs(consensus_frequencies - np.median(consensus_frequencies))
                )
            )
            result_diagnostics.update({
                "accepted_bands": [],
                "rejected_bands": [],
                "rejection_reasons": {},
                "per_band_diagnostics": {},
                "per_band_dominant_periods": {},
                "per_band_dominant_frequencies": {},
                "candidate_count": int(consensus_frequencies.size),
                "trusted_candidate_count": int(consensus_frequencies.size),
                "consensus_frequencies": consensus_frequencies.tolist(),
                "consensus_frequency_widths": (
                    consensus_frequency_width.tolist()
                    if hasattr(consensus_frequency_width, "tolist")
                    else list(consensus_frequency_width)
                ),
                "consensus_frequency": None,
                "consensus_period": None,
                "consensus_frequency_width": None,
                "consensus_frequency_scatter": _mad_frequency_scatter,
                "median_frequency": None,
                "mad_frequency_scatter": _mad_frequency_scatter,
                "final_consensus_frequency": None,
                "final_consensus_period": None,
                "robust_frequency_width": None,
                "final_constraint_bounds": None,
                "controls": {
                    "outlier_sigma": outlier_sigma,
                    "use_acf": bool(use_acf),
                    "constrain_consensus": bool(apply_consensus_constraints),
                    "consensus_width_factor": consensus_width_factor,
                    "consensus_dedup_rtol": float(consensus_dedup_rtol),
                    "use_gp_validation": bool(use_gp_validation),
                    "gp_frequency_tolerance_factor": float(
                        gp_frequency_tolerance_factor
                    ),
                },
                "mode": "manual_consensus_frequencies",
            })
            result_diagnostics["consensus_generation_method"] = (
                "manual_consensus_frequencies"
            )
            self.consensus_diagnostics = self._consensus_finalize_result_structure(
                result_diagnostics
            )

        # When a model is explicitly specified, always build a fresh model for
        # this consensus fit.  Never reuse stale model state from a previous
        # fit: the user may have requested a different model, time_kernel_type,
        # or num_mixtures in this call.
        # If model is None, require explicit internal opt-in via the private
        # flag _allow_existing_model_for_consensus to reuse a pre-existing
        # model; otherwise raise a clear error to prevent accidental stale-
        # state reuse in public consensus fits.
        _requested_model = fit_kwargs.get("model")
        if _requested_model is not None:
            self._consensus_clear_model_state()
        elif not _allow_existing:
            raise ConsensusFitError(
                "Consensus fit requires an explicit final model. "
                "Pass model='2D' or another spectral-mixture-compatible "
                "model. Pre-existing model reuse is disabled by default "
                "to prevent stale consensus constraints."
            )
        _set_model_excluded = {
            "model",
            "likelihood",
            "num_mixtures",
            "variance",
            "guess",
            "consensus_frequencies",
            "consensus_scales",
            "consensus_frequency_width",
            "consensus_frequency_k",
            "consensus_scale_max_factor",
            "apply_consensus_constraints",
            "constrain_consensus",
            "min_points_per_band",
            "max_gap_fraction",
            "min_duty_cycle",
            "outlier_sigma",
            "use_acf",
            "consensus_width_factor",
            "use_gp_validation",
            "gp_validation_kwargs",
            "gp_frequency_tolerance_factor",
            "periods",
            "use_mls_init",
            "use_best_band_init",
            "constraint_set",
            "grid_size",
            "cuda",
            "training_iter",
            "max_cg_iterations",
            "optim",
            "miniter",
            "stop",
            "lr",
            "stopavg",
            "fit_strategy",
            "verbose",
            "_allow_existing_model_for_consensus",
        }
        _model_needs_build = (
            _requested_model is not None
            or not (
                hasattr(self, "model")
                and self.model is not None
                and hasattr(self, "_model_pars")
            )
        )
        if _model_needs_build:
            set_model_kwargs = {
                key: value
                for key, value in fit_kwargs.items()
                if key not in _set_model_excluded
            }
            self.set_model(
                _requested_model,
                fit_kwargs.get("likelihood"),
                num_mixtures=fit_kwargs.get("num_mixtures"),
                variance=fit_kwargs.get("variance", False),
                **set_model_kwargs,
            )
        fit_kwargs["model"] = None

        if apply_consensus_constraints:
            # Validate that the freshly built model supports SM time-kernel
            # constraints before attempting to resolve keys.
            self._consensus_validate_final_model_supports_sm_time_kernel(
                model_name=_requested_model,
                time_kernel_type=fit_kwargs.get("time_kernel_type"),
            )
            _constraint_dict = {}
            _keys = self._consensus_resolve_time_spectral_mixture_keys()
            _frequency_constraint_bounds = None
            # --- Step 1: apply default/LPV constraints as a base first -------
            # This ensures any constraint_set period bounds are registered
            # before the consensus constraints override the mixture_means key.
            # Calling set_default_constraints also sets __CONTRAINTS_SET=True
            # which prevents _fit_core from re-applying defaults and
            # overwriting the consensus constraints below.
            _constraint_set_for_defaults = fit_kwargs.get("constraint_set")
            self.set_default_constraints(
                constraint_set=_constraint_set_for_defaults
            )
            result_diagnostics[
                "default_constraints_applied_before_consensus"
            ] = True
            # --- Step 2: build consensus constraint dict ---------------------
            if consensus_frequency_width is not None:
                _freqs = np.asarray(
                    consensus_frequencies, dtype=float
                ).ravel()
                _widths = np.asarray(
                    consensus_frequency_width, dtype=float
                ).ravel()
                if _widths.size == 1:
                    _widths = np.broadcast_to(_widths, _freqs.shape).copy()
                if _widths.shape != _freqs.shape:
                    _msg = (
                        "consensus_frequency_width must be scalar or have one "
                        "entry per consensus frequency "
                        f"(got {_widths.shape} vs {_freqs.shape})."
                    )
                    raise ValueError(_msg)
                if not np.all(np.isfinite(_widths) & (_widths > 0)):
                    raise ValueError(
                        "consensus_frequency_width values must all be "
                        "positive and finite."
                    )
                _k = float(consensus_frequency_k)
                # Practical lower bound - frequencies must be positive.
                _lowers = np.maximum(
                    _freqs - _k * _widths, _CONSENSUS_MIN_FREQUENCY_BOUND
                )
                _uppers = _freqs + _k * _widths
                _global_lower = float(_lowers.min())
                _global_upper = float(_uppers.max())
                _constraint_dict[_keys["mixture_means"]] = Interval(
                    _global_lower, _global_upper
                )
                _frequency_constraint_bounds = (_global_lower, _global_upper)

            _freqs_arr = np.asarray(
                consensus_frequencies, dtype=float
            ).ravel()
            _scale_upper = (
                float(consensus_scale_max_factor) * float(np.median(_freqs_arr))
            )
            if not (np.isfinite(_scale_upper) and _scale_upper > 0):
                _msg = (
                    "consensus_scale_max_factor * median(consensus_frequencies)"
                    f" must be positive and finite (got {_scale_upper})."
                )
                raise ValueError(_msg)
            # Practical lower bound - scales must be positive.
            _constraint_dict[_keys["mixture_scales"]] = Interval(
                _CONSENSUS_MIN_SCALE_BOUND, _scale_upper
            )
            # --- Step 3: apply consensus constraints on top of defaults ------
            # These must win over the defaults applied in step 1.
            if _constraint_dict:
                self.set_constraint(_constraint_dict)
            # --- Step 4: mark constraints as set so _fit_core skips defaults -
            # set_default_constraints already set this flag in step 1, but we
            # re-assert it here to make the intent explicit and guard against
            # future refactors that might reorder the steps.
            self.__CONTRAINTS_SET = True
            result_diagnostics[
                "constraints_marked_set_after_consensus"
            ] = True
            # --- Step 5: validate that the consensus constraint took effect --
            self._consensus_validate_applied_sm_constraints(
                keys=_keys,
                consensus_frequencies=consensus_frequencies,
                frequency_bounds=_frequency_constraint_bounds,
            )
            # --- Step 6: record constraint-handoff diagnostics ---------------
            result_diagnostics["consensus_constraints_applied"] = True
            result_diagnostics["consensus_constraint_bounds"] = (
                list(_frequency_constraint_bounds)
                if _frequency_constraint_bounds is not None
                else None
            )
            result_diagnostics["consensus_constraint_target_key"] = (
                _keys.get("mixture_means")
            )
            result_diagnostics["consensus_scale_constraint_bounds"] = [
                float(_CONSENSUS_MIN_SCALE_BOUND), float(_scale_upper)
            ]
            result_diagnostics["consensus_scale_constraint_target_key"] = (
                _keys.get("mixture_scales")
            )
        else:
            _frequency_constraint_bounds = None
            _scale_upper = None
            result_diagnostics["consensus_constraints_applied"] = False
            result_diagnostics["default_constraints_applied_before_consensus"] = (
                False
            )
            result_diagnostics["constraints_marked_set_after_consensus"] = False


        consensus_guess = self._consensus_build_guess(
            frequencies=consensus_frequencies,
            scales=consensus_scales,
        )

        self._last_consensus_fit_info = {
            "fit_strategy": "consensus",
            "consensus_frequencies": np.asarray(
                consensus_frequencies, dtype=float
            ).ravel().tolist(),
            "consensus_scales": (
                None
                if consensus_scales is None
                else np.asarray(consensus_scales, dtype=float).ravel().tolist()
            ),
            "consensus_frequency_width": (
                None
                if consensus_frequency_width is None
                else np.asarray(consensus_frequency_width, dtype=float).ravel().tolist()
            ),
            "apply_consensus_constraints": bool(apply_consensus_constraints),
            "consensus_frequency_bounds": (
                _frequency_constraint_bounds
                if _frequency_constraint_bounds is not None
                else auto_constraint_bounds
            ),
            "consensus_scale_upper": (
                float(_scale_upper) if _scale_upper is not None else None
            ),
        }

        merged_guess = {}
        if user_guess is not None:
            merged_guess.update(user_guess)
        merged_guess.update(consensus_guess)

        fit_kwargs["guess"] = merged_guess
        fit_kwargs["fit_strategy"] = None

        _consensus_frequency_scalar = result_diagnostics.get("consensus_frequency")
        if _consensus_frequency_scalar is None:
            _consensus_frequency_scalar = result_diagnostics.get(
                "final_consensus_frequency"
            )
        _consensus_frequency_scalar_ready = _consensus_frequency_scalar is not None

        _consensus_frequencies_raw = result_diagnostics.get("consensus_frequencies")
        _consensus_frequencies_input = (
            _consensus_frequencies_raw
            if _consensus_frequencies_raw is not None
            else []
        )
        _consensus_frequencies_arr = np.asarray(
            _consensus_frequencies_input,
            dtype=float,
        ).ravel()
        _consensus_frequency_ready = bool(
            _consensus_frequencies_arr.size > 0
            and np.all(
                np.isfinite(_consensus_frequencies_arr)
                & (_consensus_frequencies_arr > 0)
            )
        )
        _trusted_candidate_count = result_diagnostics.get("trusted_candidate_count")
        _trusted_candidate_ready = bool(
            _trusted_candidate_count is not None
            and int(_trusted_candidate_count) > 0
        )
        _consensus_init_ready = bool(consensus_guess)
        _consensus_ready_for_success = bool(
            (_consensus_frequency_scalar_ready or _consensus_frequency_ready)
            and _trusted_candidate_ready
            and _consensus_init_ready
        )

        self.consensus_diagnostics = self._consensus_finalize_result_structure(
            result_diagnostics
        )
        try:
            fit_result = self.fit(**fit_kwargs)
        except Exception:
            result_diagnostics["consensus_success"] = False
            # validate=False prevents masking the original exception when
            # partial error-recovery diagnostics are finalized.  Validation
            # is skipped here because diagnostics may be incomplete during
            # exception handling, and the original error is more important.
            self.consensus_diagnostics = self._consensus_finalize_result_structure(
                result_diagnostics, validate=False
            )
            raise

        if apply_consensus_constraints:
            _final_bounds = self._consensus_get_registered_sm_constraint_bounds(_keys)
            result_diagnostics["consensus_constraint_bounds_final"] = _final_bounds
            self._consensus_validate_applied_sm_constraints(
                keys=_keys,
                consensus_frequencies=consensus_frequencies,
                frequency_bounds=_frequency_constraint_bounds,
            )

        result_diagnostics["consensus_success"] = _consensus_ready_for_success
        self.consensus_diagnostics = self._consensus_finalize_result_structure(
            result_diagnostics
        )
        return fit_result

    def _consensus_multicomp_fit(self, **fit_kwargs):
        """Run the staged multi-component consensus fit workflow.

        For ``fit_strategy="consensus_multicomp"``, this method now performs:
        1) per-band multi-component LS candidate extraction,
        2) cross-band clustering in frequency space,
        3) accepted-cluster multi-component consensus aggregation,
        4) initialization of a multi-component 2D spectral-mixture fit, and
        5) dispatch to the existing fit backend.

        The final fit currently reuses the existing global constraint system,
        so all accepted components share one broad frequency interval.
        Component-specific mixture constraints remain future work.
        """
        user_guess = fit_kwargs.pop("guess", None)
        consensus_frequency_k = fit_kwargs.pop("consensus_frequency_k", 3.0)
        consensus_scale_max_factor = fit_kwargs.pop("consensus_scale_max_factor", 0.2)
        legacy_apply_constraints = fit_kwargs.pop("apply_consensus_constraints", None)
        constrain_consensus = fit_kwargs.pop("constrain_consensus", None)
        if constrain_consensus is None:
            apply_consensus_constraints = (
                True
                if legacy_apply_constraints is None
                else bool(legacy_apply_constraints)
            )
        else:
            apply_consensus_constraints = bool(constrain_consensus)

        min_points_per_band = fit_kwargs.pop("min_points_per_band", None)
        max_gap_fraction = fit_kwargs.pop("max_gap_fraction", None)
        min_duty_cycle = fit_kwargs.pop("min_duty_cycle", None)
        max_components_per_band = int(fit_kwargs.pop("max_components_per_band", 3))
        cluster_frequency_rtol = float(fit_kwargs.pop("cluster_frequency_rtol", 0.10))
        min_bands_per_component = int(fit_kwargs.pop("min_bands_per_component", 2))
        min_width_fraction = float(fit_kwargs.pop("min_width_fraction", 0.05))
        drift_warning_fraction = float(
            fit_kwargs.pop(
                "drift_warning_fraction",
                _CONSENSUS_MULTICOMP_DRIFT_WARNING_FRACTION,
            )
        )
        verbose = fit_kwargs.get("verbose", False)
        _allow_existing = fit_kwargs.pop("_allow_existing_model_for_consensus", False)
        # Capture the user-requested mixture count before any mutation.
        # This value is authoritative: the final GP model must use exactly
        # requested_num_mixtures components when the user specifies it.
        requested_num_mixtures = fit_kwargs.get("num_mixtures", None)

        if self.ndim != 2:
            raise ValueError(
                "Automatic multi-component consensus construction requires a 2D "
                "light curve (multiband time+wavelength data)."
            )
        if max_components_per_band < 1:
            raise ValueError("max_components_per_band must be >= 1.")
        if not np.isfinite(cluster_frequency_rtol) or cluster_frequency_rtol < 0:
            raise ValueError(
                "cluster_frequency_rtol must be a finite, non-negative float."
            )
        if min_bands_per_component < 1:
            raise ValueError("min_bands_per_component must be >= 1.")
        if not np.isfinite(min_width_fraction) or min_width_fraction <= 0:
            raise ValueError("min_width_fraction must be a finite, positive float.")
        if not np.isfinite(drift_warning_fraction) or drift_warning_fraction < 0:
            raise ValueError(
                "drift_warning_fraction must be a finite, non-negative float."
            )

        result_diagnostics = self._consensus_initialize_result_structure(
            fit_strategy="consensus_multicomp"
        )
        result_diagnostics.update({
            "use_acf_validation": False,
            "use_gp_validation": False,
            "gp_validation_requested": False,
            "gp_validation_performed": False,
            "consensus_generation_method": "auto_multicomponent_consensus",
            "controls": {
                "min_points_per_band": min_points_per_band,
                "max_gap_fraction": max_gap_fraction,
                "min_duty_cycle": min_duty_cycle,
                "max_components_per_band": max_components_per_band,
                "cluster_frequency_rtol": float(cluster_frequency_rtol),
                "min_bands_per_component": min_bands_per_component,
                "min_width_fraction": float(min_width_fraction),
                "drift_warning_fraction": float(drift_warning_fraction),
                "constrain_consensus": bool(apply_consensus_constraints),
            },
            "constraint_strategy": (
                "global_frequency_interval"
                if apply_consensus_constraints
                else "constraints_disabled"
            ),
            "drift_warning_fraction": float(drift_warning_fraction),
        })
        self.consensus_diagnostics = self._consensus_finalize_result_structure(
            result_diagnostics
        )

        band_component_candidates = self._consensus_collect_band_component_candidates(
            max_components_per_band=max_components_per_band,
            min_points_per_band=min_points_per_band,
            max_gap_fraction=max_gap_fraction,
            min_duty_cycle=min_duty_cycle,
            verbose=verbose,
        )
        accepted_bands = sorted(
            {
                str(entry.get("band_name"))
                for entry in band_component_candidates
                if isinstance(entry, dict) and entry.get("band_name") is not None
            }
        )
        per_band_diagnostics = {}
        for entry in band_component_candidates:
            if not isinstance(entry, dict):
                continue
            band_name = entry.get("band_name")
            if band_name is None:
                continue
            record = self._consensus_initialize_band_record(band_name)
            self._consensus_set_band_status(
                record,
                _CONSENSUS_BAND_STATUS_ACCEPTED,
                [],
            )
            component_candidates = list(entry.get("component_candidates") or [])
            record["component_candidates"] = component_candidates
            if component_candidates:
                dominant_candidate = component_candidates[0]
                if isinstance(dominant_candidate, dict):
                    record["dominant_frequency"] = dominant_candidate.get("frequency")
                    record["dominant_period"] = dominant_candidate.get("period")
                    record["ls_significant"] = dominant_candidate.get("significant")
                    record["ls_peak_power"] = dominant_candidate.get("peak_power")
                    record["ls_peak_prominence"] = dominant_candidate.get(
                        "peak_prominence"
                    )
                    record["selected_from"] = "multicomponent_candidates"
            per_band_diagnostics[str(band_name)] = self._consensus_make_json_safe(
                record
            )

        result_diagnostics.update({
            "accepted_bands": accepted_bands,
            "rejected_bands": [],
            "rejection_reasons": {},
            "per_band_diagnostics": per_band_diagnostics,
            "band_component_candidates": self._consensus_make_json_safe(
                band_component_candidates
            ),
        })

        if not band_component_candidates:
            self.consensus_diagnostics = self._consensus_finalize_result_structure(
                result_diagnostics
            )
            raise ConsensusFitError(
                "Consensus multi-component fit failed: no accepted bands produced "
                "usable multi-component candidates. Every band was either rejected "
                "during quality gating or did not yield detectable multi-component "
                "frequency peaks. Check per-band sampling quality.",
                failure_diagnostics={
                    "status": "failed",
                    "reason": "no_multicomp_candidates",
                    "accepted_bands": result_diagnostics.get("accepted_bands", []),
                },
            )

        component_clusters = self._consensus_cluster_component_candidates(
            band_component_candidates,
            cluster_frequency_rtol=cluster_frequency_rtol,
            min_bands_per_component=min_bands_per_component,
        )
        result_diagnostics["component_clusters"] = self._consensus_make_json_safe(
            component_clusters
        )

        if not component_clusters:
            self.consensus_diagnostics = self._consensus_finalize_result_structure(
                result_diagnostics
            )
            raise ConsensusFitError(
                "Consensus multi-component fit failed: no component clusters "
                "could be formed from the extracted band candidates. The "
                "per-band frequency peaks may be too spread or too sparse to "
                "group into coherent multi-component clusters.",
                failure_diagnostics={
                    "status": "failed",
                    "reason": "no_component_clusters",
                    "accepted_bands": result_diagnostics.get("accepted_bands", []),
                },
            )

        multicomponent_consensus = self._consensus_build_multicomponent_frequency_consensus(
            component_clusters,
            min_width_fraction=min_width_fraction,
        )
        result_diagnostics["multicomponent_consensus"] = self._consensus_make_json_safe(
            multicomponent_consensus
        )

        accepted_clusters = [
            cluster for cluster in component_clusters if bool(cluster.get("accepted"))
        ]
        if not accepted_clusters:
            self.consensus_diagnostics = self._consensus_finalize_result_structure(
                result_diagnostics
            )
            raise ConsensusFitError(
                "Consensus multi-component fit failed: no accepted "
                "multicomponent consensus clusters were available. All "
                "candidate component clusters were rejected during the "
                "consensus quality evaluation.",
                failure_diagnostics={
                    "status": "failed",
                    "reason": "no_accepted_multicomp_clusters",
                    "n_clusters": len(component_clusters),
                    "accepted_bands": result_diagnostics.get("accepted_bands", []),
                },
            )

        initialization_payload = self._consensus_build_multicomponent_initialization(
            multicomponent_consensus
        )
        # M = accepted consensus component count (from clustering).
        n_components = initialization_payload["n_components"]

        # ------------------------------------------------------------------ #
        # Reconcile accepted components to the user-requested count N.        #
        # The user-requested num_mixtures is authoritative; consensus         #
        # clustering provides initialization information only.               #
        # ------------------------------------------------------------------ #
        accepted_comp_summaries = list(
            multicomponent_consensus.get("component_summaries") or []
        )
        rejected_clusters_for_fallback = list(
            multicomponent_consensus.get("rejected_clusters") or []
        )
        reconciled_summaries, reconciliation_diagnostics = (
            self._consensus_multicomp_reconcile_to_n_components(
                accepted_component_summaries=accepted_comp_summaries,
                requested_num_mixtures=requested_num_mixtures,
                rejected_clusters=rejected_clusters_for_fallback,
                band_component_candidates=band_component_candidates,
                min_width_fraction=min_width_fraction,
            )
        )
        # Update component arrays to reflect the reconciled count N.
        final_n = len(reconciled_summaries)
        consensus_frequencies = np.array(
            [float(s["consensus_frequency"]) for s in reconciled_summaries],
            dtype=float,
        )
        consensus_scales = np.array(
            [float(s.get("consensus_scale") or 1.0) for s in reconciled_summaries],
            dtype=float,
        )
        consensus_frequency_width = np.array(
            [float(s["consensus_frequency_width"]) for s in reconciled_summaries],
            dtype=float,
        )
        n_components = final_n

        consensus_periods = 1.0 / consensus_frequencies
        consensus_period_widths = consensus_frequency_width / (consensus_frequencies**2)
        if (
            consensus_frequencies.size < int(n_components)
            or consensus_frequency_width.size < int(n_components)
            or consensus_scales.size < int(n_components)
            or consensus_periods.size < int(n_components)
            or consensus_period_widths.size < int(n_components)
        ):
            raise RuntimeError(
                "Consensus multi-component diagnostics failed: component vector "
                "length mismatch after component-count reconciliation."
            )
        # Primary component is canonical component_index==0 by cluster order.
        primary_component_index = 0 if int(n_components) > 0 else None

        # Use the reconciled (user-requested) component count for the final model.
        fit_kwargs["num_mixtures"] = n_components
        mad_frequency_scatter = float(
            np.median(
                np.abs(consensus_frequencies - np.median(consensus_frequencies))
            )
        )
        trusted_candidate_count = int(
            sum(
                len(cluster.get("members") or [])
                for cluster in accepted_clusters
                if isinstance(cluster, dict)
            )
        )
        result_diagnostics.update({
            "n_components": int(n_components),
            "candidate_count": int(
                sum(
                    len(entry.get("component_candidates") or [])
                    for entry in band_component_candidates
                    if isinstance(entry, dict)
                )
            ),
            "trusted_candidate_count": trusted_candidate_count,
            "consensus_frequency": None,
            "consensus_period": None,
            "consensus_frequency_width": None,
            "consensus_frequency_scatter": mad_frequency_scatter,
            "median_frequency": None,
            "mad_frequency_scatter": mad_frequency_scatter,
            "final_consensus_frequency": None,
            "final_consensus_period": None,
            "robust_frequency_width": None,
            "consensus_frequencies": consensus_frequencies.tolist(),
            "consensus_frequency_widths": consensus_frequency_width.tolist(),
            "consensus_periods": consensus_periods.tolist(),
            "consensus_period_widths": consensus_period_widths.tolist(),
            "consensus_scales": consensus_scales.tolist(),
            "consensus_component_strengths": consensus_scales.tolist(),
            "consensus_mixture_init_scales": consensus_frequency_width.tolist(),
            "primary_component_index": primary_component_index,
            "primary_consensus_frequency": (
                float(consensus_frequencies[0]) if primary_component_index == 0 else None
            ),
            "primary_consensus_period": (
                float(consensus_periods[0]) if primary_component_index == 0 else None
            ),
        })
        # Add component-count reconciliation diagnostics.
        result_diagnostics.update(reconciliation_diagnostics)

        _requested_model = fit_kwargs.get("model")
        if _requested_model is not None:
            self._consensus_clear_model_state()
        elif not _allow_existing:
            raise ConsensusFitError(
                "Consensus fit requires an explicit final model. "
                "Pass model='2D' or another spectral-mixture-compatible "
                "model. Pre-existing model reuse is disabled by default "
                "to prevent stale consensus constraints."
            )

        _set_model_excluded = {
            "model",
            "likelihood",
            "num_mixtures",
            "variance",
            "guess",
            "consensus_frequency_k",
            "consensus_scale_max_factor",
            "apply_consensus_constraints",
            "constrain_consensus",
            "min_points_per_band",
            "max_gap_fraction",
            "min_duty_cycle",
            "max_components_per_band",
            "cluster_frequency_rtol",
            "min_bands_per_component",
            "min_width_fraction",
            "periods",
            "use_mls_init",
            "use_best_band_init",
            "constraint_set",
            "grid_size",
            "cuda",
            "training_iter",
            "max_cg_iterations",
            "optim",
            "miniter",
            "stop",
            "lr",
            "stopavg",
            "fit_strategy",
            "verbose",
            "_allow_existing_model_for_consensus",
        }
        _model_needs_build = (
            _requested_model is not None
            or not (
                hasattr(self, "model")
                and self.model is not None
                and hasattr(self, "_model_pars")
            )
        )
        if _model_needs_build:
            set_model_kwargs = {
                key: value
                for key, value in fit_kwargs.items()
                if key not in _set_model_excluded
            }
            self.set_model(
                _requested_model,
                fit_kwargs.get("likelihood"),
                num_mixtures=fit_kwargs.get("num_mixtures"),
                variance=fit_kwargs.get("variance", False),
                **set_model_kwargs,
            )
        fit_kwargs["model"] = None

        if apply_consensus_constraints:
            self._consensus_validate_final_model_supports_sm_time_kernel(
                model_name=_requested_model,
                time_kernel_type=fit_kwargs.get("time_kernel_type"),
            )
            _constraint_dict = {}
            _keys = self._consensus_resolve_time_spectral_mixture_keys()
            _constraint_set_for_defaults = fit_kwargs.get("constraint_set")
            self.set_default_constraints(
                constraint_set=_constraint_set_for_defaults
            )
            result_diagnostics[
                "default_constraints_applied_before_consensus"
            ] = True

            _freqs = np.asarray(consensus_frequencies, dtype=float).ravel()
            _widths = np.asarray(consensus_frequency_width, dtype=float).ravel()
            _k = float(consensus_frequency_k)
            _lowers = np.maximum(
                _freqs - _k * _widths,
                _CONSENSUS_MIN_FREQUENCY_BOUND,
            )
            _uppers = _freqs + _k * _widths
            _global_lower = float(_lowers.min())
            _global_upper = float(_uppers.max())
            _frequency_constraint_bounds = (_global_lower, _global_upper)
            _constraint_dict[_keys["mixture_means"]] = Interval(
                _global_lower,
                _global_upper,
            )

            _scale_upper = (
                float(consensus_scale_max_factor) * float(np.median(_freqs))
            )
            if not (np.isfinite(_scale_upper) and _scale_upper > 0):
                raise ValueError(
                    "consensus_scale_max_factor * median(consensus_frequencies) "
                    f"must be positive and finite (got {_scale_upper})."
                )
            _constraint_dict[_keys["mixture_scales"]] = Interval(
                _CONSENSUS_MIN_SCALE_BOUND,
                _scale_upper,
            )
            self.set_constraint(_constraint_dict)
            self.__CONTRAINTS_SET = True
            result_diagnostics[
                "constraints_marked_set_after_consensus"
            ] = True
            self._consensus_validate_applied_sm_constraints(
                keys=_keys,
                consensus_frequencies=consensus_frequencies,
                frequency_bounds=_frequency_constraint_bounds,
            )
            result_diagnostics["consensus_constraints_applied"] = True
            result_diagnostics["consensus_constraint_bounds"] = list(
                _frequency_constraint_bounds
            )
            result_diagnostics["consensus_constraint_target_key"] = (
                _keys.get("mixture_means")
            )
            result_diagnostics["consensus_scale_constraint_bounds"] = [
                float(_CONSENSUS_MIN_SCALE_BOUND),
                float(_scale_upper),
            ]
            result_diagnostics["consensus_scale_constraint_target_key"] = (
                _keys.get("mixture_scales")
            )
            result_diagnostics["final_constraint_bounds"] = list(
                _frequency_constraint_bounds
            )
        else:
            _frequency_constraint_bounds = None
            _scale_upper = None
            result_diagnostics["consensus_constraints_applied"] = False
            result_diagnostics["default_constraints_applied_before_consensus"] = (
                False
            )
            result_diagnostics["constraints_marked_set_after_consensus"] = False

        consensus_guess = self._consensus_build_guess(
            frequencies=consensus_frequencies,
            scales=consensus_frequency_width,
        )
        initialization_diagnostics = self._consensus_collect_initialization_diagnostics(
            requested_consensus_frequencies=consensus_frequencies,
            requested_consensus_scales=consensus_frequency_width,
            consensus_guess=consensus_guess,
        )
        initialization_diagnostics["requested_consensus_frequency_widths"] = (
            np.asarray(consensus_frequency_width, dtype=float).ravel().tolist()
        )
        initialization_diagnostics["requested_consensus_periods"] = (
            np.asarray(consensus_periods, dtype=float).ravel().tolist()
        )
        initialization_diagnostics["requested_consensus_period_widths"] = (
            np.asarray(consensus_period_widths, dtype=float).ravel().tolist()
        )
        result_diagnostics.update(initialization_diagnostics)
        initialized_mixture_means = np.asarray(
            initialization_diagnostics.get("initialized_mixture_means", []), dtype=float
        ).ravel()
        initialized_mixture_scales = np.asarray(
            initialization_diagnostics.get("initialized_mixture_scales", []), dtype=float
        ).ravel()
        initialized_mixture_periods = np.asarray(
            initialization_diagnostics.get("initialized_mixture_periods", []), dtype=float
        ).ravel()
        initialized_mixture_period_widths = np.asarray(
            initialization_diagnostics.get("initialized_mixture_period_widths", []),
            dtype=float,
        ).ravel()
        # Use the reconciled component summaries (may differ from the original
        # multicomponent_consensus["component_summaries"] when N != M).
        component_summaries = reconciled_summaries
        period_summaries = []
        for component_index in range(int(n_components)):
            source_summary = (
                component_summaries[component_index]
                if component_index < len(component_summaries)
                and isinstance(component_summaries[component_index], dict)
                else {}
            )
            period_summaries.append(
                {
                    "component_index": component_index,
                    "component_source": source_summary.get(
                        "component_source", "accepted_consensus"
                    ),
                    "source_cluster_id": source_summary.get("source_cluster_id"),
                    "consensus_frequency": float(consensus_frequencies[component_index]),
                    "consensus_period": float(consensus_periods[component_index]),
                    "consensus_frequency_width": float(
                        consensus_frequency_width[component_index]
                    ),
                    "consensus_period_width": float(
                        consensus_period_widths[component_index]
                    ),
                    "consensus_component_strength": float(
                        consensus_scales[component_index]
                    ),
                    "consensus_mixture_init_scale": float(
                        consensus_frequency_width[component_index]
                    ),
                    "initialized_mixture_mean": (
                        float(initialized_mixture_means[component_index])
                        if component_index < initialized_mixture_means.size
                        else None
                    ),
                    "initialized_mixture_period": (
                        float(initialized_mixture_periods[component_index])
                        if component_index < initialized_mixture_periods.size
                        else None
                    ),
                    "initialized_mixture_scale": (
                        float(initialized_mixture_scales[component_index])
                        if component_index < initialized_mixture_scales.size
                        else None
                    ),
                    "initialized_mixture_period_width": (
                        float(initialized_mixture_period_widths[component_index])
                        if component_index < initialized_mixture_period_widths.size
                        else None
                    ),
                    "fitted_mixture_frequency": None,
                    "fitted_mixture_period": None,
                    "fitted_mixture_scale": None,
                    "fitted_mixture_period_width": None,
                    "fitted_frequency_shift_from_initialization": None,
                    "fitted_period_shift_from_initialization": None,
                    "fitted_fractional_frequency_shift_from_initialization": None,
                    "fitted_fractional_period_shift_from_initialization": None,
                    "fitted_abs_fractional_period_shift_from_initialization": None,
                    "fitted_abs_fractional_frequency_shift_from_initialization": None,
                    "fitted_period_drift_flag": None,
                    "fitted_frequency_drift_flag": None,
                    "nearest_initialized_component_index": None,
                    "nearest_initialized_component_fractional_period_distance": None,
                    "nearest_initialized_component_fractional_frequency_distance": None,
                    "component_identity_preserved": None,
                    "member_bands": list(source_summary.get("member_bands") or []),
                    "n_member_bands": source_summary.get("n_member_bands"),
                }
            )
        result_diagnostics["multicomponent_period_summaries"] = (
            self._consensus_make_json_safe(period_summaries)
        )
        self._last_consensus_fit_info = {
            "fit_strategy": "consensus_multicomp",
            "consensus_frequencies": consensus_frequencies.ravel().tolist(),
            "consensus_periods": consensus_periods.ravel().tolist(),
            "consensus_scales": consensus_scales.ravel().tolist(),
            "consensus_component_strengths": consensus_scales.ravel().tolist(),
            "consensus_frequency_width": consensus_frequency_width.ravel().tolist(),
            "consensus_frequency_widths": consensus_frequency_width.ravel().tolist(),
            "consensus_period_widths": consensus_period_widths.ravel().tolist(),
            "consensus_mixture_init_scales": consensus_frequency_width.ravel().tolist(),
            "apply_consensus_constraints": bool(apply_consensus_constraints),
            "consensus_frequency_bounds": _frequency_constraint_bounds,
            "consensus_scale_upper": (
                float(_scale_upper) if _scale_upper is not None else None
            ),
            "constraint_strategy": result_diagnostics.get("constraint_strategy"),
            "initialization_strategy": result_diagnostics.get(
                "initialization_strategy"
            ),
        }

        merged_guess = {}
        if user_guess is not None:
            merged_guess.update(user_guess)
        merged_guess.update(consensus_guess)

        fit_kwargs["guess"] = merged_guess
        fit_kwargs["fit_strategy"] = None

        self.consensus_diagnostics = self._consensus_finalize_result_structure(
            result_diagnostics
        )
        try:
            fit_result = self.fit(**fit_kwargs)
        except Exception:
            result_diagnostics["consensus_success"] = False
            self.consensus_diagnostics = self._consensus_finalize_result_structure(
                result_diagnostics, validate=False
            )
            raise

        if apply_consensus_constraints:
            _final_bounds = self._consensus_get_registered_sm_constraint_bounds(_keys)
            result_diagnostics["consensus_constraint_bounds_final"] = _final_bounds
            result_diagnostics["final_constraint_bounds"] = _final_bounds
            self._consensus_validate_applied_sm_constraints(
                keys=_keys,
                consensus_frequencies=consensus_frequencies,
                frequency_bounds=_frequency_constraint_bounds,
            )

        try:
            fitted_diagnostics = self._consensus_collect_fitted_mixture_diagnostics(
                initialized_mixture_frequencies=initialized_mixture_means,
                initialized_mixture_scales=initialized_mixture_scales,
            )
            fitted_frequencies = np.asarray(
                fitted_diagnostics.get("fitted_mixture_frequencies", []), dtype=float
            ).ravel()
            fitted_periods = np.asarray(
                fitted_diagnostics.get("fitted_mixture_periods", []), dtype=float
            ).ravel()
            fitted_scales = np.asarray(
                fitted_diagnostics.get("fitted_mixture_scales", []), dtype=float
            ).ravel()
            fitted_period_widths = np.asarray(
                fitted_diagnostics.get("fitted_mixture_period_widths", []), dtype=float
            ).ravel()
            fitted_frequency_shift = np.asarray(
                fitted_diagnostics.get(
                    "fitted_frequency_shift_from_initialization", []
                ),
                dtype=float,
            ).ravel()
            fitted_period_shift = np.asarray(
                fitted_diagnostics.get("fitted_period_shift_from_initialization", []),
                dtype=float,
            ).ravel()
            fitted_fractional_frequency_shift = np.asarray(
                fitted_diagnostics.get(
                    "fitted_fractional_frequency_shift_from_initialization", []
                ),
                dtype=float,
            ).ravel()
            fitted_fractional_period_shift = np.asarray(
                fitted_diagnostics.get(
                    "fitted_fractional_period_shift_from_initialization", []
                ),
                dtype=float,
            ).ravel()
            expected_size = int(initialized_mixture_means.size)
            diagnostics_vectors = {
                "fitted_mixture_frequencies": fitted_frequencies,
                "fitted_mixture_periods": fitted_periods,
                "fitted_mixture_scales": fitted_scales,
                "fitted_mixture_period_widths": fitted_period_widths,
                "fitted_frequency_shift_from_initialization": fitted_frequency_shift,
                "fitted_period_shift_from_initialization": fitted_period_shift,
                "fitted_fractional_frequency_shift_from_initialization": (
                    fitted_fractional_frequency_shift
                ),
                "fitted_fractional_period_shift_from_initialization": (
                    fitted_fractional_period_shift
                ),
            }
            for diagnostic_key, diagnostic_values in diagnostics_vectors.items():
                if diagnostic_values.size != expected_size:
                    raise RuntimeError(
                        "Consensus multi-component fitted diagnostics mismatch: "
                        f"len({diagnostic_key})={int(diagnostic_values.size)} does "
                        "not match "
                        "len(initialized_mixture_means)="
                        f"{expected_size}."
                    )

            result_diagnostics.update(fitted_diagnostics)
            fitted_abs_fractional_period_shift = np.abs(fitted_fractional_period_shift)
            fitted_abs_fractional_frequency_shift = np.abs(
                fitted_fractional_frequency_shift
            )
            period_drift_flags = (
                fitted_abs_fractional_period_shift >= float(drift_warning_fraction)
            )
            frequency_drift_flags = (
                fitted_abs_fractional_frequency_shift >= float(drift_warning_fraction)
            )
            result_diagnostics[
                "max_abs_fractional_period_shift_from_initialization"
            ] = float(np.max(fitted_abs_fractional_period_shift))
            result_diagnostics[
                "max_abs_fractional_frequency_shift_from_initialization"
            ] = float(np.max(fitted_abs_fractional_frequency_shift))
            result_diagnostics["component_fit_drift_flags"] = [
                bool(v) for v in period_drift_flags.tolist()
            ]
            result_diagnostics["components_with_large_period_drift"] = [
                int(idx) for idx in np.flatnonzero(period_drift_flags).tolist()
            ]
            result_diagnostics["components_with_large_frequency_drift"] = [
                int(idx) for idx in np.flatnonzero(frequency_drift_flags).tolist()
            ]
            result_diagnostics["drift_warning_fraction"] = float(drift_warning_fraction)
            component_identity_diagnostics = (
                self._consensus_compute_component_identity_diagnostics(
                    initialized_periods=initialized_mixture_periods,
                    fitted_periods=fitted_periods,
                    initialized_frequencies=initialized_mixture_means,
                    fitted_frequencies=fitted_frequencies,
                )
            )
            result_diagnostics.update(component_identity_diagnostics)

            for component_index in range(expected_size):
                period_summaries[component_index]["fitted_mixture_frequency"] = float(
                    fitted_frequencies[component_index]
                )
                period_summaries[component_index]["fitted_mixture_period"] = float(
                    fitted_periods[component_index]
                )
                period_summaries[component_index]["fitted_mixture_scale"] = float(
                    fitted_scales[component_index]
                )
                period_summaries[component_index][
                    "fitted_mixture_period_width"
                ] = float(fitted_period_widths[component_index])
                period_summaries[component_index][
                    "fitted_frequency_shift_from_initialization"
                ] = float(fitted_frequency_shift[component_index])
                period_summaries[component_index][
                    "fitted_period_shift_from_initialization"
                ] = float(fitted_period_shift[component_index])
                period_summaries[component_index][
                    "fitted_fractional_frequency_shift_from_initialization"
                ] = float(fitted_fractional_frequency_shift[component_index])
                period_summaries[component_index][
                    "fitted_fractional_period_shift_from_initialization"
                ] = float(fitted_fractional_period_shift[component_index])
                period_summaries[component_index][
                    "fitted_abs_fractional_period_shift_from_initialization"
                ] = float(fitted_abs_fractional_period_shift[component_index])
                period_summaries[component_index][
                    "fitted_abs_fractional_frequency_shift_from_initialization"
                ] = float(fitted_abs_fractional_frequency_shift[component_index])
                period_summaries[component_index]["fitted_period_drift_flag"] = bool(
                    period_drift_flags[component_index]
                )
                period_summaries[component_index][
                    "fitted_frequency_drift_flag"
                ] = bool(frequency_drift_flags[component_index])
                period_summaries[component_index][
                    "nearest_initialized_component_index"
                ] = int(
                    component_identity_diagnostics[
                        "nearest_initialized_component_index"
                    ][component_index]
                )
                period_summaries[component_index][
                    "nearest_initialized_component_fractional_period_distance"
                ] = float(
                    component_identity_diagnostics[
                        "nearest_initialized_component_fractional_period_distance"
                    ][component_index]
                )
                nearest_frequency_distance = component_identity_diagnostics.get(
                    "nearest_initialized_component_fractional_frequency_distance"
                ) or []
                period_summaries[component_index][
                    "nearest_initialized_component_fractional_frequency_distance"
                ] = (
                    float(nearest_frequency_distance[component_index])
                    if component_index < len(nearest_frequency_distance)
                    else None
                )
                period_summaries[component_index][
                    "component_identity_preserved"
                ] = bool(
                    component_identity_diagnostics["component_identity_preserved"][
                        component_index
                    ]
                )

            result_diagnostics["multicomponent_period_summaries"] = (
                self._consensus_make_json_safe(period_summaries)
            )
        except Exception:
            result_diagnostics["consensus_success"] = False
            self.consensus_diagnostics = self._consensus_finalize_result_structure(
                result_diagnostics, validate=False
            )
            raise

        result_diagnostics["consensus_success"] = True
        result_diagnostics["fitted_num_mixtures"] = int(n_components)
        self.consensus_diagnostics = self._consensus_finalize_result_structure(
            result_diagnostics
        )
        return fit_result

    def _consensus_relaxed_fit(self, **fit_kwargs):
        """Frequency-space consensus-fit stub for relaxed-consensus behavior."""
        raise NotImplementedError(
            "fit_strategy='consensus_relaxed' is not implemented yet."
        )


    def mcmc(
        self,
        sampler=None,
        num_samples=500,
        warmup_steps=100,
        num_chains=1,
        disable_progbar=False,
        max_cg_iterations=None,
        cuda=False,
        **kwargs,
    ):
        """Run an MCMC sampler on the model

        This function runs an MCMC sampler on the model, using the sampler
        specified in the `sampler` attribute. The results are stored in the
        `mcmc_results` attribute.

        Parameters
        ----------
        sampler : str or MCMC, optional
            The name of the sampler to use. If None, pyro.infer.mcmc.NUTS will
            be used. If a string, it must be one of the following:
                'NUTS': pyro.infer.mcmc.NUTS
                'HMC': pyro.infer.mcmc.HMC
            Otherwise, it must be an instance of pyro.infer.mcmc.MCMC.
        num_samples : int, optional
            The number of samples to draw from the posterior, by default 500.
        warmup_steps : int, optional
            The number of warmup steps to use, by default 100.
        disable_progbar : bool, optional
            Whether to disable the progress bar, by default False.
        **kwargs : dict, optional

        Returns
        -------
        mcmc_results : dict
            A dictionary containing the results of the MCMC sampling. The
            keys are the names of the parameters, and the values are the
            samples of the parameters.
        """
        msg = "MCMC is not currently exposed. It will be available in future releases."
        raise NotImplementedError(msg)
        if sampler is None:
            sampler = NUTS
        elif isinstance(sampler, str):
            if sampler == "NUTS":
                sampler = NUTS
            elif sampler == "HMC":
                sampler = HMC
            else:
                raise ValueError("sampler must be one of 'NUTS' or 'HMC'")
        elif not isinstance(sampler, MCMC):
            raise TypeError(
                "sampler must be either None, a string, or an instance of "
                "pyro.infer.mcmc.MCMC"
            )

        # we need to make sure that the model is in train mode
        # self._train()
        # self._eval()

        if cuda:
            self.cuda()

        if not self.__PRIORS_SET:
            self.set_default_priors()

        if max_cg_iterations is None:
            max_cg_iterations = 10000

        model = self.model

        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)

        # def pyro_model(x, y):
        #     model.pyro_sample_from_prior()
        #     output = model(x)
        #     loss = mll.pyro_factor(output, y)
        #     return y

        def pyro_model(x, y):
            with (
                gpytorch.settings.fast_computations(False, False, False),
                gpytorch.settings.max_cg_iterations(max_cg_iterations),
            ):
                for key in self.state_dict().keys():
                    print(key)
                    with contextlib.suppress(AttributeError):
                        print(self.state_dict()[key].device)
                        # self.state_dict()[key] = self.state_dict()[key].cuda()
                for param_name, param in self.model.named_parameters():
                    print(
                        f"Parameter name: {param_name:42} value = {param.data}, "
                        f"device = {param.data.device}"
                    )
                print(self.model.covar_module.mixture_means)
                print(self.model.covar_module.mixture_scales)
                print(self.model.covar_module.mixture_weights)
                print(self.model.covar_module.mixture_means_prior)
                print("----")
                print("Lookup dict:")
                for param_name, param in self._model_pars.items():
                    print(param)
                    print("----")
                    # print(f'Parameter name: {param_name:42} value = {param["module"].value}, device = {param["module"].value.device}')  # noqa: E501
                sampled_model = model.pyro_sample_from_prior()  # .detatch()
                output = sampled_model.likelihood(sampled_model(x))  # .detatch()
                pyro.sample("obs", output, obs=y)
            return y

        self.num_samples = num_samples

        nuts_kernel = sampler(pyro_model)
        self.mcmc_run = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
            disable_progbar=disable_progbar,
        )
        import linear_operator.utils.errors as linear

        for key in self.state_dict().keys():
            print(key)
            with contextlib.suppress(AttributeError):
                print(self.state_dict()[key].device)
                # self.state_dict()[key] = self.state_dict()[key].cuda()
        for param_name, param in self.model.named_parameters():
            print(
                f"Parameter name: {param_name:42} value = {param.data}, "
                f"device = {param.data.device}"
            )

        try:
            if cuda:
                self.mcmc_run.run(
                    self._xdata_transformed.cuda(), self._ydata_transformed.cuda()
                )
            else:
                self.mcmc_run.run(self._xdata_transformed, self._ydata_transformed)
        except linear.NanError as e:
            print("NaNError encountered, returning None")
            print(list(model.named_parameters()))
            self.print_parameters()
            raise e

        self.__FITTED_MCMC = True

        # self.mcmc_run.summary(prob=0.683)
        self.inference_data = az.from_pyro(self.mcmc_run)
        samples = self.mcmc_run.get_samples()
        self.model.pyro_load_from_samples(samples)

        self.post = self.inference_data.posterior
        transformed_mcmc_periods = 1 / self.post["covar_module.mixture_means_prior"]
        raw_mcmc_periods = self.xtransform.inverse(
            torch.as_tensor(transformed_mcmc_periods.to_numpy()), shift=False
        )
        raw_mcmc_frequencies = 1 / raw_mcmc_periods
        transformed_mcmc_period_scales = 1 / (
            2 * torch.pi * self.post["covar_module.mixture_scales_prior"]
        )
        raw_mcmc_period_scales = self.xtransform.inverse(
            torch.as_tensor(transformed_mcmc_period_scales.to_numpy()), shift=False
        )
        raw_mcmc_frequency_scales = 1 / (2 * torch.pi * raw_mcmc_period_scales)

        self.inference_data.posterior["transformed_periods"] = transformed_mcmc_periods
        self.inference_data.posterior["raw_periods"] = xr.DataArray(
            raw_mcmc_periods.reshape(
                self.post["covar_module.mixture_means_prior"].shape
            ),
            coords=self.inference_data.posterior[
                "covar_module.mixture_means_prior"
            ].indexes,
        )
        self.inference_data.posterior["raw_frequencies"] = xr.DataArray(
            raw_mcmc_frequencies.reshape(
                self.post["covar_module.mixture_means_prior"].shape
            ),
            coords=self.inference_data.posterior[
                "covar_module.mixture_means_prior"
            ].indexes,
        )
        self.inference_data.posterior["transformed_period_scales"] = (
            transformed_mcmc_period_scales
        )
        self.inference_data.posterior["raw_period_scales"] = xr.DataArray(
            raw_mcmc_period_scales.reshape(
                self.post["covar_module.mixture_scales_prior"].shape
            ),
            coords=self.inference_data.posterior[
                "covar_module.mixture_scales_prior"
            ].indexes,
        )
        self.inference_data.posterior["raw_frequency_scales"] = xr.DataArray(
            raw_mcmc_frequency_scales.reshape(
                self.post["covar_module.mixture_scales_prior"].shape
            ),
            coords=self.inference_data.posterior[
                "covar_module.mixture_scales_prior"
            ].indexes,
        )

        # self.mcmc_results = mcmc(self, sampler, **kwargs)

    def summary(
        self,
        prob=0.683,
        use_arviz=True,
        var_names=None,
        filter_vars="like",
        stat_focus="median",
        **kwargs,
    ):
        """Print a summary of the results of the MCMC sampling

        Parameters
        ----------
        prob : float, optional
            The probability to use for the credible intervals, by default
            0.683.
        use_arviz : bool, optional
            Whether to use arviz to print the summary, by default True.
        var_names : list, optional
            A list of the names of the variables to include in the summary. If
            None, the variables for the mean function and the covariance
            function will be included, by default None.
        filter_vars : str, optional
            A string specifying how to filter the variables, based on
            `arviz.summary`. If None, the default behaviour of `arviz.summary`
            will be used, by default 'like'.
        stat_focus : str, optional
            A string specifying which statistic to focus on, based on
            `arviz.summary`. If None, the default behaviour of `arviz.summary`
            will be used ('mean'), by default 'median'.
        """
        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        msg = "MCMC is not currently exposed. It will be available in future releases."
        raise NotImplementedError(msg)
        if var_names is None:
            var_names = ["mean_module", "covar_module.mixture_weights", "raw"]
        elif var_names == "all":
            var_names = None
        if stat_focus is None:
            stat_focus = "mean"
        if use_arviz:
            self.summary = az.summary(
                self.inference_data,
                round_to=2,
                hdi_prob=prob,
                var_names=var_names,
                filter_vars=filter_vars,
                stat_focus=stat_focus,
                **kwargs,
            )
        # self.mcmc_run.summary(prob=prob)
        self.diagnostics = self.mcmc_run.diagnostics()
        # figure out how to filter these before printing!
        # print(self.diagnostics)
        return self.summary

    def plot_corner(
        self,
        kind="scatter",
        var_names=None,
        filter_vars="like",
        marginals=True,
        point_estimate="median",
        **kwargs,
    ):
        """Plot a corner plot of the results of the MCMC sampling

        Parameters
        ----------
        kind : str, optional
            The kind of plot to use, based on `arviz.plot_pair`. If None, the
            default behaviour of `arviz.plot_pair` will be used, by default
            'scatter'. Other options are 'kde' and 'hexbin'.
        var_names : list, optional
            A list of the names of the variables to include in the corner plot.
            If None, the variables for the mean function and the covariance
            function will be included, by default None.
        filter_vars : str, optional
            A string specifying how to filter the variables, based on
            `arviz.plot_pair`. If None, the default behaviour of
            `arviz.plot_pair` will be used, by default 'like'.
        marginals : bool, optional
            Whether to include the marginal distributions, by default True.
        point_estimate : str, optional
            The point estimate to plot, based on `arviz.plot_pair`. If None,
            no point estimate will be plotted, by default 'median'.
        """
        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        msg = "MCMC is not currently exposed. It will be available in future releases."
        raise NotImplementedError(msg)
        if var_names is None:
            var_names = ["mean_module", "covar_module.mixture_weights", "raw"]
        if point_estimate is None:
            point_estimate = "median"
        az.plot_pair(
            self.inference_data,
            kind=kind,
            var_names=var_names,
            filter_vars=filter_vars,
            marginals=marginals,
            point_estimate=point_estimate,
            **kwargs,
        )

    def plot_trace(self, var_names=None, filter_vars="like", figsize=None, **kwargs):
        """Plot a trace plot of the results of the MCMC sampling

        Parameters
        ----------
        var_names : list, optional
            A list of the names of the variables to include in the trace plot.
            If None, the variables for the mean function and the covariance
            function will be included, by default None.
        """
        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        msg = "MCMC is not currently exposed. It will be available in future releases."
        raise NotImplementedError(msg)
        if var_names is None:
            # we carefully choose the default variables to plot
            # we want to plot all parameters relating to the mean function
            # but we don't want to plot all the covariance parameters
            # because those are in the transformed space. Instead, we want
            # to plot the extra parameters we have created, which are in the
            # raw space, as well as the periods and the mixture weights
            var_names = [
                "mean_module",
                "covar_module.mixture_weights",
                "raw",
            ]  # ['mean_module', 'covar_module']
        az.plot_trace(
            self.inference_data,
            var_names=var_names,
            filter_vars=filter_vars,
            figsize=figsize,
            **kwargs,
        )

    def print_periods(self):
        if self.ndim == 1:
            for i in range(len(self.model.covar_module.mixture_means)):
                if self.xtransform is None:
                    p = 1 / self.model.covar_module.mixture_means[i]
                else:
                    p = (
                        self.xtransform.inverse(
                            1 / self.model.covar_module.mixture_means[i],
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                print(
                    f"Period {i}: "
                    f"{p}"
                    f" weight: {self.model.covar_module.mixture_weights[i]}"
                )
        elif self.ndim == 2:
            for i in range(len(self.model.covar_module.mixture_means[:, 0])):
                if self.xtransform is None:
                    p = 1 / self.model.covar_module.mixture_means[i, 0]
                else:
                    p = (
                        self.xtransform.inverse(
                            1 / self.model.covar_module.mixture_means[i, 0],
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0, 0]
                    )
                print(
                    f"Period {i}: "
                    f"{p}"
                    f" weight: {self.model.covar_module.mixture_weights[i]}"
                )

    def get_periods(self):
        """
        Returns a list of the periods, scales and weights of the model. This
        is useful for getting the periods after training, for example.
        """
        periods = []
        scales = []
        weights = []
        if self.ndim == 1:
            for i in range(len(self.model.sci_kernel.mixture_means)):
                if self.xtransform is None:
                    p = 1 / self.model.sci_kernel.mixture_means[i]
                    scales.append(
                        1 / (2 * torch.pi * self.model.sci_kernel.mixture_scales[i])
                    )
                else:
                    p = (
                        self.xtransform.inverse(
                            1 / self.model.sci_kernel.mixture_means[i],
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                    scales.append(
                        self.xtransform.inverse(
                            1
                            / (2 * torch.pi * self.model.sci_kernel.mixture_scales[i]),
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                periods.append(p)
                weights.append(
                    self.model.sci_kernel.mixture_weights[i].detach().numpy()
                )
        elif self.ndim == 2:
            for i in range(len(self.model.sci_kernel.mixture_means[:, 0])):
                if self.xtransform is None:
                    p = 1 / self.model.sci_kernel.mixture_means[i, 0]
                    scales.append(
                        1 / (2 * torch.pi * self.model.sci_kernel.mixture_scales[i, 0])
                    )
                else:
                    p = (
                        self.xtransform.inverse(
                            1 / self.model.sci_kernel.mixture_means[i, 0],
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0, 0]
                    )
                    scales.append(
                        self.xtransform.inverse(
                            1
                            / (
                                2
                                * torch.pi
                                * self.model.sci_kernel.mixture_scales[i, 0]
                            ),
                            shift=False,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0, 0]
                    )
                periods.append(p)
                weights.append(
                    self.model.sci_kernel.mixture_weights[i].detach().numpy()
                )

        weights = np.array([_to_numpy(w) for w in weights])
        periods = np.array([_to_numpy(p) for p in periods])
        scales = np.array([_to_numpy(s) for s in scales])

        return (
            torch.as_tensor(periods),
            torch.as_tensor(weights),
            torch.as_tensor(scales),
        )

    def _infer_num_mixtures_from_model(self):
        """Infer the number of spectral-mixture components from the current model.

        Walks the kernel tree of ``self.model.sci_kernel`` (including SKI
        and separable-2D wrappers) looking for a ``mixture_means`` attribute
        and returns ``len(mixture_means)``.  Falls back to inspecting
        ``mixture_scales`` or ``mixture_weights`` if ``mixture_means`` is
        unavailable.

        Returns ``None`` if the model does not expose mixture parameters or if
        ``self.model`` has not been initialised.

        Returns
        -------
        n_mix : int or None
        """
        if not hasattr(self, "model") or self.model is None:
            return None
        if not hasattr(self.model, "sci_kernel"):
            return None
        sk = self.model.sci_kernel
        # Unwrap GridInterpolationKernel (SKI)
        actual_sk = getattr(sk, "base_kernel", sk)
        # For separable 2D, look for the time sub-kernel
        if not hasattr(actual_sk, "mixture_means"):
            from gpytorch.kernels import ProductKernel

            if isinstance(actual_sk, ProductKernel):
                for k in actual_sk.kernels:
                    inner = getattr(k, "base_kernel", k)
                    if hasattr(inner, "mixture_means"):
                        actual_sk = inner
                        break
        # Try mixture_means first, then fallbacks
        for attr in ("mixture_means", "mixture_scales", "mixture_weights"):
            if hasattr(actual_sk, attr):
                try:
                    return len(getattr(actual_sk, attr))
                except (TypeError, RuntimeError):
                    pass
        return None

    def _extract_sm_params(self):
        """Extract raw spectral-mixture parameters in physical (data) units.

        Helper for :meth:`get_period_summary`.  Extracts per-component
        means, scales, and weights from ``self.model.sci_kernel`` (or its
        ``base_kernel`` if the sci_kernel is a
        :class:`~gpytorch.kernels.GridInterpolationKernel`) and converts
        them from the transformed (normalised) frequency space back to the
        original data units, following the same convention as
        :meth:`get_periods`.

        The conversions performed here are scientifically important:

        * If ``self.xtransform is None`` the model operates directly in the
          raw time units, so ``mixture_mean`` is already the raw frequency
          and ``mixture_scale`` is already the raw frequency scale.
        * If ``self.xtransform is not None`` the model was trained in a
          normalised time coordinate.  Frequencies and scales must be
          inverse-transformed (with ``shift=False``, i.e. scaling only)
          to recover quantities in the original time units.

        For both 1-D and 2-D spectral-mixture models the *time* dimension
        (index 0 of the last axis of ``mixture_means``) is used, consistent
        with :meth:`get_periods`.  The indexing ``[i, 0, 0]`` selects
        mixture component ``i``, collapses the redundant size-1 middle
        dimension, and picks time-dimension index 0 from the last axis.
        For a 1-D kernel the shape is ``(n_mix, 1, 1)``; for a 2-D kernel
        the shape is ``(n_mix, 1, 2)``, and index 0 of the last axis is
        always the time dimension.

        Returns
        -------
        params : dict
            Keys and values (all 1-D :class:`numpy.ndarray` of length
            ``num_mixtures``):

            * ``component_frequencies``    - raw centre frequencies
            * ``component_periods``        - raw centre periods
            * ``component_frequency_scales`` - Gaussian sigma in frequency
            * ``component_period_scales``  - Gaussian sigma in period units
            * ``component_weights``        - kernel component weights

        Raises
        ------
        RuntimeError
            If the model has not been initialised.
        ValueError
            If neither the ``sci_kernel`` nor its ``base_kernel`` expose
            ``mixture_means`` (i.e. the model is not spectral-mixture).
        """
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError(
                "Model not initialised.  Call set_model() first."
            )
        # Some SKI variants set sci_kernel to the GridInterpolationKernel
        # wrapper rather than the SpectralMixtureKernel itself.  Unwrap it.
        sk = self.model.sci_kernel
        if not hasattr(sk, "mixture_means") and hasattr(
            sk, "base_kernel"
        ):
            sk = sk.base_kernel

        if not hasattr(sk, "mixture_means"):
            raise ValueError(
                "_extract_sm_params() requires a spectral-mixture kernel.  "
                "The current sci_kernel does not expose mixture_means."
            )

        n_mix = len(sk.mixture_means)

        freqs = []
        periods = []
        freq_scales = []
        period_scales = []
        wts = []

        for i in range(n_mix):
            # -- extract the time-dimension mean and scale ------------------
            # mixture_means shape is [n_mix, 1, ard_num_dims].
            # Index [i, 0, 0] selects mixture i, the redundant size-1
            # dimension, and dimension 0 (time axis).  This is consistent
            # with the [i, 0] indexing used in get_periods() for 2-D models.
            mu_t = sk.mixture_means[i, 0, 0]
            sig_t = sk.mixture_scales[i, 0, 0]

            if self.xtransform is None:
                # No coordinate transform: mixture_mean IS the raw frequency,
                # and mixture_scale IS the raw frequency-domain half-width.
                raw_freq = float(mu_t.detach().cpu())
                raw_period = 1.0 / raw_freq
                # Convert frequency-domain scale to period-domain scale.
                raw_freq_scale = float(sig_t.detach().cpu())
                raw_period_scale = (
                    1.0 / (2.0 * np.pi * raw_freq_scale)
                )
            else:
                # The model was trained in normalised time units.
                # Inverse-transform (shift=False = scale only) to recover
                # physical (raw) period, then compute frequency from it.
                raw_period = float(
                    self.xtransform.inverse(
                        1.0 / mu_t, shift=False
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .ravel()[0]
                )
                raw_freq = 1.0 / raw_period
                # Same inverse transform for the scale parameter,
                # converting normalised frequency scale to raw period scale.
                raw_period_scale = float(
                    self.xtransform.inverse(
                        1.0 / (2.0 * torch.pi * sig_t),
                        shift=False,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .ravel()[0]
                )
                raw_freq_scale = (
                    1.0 / (2.0 * np.pi * raw_period_scale)
                )

            freqs.append(raw_freq)
            periods.append(raw_period)
            freq_scales.append(raw_freq_scale)
            period_scales.append(raw_period_scale)
            wts.append(float(sk.mixture_weights[i].detach().cpu()))

        return {
            "component_frequencies": np.array(freqs),
            "component_periods": np.array(periods),
            "component_frequency_scales": np.array(freq_scales),
            "component_period_scales": np.array(period_scales),
            "component_weights": np.array(wts),
        }

    @staticmethod
    def _sm_psd_on_grid(freq_grid, params):
        """Evaluate the total spectral-mixture PSD on a frequency grid.

        The PSD is the (non-normalised) sum of weighted Gaussians in
        frequency space::

            PSD(f) = sum_k  w_k * exp(-0.5 * ((f - mu_k) / sigma_k)^2)

        where ``mu_k``, ``sigma_k`` and ``w_k`` are the raw (physical-unit)
        component frequencies, frequency scales, and weights returned by
        :meth:`_extract_sm_params`.  Overall normalisation is not enforced
        because only the peak *location* matters for period identification.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            1-D positive-frequency evaluation grid in physical units.
        params : dict
            Output of :meth:`_extract_sm_params`.

        Returns
        -------
        psd : numpy.ndarray
            PSD values on ``freq_grid``, same shape as ``freq_grid``.
        """
        psd = np.zeros_like(freq_grid, dtype=float)
        mus = params["component_frequencies"]
        sigs = params["component_frequency_scales"]
        wts = params["component_weights"]
        if not (len(mus) == len(sigs) == len(wts)):
            raise ValueError(
                f"Spectral-mixture parameter arrays have inconsistent "
                f"lengths: component_frequencies={len(mus)}, "
                f"component_frequency_scales={len(sigs)}, "
                f"component_weights={len(wts)}.  This indicates an "
                f"internal error in _extract_sm_params()."
            )
        for mu_k, sig_k, w_k in zip(mus, sigs, wts, strict=True):
            psd += w_k * np.exp(
                -0.5 * ((freq_grid - mu_k) / sig_k) ** 2
            )
        return psd

    def _detect_period_summary_backend(self):
        """Classify the fitted model into a period-summary backend family.

        Inspects the actual kernel objects attached to the model and returns
        a string label that :meth:`get_period_summary` uses to dispatch to
        the appropriate extraction routine.

        Returns
        -------
        backend : str
            One of:

            * ``"spectral_mixture"`` - SpectralMixture kernel (or SKI
              wrapper around one) - use PSD-peak extraction.
            * ``"explicit_period"`` - kernel tree contains a
              :class:`~gpytorch.kernels.PeriodicKernel` with a fitted
              ``period_length`` parameter (e.g. quasi-periodic models).
            * ``"periodic_plus_stochastic"`` - AdditiveKernel combining a
              quasi-periodic term with a stochastic (RBF) term.
            * ``"separable_2d"`` - ProductKernel with per-dimension
              ``active_dims`` (separable 2D models); the time sub-kernel
              is inspected independently.
            * ``"non_periodic"`` - no periodic structure found (e.g.
              Matérn-only model).
        """
        from gpytorch.kernels import AdditiveKernel, ProductKernel

        sk = self.model.sci_kernel
        # Unwrap GridInterpolationKernel if present
        actual_sk = getattr(sk, "base_kernel", sk)

        # 1. Spectral-mixture family (includes SKI wrappers)
        if hasattr(actual_sk, "mixture_means"):
            return "spectral_mixture"

        # 2. Additive kernel - periodic + stochastic decomposition
        if isinstance(sk, AdditiveKernel):
            return "periodic_plus_stochastic"

        # 3. Product kernel with active_dims on sub-kernels - separable 2D
        if isinstance(sk, ProductKernel):
            has_active_dims = any(
                hasattr(k, "active_dims") and k.active_dims is not None
                for k in sk.kernels
            )
            if has_active_dims:
                return "separable_2d"

        # 4. Any kernel that contains a PeriodicKernel with period_length
        if self._find_period_length_in_kernel(sk) is not None:
            return "explicit_period"

        # 5. Non-periodic fallback
        return "non_periodic"

    @staticmethod
    def _find_period_length_in_kernel(kernel):
        """Recursively search a kernel tree for a PeriodicKernel.

        Walks the kernel tree depth-first via ``base_kernel`` and
        ``kernels`` attributes and returns the first kernel instance that
        has a ``period_length`` attribute (i.e. a
        :class:`~gpytorch.kernels.PeriodicKernel`).

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel
            Root kernel to search.

        Returns
        -------
        periodic_kernel : gpytorch.kernels.Kernel or None
            The first kernel with ``period_length``, or ``None`` if none
            is found.
        """
        if hasattr(kernel, "period_length"):
            return kernel
        # Unwrap ScaleKernel / GridInterpolationKernel wrappers
        if hasattr(kernel, "base_kernel"):
            result = Lightcurve._find_period_length_in_kernel(
                kernel.base_kernel
            )
            if result is not None:
                return result
        # Recurse into ProductKernel / AdditiveKernel sub-kernels
        if hasattr(kernel, "kernels"):
            for k in kernel.kernels:
                result = Lightcurve._find_period_length_in_kernel(k)
                if result is not None:
                    return result
        return None

    def _extract_explicit_period_params(self, kernel):
        """Extract the dominant period from a kernel containing a PeriodicKernel.

        Finds the first :class:`~gpytorch.kernels.PeriodicKernel` in the
        kernel tree (via :meth:`_find_period_length_in_kernel`), reads its
        ``period_length``, and inverse-transforms it back to raw data units
        using ``self.xtransform`` (with ``shift=False`` - scaling only,
        since a period is a duration, not an absolute coordinate).

        If an RBF sub-kernel is found alongside the PeriodicKernel (as in
        the quasi-periodic product), its lengthscale is used to derive a
        practical coherence-based period interval and Q-factor.

        The scientifically important transforms are:

        * If ``self.xtransform is None``: period is stored in raw units
          already.
        * If ``self.xtransform is not None``: ``period_length`` is in the
          normalised time coordinate; ``xtransform.inverse(..., shift=False)``
          recovers the raw-unit period (``shift=False`` because a period is a
          *duration*, not an absolute time, so only the scale factor matters).

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel
            Kernel tree to search (typically ``self.model.sci_kernel`` or a
            sub-kernel thereof).

        Returns
        -------
        params : dict or None
            Dictionary with:

            * ``raw_period`` - dominant period in raw data units
            * ``raw_freq`` - ``1 / raw_period``
            * ``raw_rbf_lengthscale`` - coherence timescale in raw units, or
              ``None`` if no RBF kernel was found alongside the periodic one
            * ``period_lo``, ``period_hi`` - coherence-based interval (or
              equal to ``raw_period`` when no RBF lengthscale is available)
            * ``q_factor`` - coherence Q (RBF-based), or ``None``

            Returns ``None`` if no PeriodicKernel is found.
        """
        pk = self._find_period_length_in_kernel(kernel)
        if pk is None:
            return None

        period_norm = float(
            pk.period_length.detach().cpu().numpy().ravel()[0]
        )

        if self.xtransform is None:
            # period_length is in raw data units already
            raw_period = period_norm
        else:
            # Inverse-transform: shift=False because a period is a duration
            # (only the scale factor matters, not the origin shift).
            raw_period = float(
                self.xtransform.inverse(
                    torch.as_tensor([period_norm]), shift=False
                )
                .detach()
                .cpu()
                .numpy()
                .ravel()[0]
            )

        raw_period = abs(raw_period)
        raw_freq = 1.0 / raw_period if raw_period > 0 else np.nan

        # -- RBF lengthscale for coherence estimate (optional) --------------
        # In a quasi-periodic kernel the RBF lengthscale sets the coherence
        # time.  We search the same kernel tree for an RBF lengthscale that
        # lives alongside the PeriodicKernel.
        raw_rbf_ls = None
        if hasattr(kernel, "kernels"):
            # ProductKernel or AdditiveKernel at top level
            _kernels_to_search = list(kernel.kernels)
        elif hasattr(kernel, "base_kernel") and hasattr(
            kernel.base_kernel, "kernels"
        ):
            # ScaleKernel wrapping a ProductKernel
            _kernels_to_search = list(kernel.base_kernel.kernels)
        else:
            _kernels_to_search = []

        for k in _kernels_to_search:
            # Unwrap ScaleKernel wrappers
            inner = getattr(k, "base_kernel", k)
            if hasattr(inner, "lengthscale") and not hasattr(
                inner, "period_length"
            ):
                ls_norm = float(
                    inner.lengthscale.detach().cpu().numpy().ravel()[0]
                )
                if self.xtransform is None:
                    raw_rbf_ls = ls_norm
                else:
                    raw_rbf_ls = float(
                        self.xtransform.inverse(
                            torch.as_tensor([ls_norm]), shift=False
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .ravel()[0]
                    )
                break

        # -- period interval and Q from RBF coherence time -----------------
        if raw_rbf_ls is not None and raw_rbf_ls > 0:
            # Bandwidth from Gaussian (RBF) envelope: delta_f ~ 1/(2pi*L)
            # Linearised period uncertainty: delta_p ~ P^2 * delta_f
            delta_p = raw_period**2 / (2.0 * np.pi * raw_rbf_ls)
            period_lo = max(raw_period - delta_p / 2.0, 1e-12)
            period_hi = raw_period + delta_p / 2.0
            # Q = f_peak / FWHM_f ~ (2*pi * L) / P
            q_factor = 2.0 * np.pi * raw_rbf_ls / raw_period
        else:
            period_lo = raw_period
            period_hi = raw_period
            q_factor = None

        return {
            "raw_period": raw_period,
            "raw_freq": raw_freq,
            "raw_rbf_lengthscale": raw_rbf_ls,
            "period_lo": period_lo,
            "period_hi": period_hi,
            "q_factor": q_factor,
        }

    @staticmethod
    def _kernel_family_name(kernel):
        """Return the class name of *kernel*, or ``""`` if not available.

        Used to populate ``kernel_family`` and ``time_kernel_family`` on
        :class:`PeriodSummaryResult` objects.  If *kernel* is ``None`` an
        empty string is returned so that callers can fall back gracefully.

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel or None
            The kernel whose class name should be returned.

        Returns
        -------
        str
            ``type(kernel).__name__`` or ``""``.
        """
        if kernel is None:
            return ""
        return type(kernel).__name__

    @staticmethod
    def _resolve_time_kernel_family(kernel):
        """Return the class name of the time-dimension sub-kernel.

        For separable 2D models the ``sci_kernel`` is a
        :class:`~gpytorch.kernels.ProductKernel` whose sub-kernels carry
        ``active_dims`` attributes.  This helper locates the sub-kernel
        acting on dimension 0 (time), unwraps any
        :class:`~gpytorch.kernels.ScaleKernel` wrapper, and returns its
        class name.

        For 1-D models the supplied *kernel* IS the time kernel; its
        ``base_kernel`` is unwrapped when present (e.g.
        ``ScaleKernel(SpectralMixtureKernel)`` → ``SpectralMixtureKernel``).

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel or None
            The top-level ``sci_kernel`` from the fitted model.

        Returns
        -------
        str
            Class name of the resolved time kernel, or ``""`` if *kernel*
            is ``None`` or no time sub-kernel can be identified.
        """
        if kernel is None:
            return ""
        from gpytorch.kernels import ProductKernel as _ProductKernel

        # Separable 2D: ProductKernel whose sub-kernels carry active_dims.
        if isinstance(kernel, _ProductKernel):
            for k in kernel.kernels:
                ad = getattr(k, "active_dims", None)
                if ad is not None and 0 in ad.tolist():
                    # Unwrap ScaleKernel / GridInterpolationKernel wrappers
                    actual_tk = getattr(k, "base_kernel", k)
                    return type(actual_tk).__name__
            # ProductKernel without active_dims – fall through to 1-D path.

        # 1-D (or non-separable): unwrap wrapper kernels one level.
        actual_k = getattr(kernel, "base_kernel", kernel)
        return type(actual_k).__name__

    @staticmethod
    def _coerce_float_or_none(value):
        """Safely coerce scalar input to finite float.

        Parameters
        ----------
        value : any
            Candidate scalar value.

        Returns
        -------
        float or None
            Finite float value, otherwise ``None``.
        """
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    def _get_consensus_multicomp_period_summary(self):
        """Return a multicomp period summary from consensus diagnostics."""
        diagnostics = getattr(self, "consensus_diagnostics", None)
        if not isinstance(diagnostics, dict):
            return None
        if diagnostics.get("fit_strategy") != "consensus_multicomp":
            return None
        if not bool(diagnostics.get("consensus_success")):
            return None

        raw_component_summaries = diagnostics.get("multicomponent_period_summaries")
        if not isinstance(raw_component_summaries, list):
            return None
        component_summaries = [
            dict(entry)
            for entry in raw_component_summaries
            if isinstance(entry, dict)
        ]
        if not component_summaries:
            return None

        consensus_periods = list(diagnostics.get("consensus_periods") or [])
        consensus_period_widths = list(
            diagnostics.get("consensus_period_widths") or []
        )
        fitted_periods = list(diagnostics.get("fitted_mixture_periods") or [])
        initialized_periods = list(
            diagnostics.get("initialized_mixture_periods") or []
        )
        consensus_strengths = list(
            diagnostics.get("consensus_component_strengths") or []
        )

        component_periods = []
        component_period_widths = []
        component_fitted_periods = []
        component_initialized_periods = []
        component_strengths = []
        component_source_cluster_ids = []
        component_member_bands = []

        for idx, component in enumerate(component_summaries):
            component_periods.append(
                self._coerce_float_or_none(
                    component.get(
                        "consensus_period",
                        (
                            consensus_periods[idx]
                            if idx < len(consensus_periods)
                            else None
                        ),
                    )
                )
            )
            component_period_widths.append(
                self._coerce_float_or_none(
                    component.get(
                        "consensus_period_width",
                        (
                            consensus_period_widths[idx]
                            if idx < len(consensus_period_widths)
                            else None
                        ),
                    )
                )
            )
            component_fitted_periods.append(
                self._coerce_float_or_none(
                    component.get(
                        "fitted_mixture_period",
                        fitted_periods[idx] if idx < len(fitted_periods) else None,
                    )
                )
            )
            component_initialized_periods.append(
                self._coerce_float_or_none(
                    component.get(
                        "initialized_mixture_period",
                        (
                            initialized_periods[idx]
                            if idx < len(initialized_periods)
                            else None
                        ),
                    )
                )
            )
            component_strengths.append(
                self._coerce_float_or_none(
                    component.get(
                        "consensus_component_strength",
                        (
                            consensus_strengths[idx]
                            if idx < len(consensus_strengths)
                            else None
                        ),
                    )
                )
            )
            component_source_cluster_ids.append(component.get("source_cluster_id"))
            component_member_bands.append(list(component.get("member_bands") or []))

        _model_obj = getattr(self, "model", None)
        kernel = getattr(_model_obj, "sci_kernel", None)
        kernel_family = self._kernel_family_name(getattr(kernel, "base_kernel", kernel))
        time_kernel_family = self._resolve_time_kernel_family(kernel)
        _model_name = type(_model_obj).__name__ if _model_obj is not None else ""

        return PeriodSummaryResult(
            method="consensus_multicomp_period_summary",
            backend="consensus_multicomp",
            model_name=_model_name,
            kernel_family=kernel_family,
            time_kernel_family=time_kernel_family,
            has_stochastic_background=False,
            dominant_period=None,
            dominant_frequency=None,
            peaks=[],
            freq_grid=None,
            psd=None,
            notes=(
                "Multi-component consensus fit summary. "
                "No single dominant period is defined."
            ),
            interval_definition="none",
            is_multicomponent=True,
            component_periods=component_periods,
            component_period_widths=component_period_widths,
            component_fitted_periods=component_fitted_periods,
            component_initialized_periods=component_initialized_periods,
            component_strengths=component_strengths,
            component_source_cluster_ids=component_source_cluster_ids,
            component_member_bands=component_member_bands,
            component_summaries=component_summaries,
            drift_warning_fraction=self._coerce_float_or_none(
                diagnostics.get("drift_warning_fraction")
            ),
        )

    def _get_non_periodic_summary(self, kernel=None):
        """Return a graceful period summary for a non-periodic kernel.

        Used when the model has no periodic structure (e.g.
        :class:`~pgmuvi.gps.MaternGPModel`).  All period-related fields
        are set to ``None`` or empty arrays, and the ``backend`` field is
        set to ``"non_periodic"`` so that downstream code can
        distinguish this from a genuine period summary.

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel or None, optional
            Kernel to report as the family name.  Defaults to
            ``self.model.sci_kernel`` when available.

        Returns
        -------
        summary : PeriodSummaryResult
            Consistent structured result with no period information.
        """
        _k = (
            kernel if kernel is not None
            else getattr(getattr(self, "model", None), "sci_kernel", None)
        )
        kf = self._kernel_family_name(_k)
        return PeriodSummaryResult(
            method="non_periodic_kernel",
            backend="non_periodic",
            kernel_family=kf,
            time_kernel_family=kf,
            has_stochastic_background=False,
            dominant_period=None,
            dominant_frequency=None,
            peaks=[],
            freq_grid=None,
            psd=None,
            interval_definition="none",
            notes=(
                "This kernel family does not encode a periodic timescale, "
                "so no dominant period is defined. "
                f"Kernel: {kf}."
            ),
        )

    def _get_explicit_period_summary(self, kernel=None):
        """Return a period summary for an explicit-period kernel (e.g. QP).

        Extracts the dominant period directly from a
        :class:`~gpytorch.kernels.PeriodicKernel` embedded in the model's
        ``sci_kernel``.  This is appropriate for quasi-periodic models
        (:class:`~pgmuvi.gps.QuasiPeriodicGPModel`,
        :class:`~pgmuvi.gps.LinearMeanQuasiPeriodicGPModel`) where the
        period is a directly fitted parameter.

        The uncertainty is a coherence-based proxy derived from the RBF
        lengthscale that accompanies the PeriodicKernel in the product
        ``k_periodic * k_rbf``.  It is **not** a posterior credible
        interval; MCMC-based intervals are not yet implemented.

        Parameters
        ----------
        kernel : gpytorch.kernels.Kernel or None, optional
            Kernel to search.  Defaults to ``self.model.sci_kernel``.

        Returns
        -------
        summary : PeriodSummaryResult
            Structured result.  ``freq_grid`` and ``psd`` are ``None``
            (no PSD is computed for this backend).  The dominant period
            comes from the kernel's fitted period parameter, not a PSD
            peak search.  The uncertainty interval, when present, is a
            coherence proxy from the RBF lengthscale, labelled
            ``"coherence_proxy_from_rbf_lengthscale"``.
        """
        if kernel is None:
            kernel = self.model.sci_kernel

        kf = self._kernel_family_name(kernel)
        ep = self._extract_explicit_period_params(kernel)
        if ep is None:
            return self._get_non_periodic_summary(kernel=kernel)

        raw_period = ep["raw_period"]
        raw_freq = ep["raw_freq"]
        period_lo = ep["period_lo"]
        period_hi = ep["period_hi"]
        raw_rbf_ls = ep["raw_rbf_lengthscale"]
        q_factor = ep["q_factor"]

        if raw_rbf_ls is not None:
            interval_def = "coherence_proxy_from_rbf_lengthscale"
            notes = (
                "Dominant period extracted from the fitted period_length "
                "parameter of the PeriodicKernel (explicit_period backend). "
                "The uncertainty interval is a coherence proxy derived from "
                "the RBF lengthscale; it is NOT a PSD-derived peak interval "
                "and NOT a posterior credible interval."
            )
            # Frequency interval from the period interval
            f_lo = 1.0 / period_hi if period_hi > 0 else float("nan")
            f_hi = 1.0 / period_lo if period_lo > 0 else float("nan")
        else:
            interval_def = "none"
            notes = (
                "Dominant period extracted from the fitted period_length "
                "parameter of the PeriodicKernel (explicit_period backend). "
                "No coherence timescale found; no defensible interval is "
                "reported."
            )
            period_lo = float("nan")
            period_hi = float("nan")
            f_lo = float("nan")
            f_hi = float("nan")

        # Represent the single explicit period as a PeriodPeakResult so
        # that all dict-access keys (period_interval_fwhm_like, n_peaks, …)
        # remain backward-compatible.  area_fraction=1.0 marks this as the
        # sole significant period.
        _peak = PeriodPeakResult(
            rank=1,
            frequency=raw_freq,
            period=raw_period,
            height=float("nan"),
            prominence=float("nan"),
            area_fraction=1.0,
            interval_frequency=(f_lo, f_hi),
            interval_period=(period_lo, period_hi),
            period_ratio_to_primary=1.0,
            is_candidate_lsp=False,
            notes=(
                "Coherence-proxy interval from RBF lengthscale"
                if raw_rbf_ls is not None
                else "No interval available"
            ),
        )

        return PeriodSummaryResult(
            method="explicit_period_parameter",
            backend="explicit_period",
            kernel_family=kf,
            time_kernel_family=kf,
            has_stochastic_background=False,
            dominant_period=raw_period,
            dominant_frequency=raw_freq,
            peaks=[_peak],
            freq_grid=None,
            psd=None,
            notes=notes,
            interval_definition=interval_def,
            q_factor=q_factor,
        )

    def _get_periodic_plus_stochastic_summary(self):
        """Return a period summary for a periodic-plus-stochastic kernel.

        Used for :class:`~pgmuvi.gps.PeriodicPlusStochasticGPModel`, whose
        ``sci_kernel`` is an :class:`~gpytorch.kernels.AdditiveKernel`
        combining a quasi-periodic part (``k_periodic * k_rbf``) with a
        purely stochastic RBF part.

        The dominant period is extracted from the quasi-periodic sub-kernel
        using the same logic as :meth:`_get_explicit_period_summary`.
        The stochastic component is treated as non-periodic background
        support and is **not** interpreted as an independent period.

        Returns
        -------
        summary : PeriodSummaryResult
            Structured result.  ``backend == "periodic_plus_stochastic"``,
            ``has_stochastic_background is True``.
        """
        overall_kf = self._kernel_family_name(
            getattr(self.model, "sci_kernel", None)
        )
        # The first sub-kernel of the AdditiveKernel is the QP part.
        qp_kernel = self.model.sci_kernel.kernels[0]
        qp_kf = self._kernel_family_name(qp_kernel)
        ep_summary = self._get_explicit_period_summary(kernel=qp_kernel)

        # Build updated notes that are explicit about periodic vs stochastic
        _pps_note = (
            "Periodic-plus-stochastic model (periodic_plus_stochastic "
            "backend).  The reported period comes from the periodic "
            "sub-kernel only.  The stochastic (RBF) component is treated "
            "as non-periodic background support and is NOT interpreted as "
            "an independent period.  "
        )
        return PeriodSummaryResult(
            method="periodic_plus_stochastic",
            backend="periodic_plus_stochastic",
            kernel_family=overall_kf,
            time_kernel_family=qp_kf,
            has_stochastic_background=True,
            dominant_period=ep_summary.dominant_period,
            dominant_frequency=ep_summary.dominant_frequency,
            peaks=list(ep_summary.peaks),
            freq_grid=ep_summary.freq_grid,
            psd=ep_summary.psd,
            notes=_pps_note + ep_summary.notes,
            interval_definition=ep_summary.interval_definition,
        )

    def _get_separable_2d_period_summary(self, **kwargs):
        """Return a period summary for a separable-product 2D kernel.

        For separable 2D models (e.g. :class:`~pgmuvi.gps.SeparableGPModel`,
        :class:`~pgmuvi.gps.AchromaticGPModel`,
        :class:`~pgmuvi.gps.WavelengthDependentGPModel`,
        :class:`~pgmuvi.gps.DustMeanGPModel`,
        :class:`~pgmuvi.gps.PowerLawMeanGPModel`) the ``sci_kernel`` is a
        :class:`~gpytorch.kernels.ProductKernel` whose sub-kernels each
        carry an ``active_dims`` attribute that identifies which input
        dimension (time = 0, wavelength = 1) they act on.

        This method:

        1. Identifies the time sub-kernel (``active_dims`` contains 0).
        2. Classifies the time kernel into a period-summary backend.
        3. Delegates to the appropriate backend method.
        4. Wraps the result in a ``separable_2d``-annotated
           :class:`PeriodSummaryResult` that explicitly records the
           time-kernel family used for the summary.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`_get_sm_period_summary` when the time
            kernel is spectral-mixture.

        Returns
        -------
        summary : PeriodSummaryResult
            Period summary based on the time kernel only.
            ``backend == "separable_2d"``, ``time_kernel_family`` names
            the time kernel.  The wavelength kernel does not contribute
            to the reported period.
        """

        sk = self.model.sci_kernel
        overall_kf = self._kernel_family_name(sk)
        # Identify the time sub-kernel (active_dims contains 0)
        time_kernel = None
        for k in sk.kernels:
            ad = getattr(k, "active_dims", None)
            if ad is not None and 0 in ad.tolist():
                time_kernel = k
                break

        if time_kernel is None:
            # Cannot identify time kernel; fall back to non-periodic
            np_summary = self._get_non_periodic_summary()
            return PeriodSummaryResult(
                method=np_summary.method,
                backend="separable_2d",
                kernel_family=overall_kf,
                time_kernel_family="",
                has_stochastic_background=False,
                dominant_period=np_summary.dominant_period,
                dominant_frequency=np_summary.dominant_frequency,
                peaks=list(np_summary.peaks),
                freq_grid=np_summary.freq_grid,
                psd=np_summary.psd,
                notes=(
                    "Separable 2D model: could not identify the time "
                    "sub-kernel; no period summary is available."
                ),
                interval_definition=np_summary.interval_definition,
            )

        # Classify the time kernel
        actual_tk = getattr(time_kernel, "base_kernel", time_kernel)
        tkf = self._kernel_family_name(actual_tk)
        _2d_prefix = (
            "Separable 2D model (separable_2d backend): period summary "
            f"derived from the time kernel ({tkf}) only.  "
            "The wavelength kernel does not contribute to this period "
            "determination.  "
        )

        if hasattr(actual_tk, "mixture_means"):
            # Spectral-mixture time kernel - use PSD method
            # Temporarily swap sci_kernel to expose the time kernel
            # as a stand-alone SM kernel for _extract_sm_params.
            _orig_sk = self.model.sci_kernel
            self.model.sci_kernel = actual_tk
            try:
                inner = self._get_sm_period_summary(**kwargs)
            finally:
                self.model.sci_kernel = _orig_sk
            return PeriodSummaryResult(
                method=inner.method,
                backend="separable_2d",
                kernel_family=overall_kf,
                time_kernel_family=tkf,
                has_stochastic_background=inner.has_stochastic_background,
                dominant_period=inner.dominant_period,
                dominant_frequency=inner.dominant_frequency,
                peaks=list(inner.peaks),
                freq_grid=inner.freq_grid,
                psd=inner.psd,
                notes=_2d_prefix + inner.notes,
                component_diagnostics=inner.component_diagnostics,
                interval_definition=inner.interval_definition,
                n_peaks_detected=inner.n_peaks_detected,
                n_peaks_analyzed=inner.n_peaks_analyzed,
                n_peaks_requested=inner.n_peaks_requested,
            )

        if self._find_period_length_in_kernel(time_kernel) is not None:
            # Explicit-period time kernel (e.g. quasi-periodic)
            inner = self._get_explicit_period_summary(kernel=time_kernel)
            return PeriodSummaryResult(
                method=inner.method,
                backend="separable_2d",
                kernel_family=overall_kf,
                time_kernel_family=tkf,
                has_stochastic_background=inner.has_stochastic_background,
                dominant_period=inner.dominant_period,
                dominant_frequency=inner.dominant_frequency,
                peaks=list(inner.peaks),
                freq_grid=inner.freq_grid,
                psd=inner.psd,
                notes=_2d_prefix + inner.notes,
                interval_definition=inner.interval_definition,
            )

        # Non-periodic time kernel
        return PeriodSummaryResult(
            method="non_periodic_kernel",
            backend="separable_2d",
            kernel_family=overall_kf,
            time_kernel_family=tkf,
            has_stochastic_background=False,
            dominant_period=None,
            dominant_frequency=None,
            peaks=[],
            freq_grid=None,
            psd=None,
            notes=(
                "Separable 2D model: the time kernel "
                f"({tkf}) is non-periodic, "
                "so no dominant period is defined."
            ),
            interval_definition="none",
        )

    @staticmethod
    def _find_dominant_peak_basin(psd, dominant_idx):
        """Identify the basin of the dominant PSD peak.

        Starting from the dominant peak index, walk left and right until a
        local minimum is found or the edge of the array is reached.  The
        basin is the contiguous region associated with the dominant mode.

        Parameters
        ----------
        psd : numpy.ndarray
            1-D PSD values on a frequency grid.
        dominant_idx : int
            Index of the dominant PSD peak in ``psd``.

        Returns
        -------
        basin_left : int
            Index of the left edge of the basin (inclusive).
        basin_right : int
            Index of the right edge of the basin (inclusive).
        left_at_boundary : bool
            ``True`` if the left edge reached the array boundary without
            finding a local minimum.
        right_at_boundary : bool
            ``True`` if the right edge reached the array boundary without
            finding a local minimum.
        """
        n = len(psd)

        # Walk left: stop at local minimum (psd starts rising)
        left = dominant_idx
        while left > 0 and psd[left - 1] < psd[left]:
            left -= 1
        left_at_boundary = left == 0

        # Walk right: stop at local minimum (psd starts rising)
        right = dominant_idx
        while right < n - 1 and psd[right + 1] < psd[right]:
            right += 1
        right_at_boundary = right == n - 1

        return left, right, left_at_boundary, right_at_boundary

    @staticmethod
    def _integrate_logspace(psd, freq_grid):
        """Integrate a PSD over a log-spaced frequency grid.

        Computes the integral of ``psd * freq`` with respect to
        ``log(freq)``, which equals the linear integral of ``psd`` over
        ``freq`` when the grid is logarithmically spaced.  Using this
        formulation on a log-spaced grid avoids the strong bias towards
        high frequencies that arises from naively applying the trapezoidal
        rule in linear frequency space.

        Formally this implements::

            integral of psd(f) df  ~  integral of (f * psd(f)) d(log f)
                                   ~  trapz(psd * freq, log(freq))

        Parameters
        ----------
        psd : numpy.ndarray
            1-D PSD values on ``freq_grid``.
        freq_grid : numpy.ndarray
            Positive, log-spaced frequency values with the same length as
            ``psd``.

        Returns
        -------
        integral : float
            Estimated integral value.  Always ≥ 0.
        """
        if len(freq_grid) < 2:
            return 0.0
        log_f = np.log(freq_grid)
        weights = psd * freq_grid
        try:
            return float(np.trapezoid(weights, log_f))
        except AttributeError:
            return float(np.trapz(weights, log_f))

    @staticmethod
    def _compute_equal_tail_mass_interval(
        freq_grid, psd, basin_left, basin_right, mass_level=0.68
    ):
        """Compute an equal-tail mass interval within the dominant peak basin.

        .. deprecated::
            This method is retained as a legacy helper.  The default
            ``uncertainty="peak_mass"`` path now uses
            :meth:`_compute_peak_centered_mass_interval`, which guarantees
            that the returned interval contains the peak frequency.

        Integrates the PSD (as a proxy for a probability density) over the
        basin region and returns the frequency interval that contains a
        centred fraction ``mass_level`` of the total basin mass, using an
        equal-tail (symmetric quantile) approach.

        Integration is performed in log-frequency space (``trapz(psd *
        freq, log_freq)``) to avoid the bias towards high frequencies that
        arises from naive linear-space integration on a log-spaced grid.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Full frequency evaluation grid (positive, log-spaced).
        psd : numpy.ndarray
            PSD values on ``freq_grid``.
        basin_left : int
            Left edge index of the dominant-peak basin (inclusive).
        basin_right : int
            Right edge index of the dominant-peak basin (inclusive).
        mass_level : float, optional
            Fraction of the basin mass to enclose (default 0.68 for
            approximately ``1 sigma`` coverage).  Must be in ``(0, 1)``.

        Returns
        -------
        f_lo : float
            Lower frequency bound of the equal-tail interval.
        f_hi : float
            Upper frequency bound of the equal-tail interval.
        success : bool
            ``True`` if the interval could be computed; ``False`` if the
            basin was too narrow (< 2 grid points) or the total mass was
            numerically zero.
        """
        f_basin = freq_grid[basin_left : basin_right + 1]
        p_basin = psd[basin_left : basin_right + 1]

        if len(f_basin) < 2:
            return float(f_basin[0]), float(f_basin[0]), False

        # Log-space integration via shared helper
        total_mass = Lightcurve._integrate_logspace(p_basin, f_basin)
        if total_mass <= 0:
            return float(f_basin[0]), float(f_basin[-1]), False

        # Build cumulative mass in log-space
        log_f = np.log(f_basin)
        weights = p_basin * f_basin
        cum = np.zeros(len(f_basin))
        for i in range(1, len(f_basin)):
            dlogf = log_f[i] - log_f[i - 1]
            cum[i] = cum[i - 1] + 0.5 * (
                weights[i - 1] + weights[i]
            ) * dlogf
        cum /= total_mass  # normalise to [0, 1]

        tail = (1.0 - mass_level) / 2.0

        # Interpolate lower quantile
        f_lo = float(np.interp(tail, cum, f_basin))
        # Interpolate upper quantile
        f_hi = float(np.interp(1.0 - tail, cum, f_basin))

        return f_lo, f_hi, True

    @staticmethod
    def _compute_peak_centered_mass_interval(
        freq_grid, psd, basin_left, basin_right, peak_idx, mass_level=0.68
    ):
        """Compute a peak-centered mass interval within a PSD basin.

        This is the preferred method for ``uncertainty="peak_mass"``.

        Unlike the equal-tail approach, this method **guarantees that the
        returned interval contains the peak frequency** by growing the
        interval outward from the peak, always expanding toward the side
        that contributes more mass per log-frequency unit.  This greedy
        "grow from the peak" strategy is equivalent to finding the shortest
        interval (in log-frequency space) that encloses the requested mass
        fraction and still contains the peak.

        Integration is performed in log-frequency space to avoid the
        high-frequency bias of naive linear-space trapezoidal integration
        on a log-spaced grid.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Full frequency evaluation grid (positive, log-spaced).
        psd : numpy.ndarray
            PSD values on ``freq_grid``.
        basin_left : int
            Left edge index of the basin (inclusive).
        basin_right : int
            Right edge index of the basin (inclusive).
        peak_idx : int
            Index of the peak in the full ``freq_grid``.  Must satisfy
            ``basin_left <= peak_idx <= basin_right``.
        mass_level : float, optional
            Target fraction of basin mass to enclose.  Default 0.68.

        Returns
        -------
        f_lo : float
            Lower frequency bound of the peak-centered interval.
        f_hi : float
            Upper frequency bound of the peak-centered interval.
        success : bool
            ``True`` if the interval was computed successfully.  ``False``
            if the basin has fewer than 2 grid points or the total mass is
            numerically zero.
        """
        f_basin = freq_grid[basin_left : basin_right + 1]
        p_basin = psd[basin_left : basin_right + 1]
        pk_rel = int(peak_idx) - int(basin_left)

        if len(f_basin) < 2:
            return float(f_basin[0]), float(f_basin[0]), False

        # Log-space integration weights
        log_f = np.log(f_basin)
        weights = p_basin * f_basin

        # Total basin mass via shared helper (avoids duplicating try/except)
        total_mass = Lightcurve._integrate_logspace(p_basin, f_basin)
        if total_mass <= 0:
            return float(f_basin[0]), float(f_basin[-1]), False

        # Per-segment mass in log-space
        # seg_mass[i] = mass of segment [i, i+1]
        n = len(f_basin)
        seg_mass = np.zeros(n - 1)
        for i in range(n - 1):
            dlogf = log_f[i + 1] - log_f[i]
            seg_mass[i] = 0.5 * (weights[i] + weights[i + 1]) * dlogf

        # Greedy grow from the peak: always expand into the denser side
        left_ptr = pk_rel
        right_ptr = pk_rel
        accumulated = 0.0

        while accumulated / total_mass < mass_level:
            can_go_left = left_ptr > 0
            can_go_right = right_ptr < n - 1

            if not can_go_left and not can_go_right:
                break

            if can_go_left and can_go_right:
                left_seg = seg_mass[left_ptr - 1]
                right_seg = seg_mass[right_ptr]
                if left_seg >= right_seg:
                    accumulated += left_seg
                    left_ptr -= 1
                else:
                    accumulated += right_seg
                    right_ptr += 1
            elif can_go_left:
                accumulated += seg_mass[left_ptr - 1]
                left_ptr -= 1
            else:
                accumulated += seg_mass[right_ptr]
                right_ptr += 1

        f_lo = float(f_basin[left_ptr])
        f_hi = float(f_basin[right_ptr])
        return f_lo, f_hi, True

    @staticmethod
    def _build_frequency_grid(min_freq, max_freq, n_grid, spacing="log"):
        """Build a 1-D positive-frequency evaluation grid.

        Parameters
        ----------
        min_freq : float
            Lowest frequency.  Must be strictly positive.
        max_freq : float
            Highest frequency.  Must be greater than ``min_freq``.
        n_grid : int
            Number of grid points.
        spacing : str, optional
            ``"log"`` (default) for logarithmically spaced points;
            ``"linear"`` for linearly spaced points.  The spectral-mixture
            summary uses ``"log"`` to ensure adequate low-frequency
            resolution across wide dynamic ranges.

        Returns
        -------
        freq_grid : numpy.ndarray
            1-D array of ``n_grid`` frequencies in ``[min_freq, max_freq]``.

        Notes
        -----
        If ``max_freq <= min_freq`` on entry, ``max_freq`` is automatically
        adjusted to ``min_freq * 2.0`` so that the grid is always valid.

        Raises
        ------
        ValueError
            If ``min_freq <= 0`` when ``spacing="log"``.
        """
        min_freq = float(min_freq)
        max_freq = float(max_freq)
        n_grid = int(n_grid)

        if max_freq <= min_freq:
            max_freq = min_freq * 2.0

        if spacing == "log":
            if min_freq <= 0:
                raise ValueError(
                    f"min_freq must be > 0 for log spacing, "
                    f"got {min_freq!r}"
                )
            return np.logspace(
                np.log10(min_freq), np.log10(max_freq), n_grid
            )
        return np.linspace(min_freq, max_freq, n_grid)

    @staticmethod
    def _refine_peak_region(
        freq_grid, psd, params, dominant_idx,
        f_left_approx, f_right_approx,
        pad_log_factor=0.2, n_refine=None,
    ):
        """Refine the half-max crossing estimate with a denser local grid.

        Builds a fine log-spaced grid over a padded window around the
        approximate half-max interval ``[f_left_approx, f_right_approx]``,
        recomputes the PSD, re-finds the dominant peak, and walks the new
        PSD to locate the bracketing indices for both crossings.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Global log-spaced frequency grid (used for fallback bounds).
        psd : numpy.ndarray
            PSD on ``freq_grid``.
        params : dict
            Output of :meth:`_extract_sm_params`.
        dominant_idx : int
            Index of the dominant peak on ``freq_grid``.
        f_left_approx, f_right_approx : float
            Approximate left and right half-max crossing frequencies from
            the global grid.
        pad_log_factor : float, optional
            Fractional padding in log space on each side.  Default 0.2
            (i.e. widen the local window by 20 % in log units on each side).
        n_refine : int or None, optional
            Number of points in the local grid.  Defaults to
            ``max(4 * len(freq_grid), 2000)``.  The factor of 4 ensures
            the local grid is at least 4x denser than the global grid;
            the minimum of 2000 avoids a coarse local grid when the
            global grid is small.

        Returns
        -------
        freq_fine : numpy.ndarray
            Dense local frequency grid.
        psd_fine : numpy.ndarray
            PSD on ``freq_fine``.
        dominant_idx_fine : int
            Index of the dominant peak on ``freq_fine``.
        """
        from scipy.signal import find_peaks

        if n_refine is None:
            n_refine = max(4 * len(freq_grid), 2000)
        n_refine = int(n_refine)

        dom_freq = float(freq_grid[dominant_idx])

        # Pad in log space on both sides
        log_lo = np.log10(f_left_approx) - pad_log_factor
        log_hi = np.log10(f_right_approx) + pad_log_factor

        # Clamp within the global grid bounds
        log_lo = max(log_lo, np.log10(float(freq_grid[0])))
        log_hi = min(log_hi, np.log10(float(freq_grid[-1])))

        # Ensure the dominant frequency is bracketed
        if np.log10(dom_freq) < log_lo:
            log_lo = np.log10(dom_freq) - pad_log_factor
        if np.log10(dom_freq) > log_hi:
            log_hi = np.log10(dom_freq) + pad_log_factor

        if log_hi <= log_lo:
            log_hi = log_lo + 0.1

        freq_fine = np.logspace(log_lo, log_hi, n_refine)
        psd_fine = Lightcurve._sm_psd_on_grid(freq_fine, params)

        peaks_fine, _ = find_peaks(psd_fine)
        if len(peaks_fine) == 0:
            dominant_idx_fine = int(np.argmax(psd_fine))
        else:
            dominant_idx_fine = int(
                peaks_fine[np.argmax(psd_fine[peaks_fine])]
            )

        return freq_fine, psd_fine, dominant_idx_fine

    @staticmethod
    def _interpolate_halfmax_crossing(freq_grid, psd, idx, direction, half_max):
        """Linearly interpolate the frequency where PSD crosses ``half_max``.

        Given that ``psd[idx]`` is the last point **above** (or at) the
        half-maximum on one side of the dominant peak, and ``psd[idx ±1]``
        is the first point **below** it, return the linearly interpolated
        crossing frequency.

        If the neighbouring index is out of range (the crossing was at the
        very boundary of the grid), the exact grid frequency at ``idx`` is
        returned as a fallback.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            1-D frequency evaluation grid.
        psd : numpy.ndarray
            PSD values on ``freq_grid``.
        idx : int
            Index of the last point whose PSD is still at or above
            ``half_max`` on the side being interpolated.
        direction : str
            ``"left"`` or ``"right"``.  Determines which neighbour to use
            for interpolation (``idx - 1`` for left, ``idx + 1`` for right).
        half_max : float
            The half-maximum level, i.e. ``0.5 * peak_height``.

        Returns
        -------
        f_crossing : float
            Interpolated crossing frequency.
        interpolated : bool
            ``True`` if a proper bracketed interpolation was performed;
            ``False`` if the boundary fallback was used.
        """
        if direction == "left":
            neighbor = idx - 1
        else:
            neighbor = idx + 1

        if neighbor < 0 or neighbor >= len(freq_grid):
            # Boundary fallback: can't interpolate, return the grid point
            return float(freq_grid[idx]), False

        f_a = float(freq_grid[idx])
        f_b = float(freq_grid[neighbor])
        psd_a = float(psd[idx])
        psd_b = float(psd[neighbor])

        # Safeguard: if psd values don't bracket half_max (shouldn't happen
        # given how idx was found, but guard against numerical edge cases)
        if psd_a == psd_b:
            return f_a, False

        # Linear interpolation: half_max = psd_a + t * (psd_b - psd_a)
        t = (half_max - psd_a) / (psd_b - psd_a)
        f_crossing = f_a + t * (f_b - f_a)
        return float(f_crossing), True

    @staticmethod
    def _expand_psd_grid_until_contained(
        freq_grid, psd, params, dominant_idx, half_max,
        max_expansions=10, expansion_factor=2.0, n_grid=5000,
    ):
        """Expand the frequency grid until both half-max crossings are inside.

        Starting from the already-computed ``freq_grid`` / ``psd``, test
        whether the half-maximum crossings of the dominant PSD peak are
        contained within the grid.  If the left crossing is at the first
        grid point (``psd[0] >= half_max``) the low end of the grid is
        extended by dividing ``min_freq`` by ``expansion_factor``.  If the
        right crossing is at the last grid point the high end is extended by
        multiplying ``max_freq`` by ``expansion_factor``.  The dominant peak
        position is re-evaluated after each expansion to remain consistent.

        The grid is always rebuilt as a **log-spaced** grid via
        :meth:`_build_frequency_grid` to ensure adequate low-frequency
        resolution across wide dynamic ranges.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Initial evaluation grid.
        psd : numpy.ndarray
            PSD values on ``freq_grid``.
        params : dict
            Output of :meth:`_extract_sm_params`, passed to
            :meth:`_sm_psd_on_grid` for PSD recomputation.
        dominant_idx : int
            Index of the dominant PSD peak on the *current* grid.
        half_max : float
            ``0.5 * psd[dominant_idx]``.
        max_expansions : int, optional
            Maximum number of expansion iterations.  Default 10.
        expansion_factor : float, optional
            Multiplicative factor for grid edge expansion.  Default 2.0.
        n_grid : int, optional
            Number of grid points to use when rebuilding.  Default 5000.

        Returns
        -------
        freq_grid : numpy.ndarray
            Possibly expanded frequency grid.
        psd : numpy.ndarray
            PSD values on the returned ``freq_grid``.
        dominant_idx : int
            Index of the dominant peak on the returned grid.
        left_truncated : bool
            ``True`` if the left half-max crossing is still at the boundary
            after all expansion attempts.
        right_truncated : bool
            ``True`` if the right half-max crossing is still at the boundary.
        n_expansions : int
            Number of expansions that were performed.
        """
        from scipy.signal import find_peaks

        min_freq = float(freq_grid[0])
        max_freq = float(freq_grid[-1])
        n_grid = int(n_grid)
        n_expansions = 0

        for _ in range(max_expansions):
            left_truncated = psd[0] >= half_max
            right_truncated = psd[-1] >= half_max

            if not left_truncated and not right_truncated:
                break  # Both crossings are inside the grid

            if left_truncated:
                min_freq = max(min_freq / expansion_factor, 1e-12)
            if right_truncated:
                max_freq = max_freq * expansion_factor

            freq_grid = Lightcurve._build_frequency_grid(
                min_freq, max_freq, n_grid, spacing="log"
            )
            psd = Lightcurve._sm_psd_on_grid(freq_grid, params)

            # Re-find dominant peak on the new grid
            peaks, _ = find_peaks(psd)
            if len(peaks) == 0:
                dominant_idx = int(np.argmax(psd))
            else:
                dominant_idx = int(peaks[np.argmax(psd[peaks])])
            half_max = 0.5 * float(psd[dominant_idx])

            n_expansions += 1

        left_truncated = bool(psd[0] >= half_max)
        right_truncated = bool(psd[-1] >= half_max)

        return (
            freq_grid, psd, dominant_idx,
            left_truncated, right_truncated, n_expansions,
        )

    @staticmethod
    def _find_psd_peaks(freq_grid, psd):
        """Detect all local maxima in a PSD array, sorted by height.

        Returns ``(peak_indices, prominences)`` where ``peak_indices`` is a
        1-D numpy int array and ``prominences`` is a 1-D float array of the
        same length.  If no peaks are detected the global maximum is returned
        as a single peak with prominence equal to its height.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Frequency evaluation grid (unused directly; kept for API symmetry).
        psd : numpy.ndarray
            1-D PSD values.

        Returns
        -------
        peak_indices : numpy.ndarray
            Indices into ``psd`` of the detected peaks, sorted by descending
            height.
        prominences : numpy.ndarray
            Corresponding peak prominences.
        """
        from scipy.signal import find_peaks as _scipy_find_peaks

        peaks_idx, props = _scipy_find_peaks(psd, prominence=0)
        if len(peaks_idx) == 0:
            dom = int(np.argmax(psd))
            return np.array([dom]), np.array([float(psd[dom])])
        proms = props["prominences"]
        order = np.argsort(psd[peaks_idx])[::-1]
        return peaks_idx[order], proms[order]

    @staticmethod
    def _characterize_peak_basin(
        freq_grid, psd, peak_idx, mass_level=0.68
    ):
        """Characterize a single PSD peak basin.

        Finds the basin boundaries by walking left/right from the peak
        until the PSD stops decreasing, computes the peak-centered mass
        interval (which always contains the peak), and returns a summary
        dict.

        Parameters
        ----------
        freq_grid : numpy.ndarray
            Frequency evaluation grid.
        psd : numpy.ndarray
            1-D PSD values.
        peak_idx : int
            Index of the peak in ``psd``.
        mass_level : float, optional
            Fraction of basin mass to enclose.  Default 0.68.

        Returns
        -------
        info : dict
            Keys: ``height``, ``basin_left``, ``basin_right``,
            ``f_lo``, ``f_hi``, ``area_fraction``, ``mass_ok``.
        """
        peak_idx = int(peak_idx)
        height = float(psd[peak_idx])

        n = len(psd)
        left = peak_idx
        while left > 0 and psd[left - 1] < psd[left]:
            left -= 1
        right = peak_idx
        while right < n - 1 and psd[right + 1] < psd[right]:
            right += 1

        f_lo, f_hi, mass_ok = Lightcurve._compute_peak_centered_mass_interval(
            freq_grid, psd, left, right, peak_idx, mass_level=mass_level
        )

        f_basin = freq_grid[left : right + 1]
        p_basin = psd[left : right + 1]
        basin_mass = Lightcurve._integrate_logspace(p_basin, f_basin)
        total_mass = Lightcurve._integrate_logspace(psd, freq_grid)
        area_fraction = (
            basin_mass / total_mass if total_mass > 0 else float("nan")
        )

        return {
            "height": height,
            "basin_left": left,
            "basin_right": right,
            "f_lo": f_lo,
            "f_hi": f_hi,
            "area_fraction": area_fraction,
            "mass_ok": mass_ok,
        }

    @staticmethod
    def _identify_lsp_candidates(
        peaks_list,
        ratio_range=(5.0, 15.0),
        min_area_fraction=0.05,
    ):
        """Flag peaks that are candidate Long Secondary Periods (LSPs).

        A peak is flagged as a candidate LSP if its
        ``period_ratio_to_primary`` lies within ``ratio_range`` and its
        ``area_fraction`` is at least ``min_area_fraction``.

        Parameters
        ----------
        peaks_list : list[PeriodPeakResult]
            Peaks with ``period_ratio_to_primary`` already set.
        ratio_range : tuple of float, optional
            ``(min_ratio, max_ratio)`` for LSP detection.  Default
            ``(5.0, 15.0)``.
        min_area_fraction : float, optional
            Minimum basin area fraction.  Default 0.05.

        Returns
        -------
        list[PeriodPeakResult]
            Same list with ``is_candidate_lsp`` updated via
            ``dataclasses.replace()``.
        """
        updated = []
        for p in peaks_list:
            r = p.period_ratio_to_primary
            is_lsp = (
                r > 1.0
                and ratio_range[0] <= r <= ratio_range[1]
                and p.area_fraction >= min_area_fraction
            )
            updated.append(dataclasses.replace(p, is_candidate_lsp=is_lsp))
        return updated

    def _get_sm_period_summary(
        self,
        n_grid=5000,
        min_freq=None,
        max_freq=None,
        peak_threshold_rel=0.2,
        uncertainty="peak_mass",
        n_peaks=None,
        mass_level=0.68,
        classify_lsp=False,
    ):
        """Return the PSD-based period summary for a spectral-mixture model.

        Implements the core PSD-peak extraction logic used when the model
        (or the time sub-kernel of a separable 2D model) is a
        :class:`~gpytorch.kernels.SpectralMixtureKernel`.

        Parameters
        ----------
        n_grid : int, optional
            Number of points in the positive-frequency evaluation grid.
        min_freq, max_freq : float or None, optional
            Initial grid limits.  If ``None``, defaults are derived from
            the data time span and the component centres + five sigma.
            These are treated as *starting* bounds only; the grid may be
            expanded automatically.
        peak_threshold_rel : float, optional
            Relative height threshold for significant peak detection.
        uncertainty : str, optional
            Uncertainty method.  Only ``"peak_mass"`` is supported
            (``"peak_width"`` raises ``NotImplementedError``).
        n_peaks : int or None, optional
            Number of peaks to analyse.  If ``None``, defaults to the
            effective number of mixtures used at fit time.
        mass_level : float, optional
            Fraction of basin mass to enclose.  Default 0.68.
        classify_lsp : bool, optional
            If ``True``, flag candidate Long Secondary Periods.

        Returns
        -------
        summary : PeriodSummaryResult
        """
        n_grid = int(n_grid)
        params = self._extract_sm_params()

        comp_freqs = params["component_frequencies"]
        comp_scales = params["component_frequency_scales"]

        if min_freq is None:
            if self.ndim == 1:
                t_span = (
                    self._xdata_raw.max() - self._xdata_raw.min()
                ).item()
            else:
                t_col = self._xdata_raw[:, 0]
                t_span = (t_col.max() - t_col.min()).item()
            t_span = max(float(t_span), 1e-10)
            min_freq = 1.0 / t_span

        if max_freq is None:
            max_freq = float(
                np.max(comp_freqs + 5.0 * comp_scales)
            )

        min_freq = max(float(min_freq), 1e-12)
        max_freq = max(float(max_freq), min_freq * 2.0)

        freq_grid = self._build_frequency_grid(
            min_freq, max_freq, n_grid, spacing="log"
        )
        psd = self._sm_psd_on_grid(freq_grid, params)

        from scipy.signal import find_peaks as _sp_find_peaks

        _peaks, _ = _sp_find_peaks(psd)
        if len(_peaks) == 0:
            dominant_idx = int(np.argmax(psd))
        else:
            dominant_idx = int(_peaks[np.argmax(psd[_peaks])])

        peak_height = float(psd[dominant_idx])
        half_max = 0.5 * peak_height

        # -- adaptive grid expansion to contain both half-max crossings ----
        (
            freq_grid, psd, dominant_idx,
            left_truncated, right_truncated, n_expansions,
        ) = self._expand_psd_grid_until_contained(
            freq_grid, psd, params, dominant_idx, half_max,
            max_expansions=10, expansion_factor=2.0, n_grid=n_grid,
        )
        peak_height = float(psd[dominant_idx])

        # -- detect all peaks, sorted by height (descending) ---------------
        all_peak_indices, all_prominences = self._find_psd_peaks(
            freq_grid, psd
        )

        # -- determine how many peaks to analyse ---------------------------
        n_peaks_requested = n_peaks
        if n_peaks is not None:
            n_peaks_to_analyze = int(n_peaks)
        else:
            n_eff = getattr(self, "_fit_num_mixtures_effective", None)
            n_peaks_to_analyze = (
                int(n_eff) if n_eff is not None else len(all_peak_indices)
            )
        n_peaks_available = len(all_peak_indices)
        n_peaks_to_analyze = min(n_peaks_to_analyze, n_peaks_available)

        selected_indices = all_peak_indices[:n_peaks_to_analyze]
        selected_proms = all_prominences[:n_peaks_to_analyze]

        # -- characterize each selected peak --------------------------------
        dominant_freq = float(freq_grid[selected_indices[0]])
        dominant_period = 1.0 / dominant_freq

        peak_objects = []
        for rank_idx, (pidx, prom) in enumerate(
            zip(selected_indices, selected_proms, strict=True)
        ):
            info = self._characterize_peak_basin(
                freq_grid, psd, pidx, mass_level=mass_level
            )
            f_pk = float(freq_grid[pidx])
            p_pk = 1.0 / f_pk
            f_lo = info["f_lo"]
            f_hi = info["f_hi"]
            p_lo = 1.0 / f_hi if f_hi > 0 else float("nan")
            p_hi = 1.0 / f_lo if f_lo > 0 else float("nan")
            ratio = p_pk / dominant_period if dominant_period > 0 else 1.0
            # Coherence proxy: ratio of peak frequency to frequency-interval
            # width.  A narrow, well-localized peak yields a large value; a
            # broad diffuse structure yields a small value.  Non-finite or
            # non-positive widths produce NaN (treated as worst in ranking).
            _width = f_hi - f_lo
            if np.isfinite(_width) and _width > 0:
                _coherence_proxy = f_pk / _width
            else:
                _coherence_proxy = float("nan")
            peak_objects.append(
                PeriodPeakResult(
                    rank=rank_idx + 1,
                    frequency=f_pk,
                    period=p_pk,
                    height=info["height"],
                    prominence=float(prom),
                    area_fraction=info["area_fraction"],
                    interval_frequency=(f_lo, f_hi),
                    interval_period=(p_lo, p_hi),
                    period_ratio_to_primary=ratio,
                    is_candidate_lsp=False,
                    notes="",
                    coherence_proxy=_coherence_proxy,
                )
            )

        if classify_lsp:
            peak_objects = self._identify_lsp_candidates(peak_objects)

        # -- backward-compat: significant peaks via threshold ---------------
        threshold = peak_threshold_rel * peak_height
        sig_mask = psd[all_peak_indices] >= threshold
        n_sig_peaks = int(np.sum(sig_mask))

        # -- notes string ---------------------------------------------------
        dominant_info = self._characterize_peak_basin(
            freq_grid, psd, dominant_idx, mass_level=mass_level
        )
        _mass_ok = dominant_info["mass_ok"]
        _basin_l, _basin_r, _basin_left_at_bdy, _basin_right_at_bdy = (
            self._find_dominant_peak_basin(psd, dominant_idx)
        )
        _note_parts = [
            "Interval is based on the integrated PSD mass within the "
            "primary peak basin (peak-centered shortest-mass interval). "
            "The interval is guaranteed to contain the peak frequency. "
            "Integration is performed in log-frequency space to avoid "
            "high-frequency bias on a log-spaced grid. "
            "PSD evaluated on a log-spaced frequency grid."
        ]
        if _basin_left_at_bdy:
            _note_parts.append(
                "  Basin reached the left grid boundary; "
                "left edge of the basin may be underestimated."
            )
        if _basin_right_at_bdy:
            _note_parts.append(
                "  Basin reached the right grid boundary; "
                "right edge of the basin may be underestimated."
            )
        if not _mass_ok:
            _note_parts.append(
                "  WARNING: peak-mass interval could not be computed "
                "(basin too narrow); falling back to basin edges."
            )
        if n_expansions > 0:
            _note_parts.append(
                f"  Grid expanded {n_expansions} time(s) to contain "
                "the half-maximum interval."
            )
        if left_truncated or right_truncated:
            _sides = []
            if left_truncated:
                _sides.append("left")
            if right_truncated:
                _sides.append("right")
            _note_parts.append(
                f"  WARNING: half-maximum crossing on the "
                f"{' and '.join(_sides)} side(s) may still be "
                "truncated; width estimate is a lower bound."
            )
        _sm_psd_note = (
            "Spectral-mixture model (spectral_mixture backend).  "
            "Periods are derived from peaks of the SUMMED PSD on a "
            "frequency grid — this is the literature-comparable output.  "
            "The component periods/frequencies listed in the kernel "
            "component diagnostics section are direct kernel hyperparameter "
            "values and are for diagnostic purposes only; they are NOT "
            "the reported period determinations.  "
        )
        notes = _sm_psd_note + "".join(_note_parts)

        if uncertainty == "peak_width":
            raise NotImplementedError(
                "uncertainty='peak_width' is not implemented for the "
                "spectral_mixture backend because the reported interval is "
                "still computed using the peak-centered mass method. "
                "Use uncertainty='peak_mass' instead."
            )
        _interval_def = "peak_centered_68pct_mass_interval"

        _kf = self._kernel_family_name(
            getattr(self.model, "sci_kernel", None)
        )
        _diag_notes = (
            "Spectral-mixture kernel components.  These are internal kernel "
            "parameters and are NOT independent physical periods.  "
            "The summed-PSD peaks (see 'peaks') are the "
            "literature-comparable period estimates."
        )
        _diag = ComponentDiagnosticsResult(
            component_periods=params["component_periods"],
            component_frequencies=params["component_frequencies"],
            component_weights=params["component_weights"],
            component_period_scales=params["component_period_scales"],
            component_frequency_scales=(
                params["component_frequency_scales"]
            ),
            n_components=len(params["component_periods"]),
            kernel_family=_kf,
            notes=_diag_notes,
        )
        return PeriodSummaryResult(
            method="spectral_mixture_psd_peak",
            backend="spectral_mixture",
            kernel_family=_kf,
            time_kernel_family=_kf,
            has_stochastic_background=False,
            model_name=type(self.model).__name__,
            n_peaks_detected=n_sig_peaks,
            n_peaks_analyzed=len(peak_objects),
            n_peaks_requested=n_peaks_requested,
            dominant_period=dominant_period,
            dominant_frequency=dominant_freq,
            peaks=peak_objects,
            freq_grid=freq_grid,
            psd=psd,
            notes=notes,
            component_diagnostics=_diag,
            interval_definition=_interval_def,
        )

    def get_period_summary(
        self,
        n_grid=5000,
        min_freq=None,
        max_freq=None,
        peak_threshold_rel=0.2,
        uncertainty="peak_mass",
        n_peaks=None,
        mass_level=0.68,
        classify_lsp=False,
    ):
        """Return a literature-comparable period summary for the fitted model.

        Unlike :meth:`get_periods`, which returns the raw kernel-basis
        parameters of each spectral-mixture component (component centres,
        scales, and weights), this method aims to produce a *single dominant
        period* that can be directly compared to published values.

        The method dispatches to the appropriate backend based on the type of
        kernel used by the model:

        **Spectral-mixture models** (all ``"1D"``, ``"2D"``, ``"SKI"``,
        ``"PowerLaw"``, ``"Dust"`` variants):
            Constructs the total positive-frequency PSD as a sum of weighted
            Gaussians, identifies the highest PSD peak, and returns its
            location as the dominant period.  The half-maximum width of the
            peak provides a practical uncertainty interval.

        **Explicit-period models** (``"1DQuasiPeriodic"``,
        ``"1DLinearQuasiPeriodic"``):
            Reads the fitted ``period_length`` parameter directly from the
            :class:`~gpytorch.kernels.PeriodicKernel`.  The RBF lengthscale
            is used as a coherence proxy to derive a period interval and
            Q-factor.

        **Periodic-plus-stochastic** (``"1DPeriodicStochastic"``):
            Extracts the period from the quasi-periodic sub-kernel.  The
            summary notes flag the mixed periodic/stochastic nature of the
            model.

        **Separable 2D models** (``"2DSeparable"``, ``"2DAchromatic"``,
        ``"2DWavelengthDependent"``, ``"2DDustMean"``,
        ``"2DPowerLawMean"``):
            Identifies the time sub-kernel (``active_dims = [0]``) and
            applies the appropriate backend to that sub-kernel only.

        **Non-periodic models** (``"1DMatern"``):
            Returns a consistent summary dictionary with ``None`` values for
            all period-related fields rather than raising an exception, so
            that automated scripts can handle all model types gracefully.

        .. note::
            All uncertainty estimates are *practical proxies*, not posterior
            credible intervals.  MCMC-based credible intervals are not yet
            implemented.

        Parameters
        ----------
        n_grid : int, optional
            Number of points in the positive-frequency evaluation grid
            (spectral-mixture backend only).  Default 5000.
        min_freq : float or None, optional
            Minimum frequency for the evaluation grid (SM backend only).
            Defaults to ``1 / time_span``.
        max_freq : float or None, optional
            Maximum frequency for the evaluation grid (SM backend only).
            Defaults to the highest component centre plus five sigma.
        peak_threshold_rel : float, optional
            Relative height threshold for significant peaks (SM backend).
            Default 0.2.
        uncertainty : str, optional
            Uncertainty method.  Only ``"peak_mass"`` is supported for the
            spectral-mixture backend (``"peak_width"`` raises
            ``NotImplementedError``).  Non-SM backends always use their
            native interval method and ignore this parameter.  Default
            ``"peak_mass"``.
        n_peaks : int or None, optional
            Number of peaks to analyze and return in ``peaks``.  If ``None``
            (default), defaults to ``_fit_num_mixtures_effective`` when that
            attribute is available (i.e. after a call to :meth:`fit` or
            :meth:`set_model`), otherwise all detected peaks are returned.
            Pass an explicit integer to override.
        mass_level : float, optional
            Fraction of basin mass to enclose in the equal-tail interval
            (``"peak_mass"`` mode only).  Default 0.68 (~1 sigma).
        classify_lsp : bool, optional
            If ``True``, flag peaks whose period ratio to the dominant peak
            falls within the Long Secondary Period range (5-15) and whose
            basin area fraction exceeds 0.05.  Default ``False``.

        Returns
        -------
        summary : dict
            Dictionary with keys:

            * ``component_periods``          - raw kernel component periods
            * ``component_weights``          - raw kernel component weights
            * ``component_period_scales``    - raw kernel period widths
            * ``component_frequencies``      - raw kernel component freqs
            * ``component_frequency_scales`` - raw kernel frequency widths
            * ``freq_grid``  - evaluation grid (``None`` for non-PSD backends)
            * ``psd``        - PSD values (``None`` for non-PSD backends)
            * ``dominant_frequency`` - frequency of the dominant peak
              (``None`` for non-periodic models)
            * ``dominant_period``    - ``1 / dominant_frequency``
              (``None`` for non-periodic models)
            * ``period_interval_fwhm_like`` - ``(period_lo, period_hi)``
              uncertainty interval (``None`` for non-periodic models;
              kept for backward compatibility)
            * ``period_interval`` - same as ``period_interval_fwhm_like``
              (generic key independent of uncertainty method)
            * ``interval_definition`` - string describing the interval type
            * ``q_factor``        - coherence Q (``None`` if not defined)
            * ``peak_fraction``   - dominant peak height / total weight
            * ``n_significant_peaks`` - peaks above threshold
            * ``significant_periods`` - periods of significant peaks
            * ``method``  - string identifying the backend used
            * ``notes``   - additional diagnostic notes

        Raises
        ------
        RuntimeError
            If the model has not been initialised.
        NotImplementedError
            If an unsupported ``uncertainty`` method is requested.
        """
        self._raise_if_fit_failed("GP period summary")
        _sm_uncertainties = {"peak_mass"}
        if uncertainty not in _sm_uncertainties:
            raise NotImplementedError(
                f"uncertainty='{uncertainty}' is not yet implemented. "
                f"Supported values: {sorted(_sm_uncertainties)!r}."
            )

        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError(
                "Model not initialised.  Call set_model() first."
            )

        multicomp_summary = self._get_consensus_multicomp_period_summary()
        if multicomp_summary is not None:
            return multicomp_summary

        backend = self._detect_period_summary_backend()

        if backend == "spectral_mixture":
            return self._get_sm_period_summary(
                n_grid=n_grid,
                min_freq=min_freq,
                max_freq=max_freq,
                peak_threshold_rel=peak_threshold_rel,
                uncertainty=uncertainty,
                n_peaks=n_peaks,
                mass_level=mass_level,
                classify_lsp=classify_lsp,
            )

        if backend == "explicit_period":
            return self._get_explicit_period_summary()

        if backend == "periodic_plus_stochastic":
            return self._get_periodic_plus_stochastic_summary()

        if backend == "separable_2d":
            return self._get_separable_2d_period_summary(
                n_grid=n_grid,
                min_freq=min_freq,
                max_freq=max_freq,
                peak_threshold_rel=peak_threshold_rel,
                uncertainty=uncertainty,
                n_peaks=n_peaks,
                mass_level=mass_level,
                classify_lsp=classify_lsp,
            )

        # backend == "non_periodic"
        return self._get_non_periodic_summary()

    def plot_period_summary(
        self,
        summary=None,
        show=True,
        log_freq=True,
        show_full_psd=None,
        max_peaks_to_mark=3,
        log_y=True,
        close=False,
        annotate_provenance=False,
        provenance_location="lower left",
        **kwargs,
    ):
        """Plot the period summary from :meth:`get_period_summary`.

        Produces a matplotlib figure appropriate for the type of period
        summary:

        * **Spectral-mixture PSD summary with a single analyzed peak**
          (``PeriodSummaryResult``, ``n_peaks_analyzed == 1``): generates a
          **single peak-centered panel** zoomed in on the dominant peak.
          Pass ``show_full_psd=True`` to add a second full-range PSD panel.
        * **Spectral-mixture PSD summary with structured peaks**
          (``PeriodSummaryResult``, ``n_peaks_analyzed > 1``): generates a
          **multi-panel figure** with the full PSD in the top panel and one
          zoomed panel per analyzed peak below.  Each peak is labeled
          P1, P2, … with a distinct color.
        * **Spectral-mixture PSD summary (plain dict)**: plots the PSD curve
          with the dominant peak and dotted lines for other significant peaks.
        * **Explicit-period summary** (e.g. quasi-periodic): plots a single
          vertical line at the dominant frequency with an annotated period,
          interval, and Q-factor.  No PSD curve is drawn because none is
          computed for this backend.
        * **Non-periodic summary**: produces a simple figure with explanatory
          text stating that no dominant period is defined for this kernel.

        The figure type is determined by ``summary["method"]`` and by whether
        ``summary["freq_grid"]`` is ``None``.

        Parameters
        ----------
        summary : dict or None, optional
            Output of :meth:`get_period_summary`.  If ``None``, it is
            computed automatically.  Extra keyword arguments (``**kwargs``)
            are forwarded to :meth:`get_period_summary`.
        show : bool, optional
            If ``True`` (default), call ``plt.show()``.  If ``False``,
            return ``(fig, ax)`` for further customisation.
        log_freq : bool, optional
            If ``True`` (default), plot the x-axis (frequency) on a log
            scale.  Ignored for non-periodic summaries.
        log_y : bool, optional
            If ``True`` (default), plot the y-axis (PSD) on a log scale.
            The lower y-axis limit is clamped automatically so that at most
            10 decades below the maximum PSD value in each panel are shown,
            preventing near-zero noise from compressing the useful range.
            Set to ``False`` to use a linear y-axis.  Ignored for
            non-periodic summaries and for panels where no PSD is drawn.
        show_full_psd : bool or None, optional
            Controls whether a full-range PSD panel is included in the
            single-peak case.  When ``None`` (default), a full-range panel
            is *not* added in single-peak mode (the main panel is already
            peak-centered) but *is* included in multi-peak mode.  Set to
            ``True`` to force a full-range panel even in single-peak mode;
            set to ``False`` to suppress it even in multi-peak mode.
        max_peaks_to_mark : int, optional
            Maximum number of peaks to mark on the plot.  In multi-peak
            mode this also limits the number of zoom panels created.
            Default is ``3``.
        close : bool, optional
            If True and show=True, close the figure immediately after displaying it.
            This is useful in notebooks or loops where many figures are generated.
            Ignored when show=False, because the figure is returned to the caller.
        annotate_provenance : bool, optional
            If ``True``, annotate the plot with lightweight fit provenance
            (model, runtime, timestamp) from the most recent fit-history entry.
            Default is ``False``.
        provenance_location : {"lower left", "lower right", "upper left",
            "upper right"}, optional
            Axes-relative location for provenance annotations when
            ``annotate_provenance=True``. Default is ``"lower left"``.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`get_period_summary` when ``summary`` is ``None``.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            Returned when ``show=False``; otherwise ``None``.
            For the multi-panel case ``ax`` is the top axes.
        """
        self._raise_if_fit_failed("period summary plot")
        if summary is None:
            summary = self.get_period_summary(**kwargs)

        method = summary.get("method", "")
        has_psd = summary["freq_grid"] is not None

        # -- non-periodic: informational plot only -------------------------
        if method == "non_periodic_kernel" or (
            summary["dominant_period"] is None
        ):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.text(
                0.5,
                0.5,
                summary.get(
                    "notes",
                    "No dominant period defined for this kernel.",
                ),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                wrap=True,
            )
            ax.set_axis_off()
            ax.set_title("Period summary")
            if annotate_provenance:
                self._plot_fit_history_provenance(
                    ax,
                    provenance_location=provenance_location,
                )
            if show:
                plt.show()
                if close:
                    plt.close(fig)
                return None
            return fig, ax

        # -- common fields -------------------------------------------------
        f_peak = summary["dominant_frequency"]
        p_dom = summary["dominant_period"]
        # Prefer the generic key; fall back to the legacy key for old summaries
        interval = summary.get(
            "period_interval", summary.get("period_interval_fwhm_like")
        )
        interval_definition = summary.get("interval_definition", "")
        q = summary["q_factor"]
        n_sig = summary["n_significant_peaks"]

        # Build a human-readable interval type label for annotations
        _interval_labels = {
            "equal_tail_68pct_peak_mass": "68% peak mass interval",
            "peak_centered_68pct_mass_interval": "68% peak-centered mass interval",
            "half_maximum_fwhm_like": "half-max interval",
            "coherence_proxy": "coherence-proxy interval",
            "coherence_proxy_from_rbf_lengthscale": (
                "coherence-proxy interval (RBF lengthscale)"
            ),
        }
        interval_label = _interval_labels.get(
            interval_definition, interval_definition or "interval"
        )

        # Decide whether we have a structured PeriodSummaryResult with peaks.
        # Plain-dict summaries (non-SM backends) do not have a .peaks attr.
        structured_peaks = getattr(summary, "peaks", None)
        has_structured_peaks = (
            structured_peaks is not None and len(structured_peaks) > 0
        )

        # -- colour palette for per-peak markers ---------------------------
        # crimson = P1 (dominant), then cycling through a friendly palette
        _peak_colors = [
            "crimson",
            "darkorange",
            "forestgreen",
            "mediumpurple",
            "saddlebrown",
            "deepskyblue",
        ]

        def _peak_color(rank):
            """Return the color for a peak by rank (1-indexed)."""
            idx = max(rank - 1, 0)
            return _peak_colors[idx % len(_peak_colors)]

        # Maximum decades shown below the PSD peak on a log y-axis
        _MAX_LOG_Y_DEC = 10

        def _clamp_log_ylim(panel_ax, psd_visible):
            """Clamp log y-axis to at most _MAX_LOG_Y_DEC decades."""
            pos = psd_visible[
                np.isfinite(psd_visible) & (psd_visible > 0)
            ]
            if pos.size == 0:
                return
            y_top = float(pos.max())
            panel_ax.set_ylim(bottom=y_top * 10.0 ** (-_MAX_LOG_Y_DEC))

        # ------------------------------------------------------------------
        # Helpers shared by both structured-peak plot paths
        # ------------------------------------------------------------------
        def _zoom_window(pk, freq_grid):
            """Return (f_win_lo, f_win_hi, f_zoom, p_zoom) for one peak.

            The window is centered on the peak and expanded symmetrically
            around it.  If the interval bounds are finite and sensible the
            interval half-width is used as the core; otherwise a ±25%
            fallback is applied.  A ±10% emergency fallback is used when
            the resulting slice is too narrow.
            """
            f_ctr = pk.frequency
            p_lo, p_hi = pk.interval_period
            if (
                np.isfinite(p_lo) and np.isfinite(p_hi)
                and p_lo > 0 and p_hi > 0
            ):
                f_int_lo = 1.0 / p_hi
                f_int_hi = 1.0 / p_lo
                # Half-width of the interval, but at least 10% of peak freq
                half = max(0.5 * (f_int_hi - f_int_lo), 0.1 * f_ctr)
                # Expand by 50% symmetrically around the peak
                f_win_lo = max(f_ctr - 1.5 * half, freq_grid[0])
                f_win_hi = min(f_ctr + 1.5 * half, freq_grid[-1])
            else:
                # Fallback: ±25% symmetric window
                half = 0.25 * f_ctr
                f_win_lo = max(f_ctr - half, freq_grid[0])
                f_win_hi = min(f_ctr + half, freq_grid[-1])
            mask = (freq_grid >= f_win_lo) & (freq_grid <= f_win_hi)
            f_zoom = freq_grid[mask]
            p_zoom = psd[mask]
            if len(f_zoom) < 2:
                # Emergency: ±10% around peak
                f_win_lo = f_ctr * 0.9
                f_win_hi = f_ctr * 1.1
                mask = (freq_grid >= f_win_lo) & (freq_grid <= f_win_hi)
                f_zoom = freq_grid[mask]
                p_zoom = psd[mask]
            return f_win_lo, f_win_hi, f_zoom, p_zoom

        def _draw_peak_zoom(panel_ax, pk, f_win_lo, f_win_hi,
                            f_zoom, p_zoom):
            """Populate a zoom panel for one peak."""
            col = _peak_color(pk.rank)
            panel_ax.plot(f_zoom, p_zoom, color="steelblue", lw=1.5)
            panel_ax.axvline(pk.frequency, color=col, lw=1.5, ls="--")
            p_lo, p_hi = pk.interval_period
            if (
                np.isfinite(p_lo) and np.isfinite(p_hi) and p_lo > 0
            ):
                f_lo_int = 1.0 / p_hi
                f_hi_int = 1.0 / p_lo
                # Always draw the span when the interval is valid; matplotlib
                # clips it to the axes automatically, so there is no need to
                # check whether it fits inside the zoom window.
                if f_lo_int < f_hi_int:
                    panel_ax.axvspan(
                        f_lo_int, f_hi_int,
                        alpha=0.25, color=col,
                        label=(
                            f"{interval_label}  "
                            f"[{p_lo:.4g}, {p_hi:.4g}]"
                        ),
                    )
            _ratio_str = (
                f"  ratio={pk.period_ratio_to_primary:.3g}"
                if pk.rank > 1
                else ""
            )
            panel_ax.set_title(
                f"P{pk.rank}  period = {pk.period:.6g}{_ratio_str}"
            )
            if log_freq:
                panel_ax.set_xscale("log")
            if log_y:
                panel_ax.set_yscale("log")
                _clamp_log_ylim(panel_ax, p_zoom)
            panel_ax.set_xlabel("Frequency")
            panel_ax.set_ylabel("PSD")
            panel_ax.legend(fontsize=7, loc="upper left")

        # ------------------------------------------------------------------
        # Structured PeriodSummaryResult with PSD available
        # ------------------------------------------------------------------
        if has_structured_peaks and has_psd:
            freq_grid = summary["freq_grid"]
            psd = summary["psd"]
            # Limit to max_peaks_to_mark peaks
            _peaks_to_plot = structured_peaks[:max_peaks_to_mark]
            _n_peaks = len(_peaks_to_plot)
            # Determine whether we are in single-peak mode.
            # show_full_psd=None means: auto (False for 1 peak, True for >1).
            _single_peak = _n_peaks == 1
            _include_full = (
                show_full_psd
                if show_full_psd is not None
                else not _single_peak
            )

            if _single_peak:
                # -------------------------------------------------------
                # Single-peak mode: one peak-centered panel (+ optional
                # full-PSD panel if show_full_psd=True was requested).
                # -------------------------------------------------------
                pk = _peaks_to_plot[0]
                col = _peak_color(pk.rank)
                f_win_lo, f_win_hi, f_zoom, p_zoom = _zoom_window(
                    pk, freq_grid
                )

                if _include_full:
                    fig, axes = plt.subplots(
                        2, 1, figsize=(9, 7), squeeze=False
                    )
                    axes = axes[:, 0]
                    ax = axes[0]  # main = peak-centered
                    ax_full = axes[1]
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
                    ax_full = None

                # Main panel: peak-centered zoom
                _draw_peak_zoom(ax, pk, f_win_lo, f_win_hi, f_zoom, p_zoom)
                ax.set_title(
                    f"Period summary - dominant peak  "
                    f"(P = {pk.period:.6g})"
                )

                if ax_full is not None:
                    # Optional full-range panel below
                    ax_full.plot(
                        freq_grid, psd,
                        color="steelblue", lw=1.5, label="PSD"
                    )
                    ax_full.axvline(
                        pk.frequency, color=col, lw=1.5, ls="--",
                        label=f"P1  period={pk.period:.4g}",
                    )
                    p_lo_fp, p_hi_fp = pk.interval_period
                    if (
                        np.isfinite(p_lo_fp) and np.isfinite(p_hi_fp)
                        and p_lo_fp > 0 and p_hi_fp > 0
                    ):
                        f_lo_int = 1.0 / p_hi_fp
                        f_hi_int = 1.0 / p_lo_fp
                        if f_lo_int < f_hi_int:
                            ax_full.axvspan(
                                f_lo_int, f_hi_int,
                                alpha=0.15, color=col,
                                label=(
                                    f"{interval_label}  "
                                    f"[{p_lo_fp:.4g}, {p_hi_fp:.4g}]"
                                ),
                            )
                    if log_freq:
                        ax_full.set_xscale("log")
                    if log_y:
                        ax_full.set_yscale("log")
                        _clamp_log_ylim(ax_full, psd)
                    ax_full.set_ylabel("PSD")
                    ax_full.set_title(
                        f"Period summary - full PSD ({method})"
                    )
                    ax_full.legend(fontsize=7, loc="upper left", ncol=2)

            else:
                # -------------------------------------------------------
                # Multi-peak mode: full PSD top + one zoom panel per peak
                # (limited to max_peaks_to_mark peaks)
                # -------------------------------------------------------
                n_panels = 1 + _n_peaks
                fig, axes = plt.subplots(
                    n_panels, 1,
                    figsize=(9, 3.5 + 2.5 * _n_peaks),
                    squeeze=False,
                )
                axes = axes[:, 0]
                ax = axes[0]  # top panel = full PSD

                # Top panel: full PSD
                ax.plot(
                    freq_grid, psd, color="steelblue", lw=1.5, label="PSD"
                )
                for pk in _peaks_to_plot:
                    col = _peak_color(pk.rank)
                    ax.axvline(
                        pk.frequency,
                        color=col,
                        lw=1.5,
                        ls="--",
                        label=f"P{pk.rank}  period={pk.period:.4g}",
                    )
                    p_lo, p_hi = pk.interval_period
                    if (
                        np.isfinite(p_lo) and np.isfinite(p_hi)
                        and p_lo > 0 and p_hi > 0
                    ):
                        f_lo_int = 1.0 / p_hi
                        f_hi_int = 1.0 / p_lo
                        if f_lo_int < f_hi_int:
                            _span_label = (
                                f"{interval_label}  "
                                f"[{p_lo:.4g}, {p_hi:.4g}]"
                                if pk.rank == 1
                                else None
                            )
                            ax.axvspan(
                                f_lo_int, f_hi_int,
                                alpha=0.15, color=col,
                                label=_span_label,
                            )
                if log_freq:
                    ax.set_xscale("log")
                if log_y:
                    ax.set_yscale("log")
                    _clamp_log_ylim(ax, psd)
                ax.set_ylabel("PSD")
                ax.set_title(f"Period summary - full PSD ({method})")
                ax.legend(fontsize=7, loc="upper left", ncol=2)

                # Per-peak zoom panels (one per plotted peak)
                for i, pk in enumerate(_peaks_to_plot):
                    panel_ax = axes[i + 1]
                    f_win_lo, f_win_hi, f_zoom, p_zoom = _zoom_window(
                        pk, freq_grid
                    )
                    _draw_peak_zoom(
                        panel_ax, pk, f_win_lo, f_win_hi, f_zoom, p_zoom
                    )

            fig.tight_layout()
            if annotate_provenance:
                self._plot_fit_history_provenance(
                    ax,
                    provenance_location=provenance_location,
                )
            if show:
                plt.show()
                return None
            return fig, ax

        # ------------------------------------------------------------------
        # Single-panel fallback (non-structured or no PSD)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # -- PSD curve (spectral-mixture only) -----------------------------
        if has_psd:
            freq_grid = summary["freq_grid"]
            psd = summary["psd"]
            ax.plot(
                freq_grid, psd, color="steelblue", lw=1.5, label="PSD"
            )

        # -- dominant peak marker -----------------------------------------
        ax.axvline(
            f_peak,
            color="crimson",
            lw=1.5,
            ls="--",
            label=f"Dominant peak  P = {p_dom:.4g}",
        )

        # -- period interval shaded band (if finite interval) --------------
        if interval is not None:
            period_lo, period_hi = interval
            f_left = (
                1.0 / period_hi if period_hi and period_hi > 0 else None
            )
            f_right = (
                1.0 / period_lo if period_lo and period_lo > 0 else None
            )
            if (
                f_left is not None and f_right is not None
                and np.isfinite(f_left) and np.isfinite(f_right)
                and f_left < f_right
            ):
                ax.axvspan(
                    f_left,
                    f_right,
                    alpha=0.25,
                    color="crimson",
                    label=(
                        f"{interval_label}  "
                        f"[{period_lo:.4g}, {period_hi:.4g}]"
                    ),
                )

        # -- other significant peaks from structured summary ---------------
        if has_structured_peaks:
            for pk in structured_peaks[1:max_peaks_to_mark]:
                col = _peak_color(pk.rank)
                ax.axvline(
                    pk.frequency,
                    color=col,
                    lw=1.0,
                    ls=":",
                    alpha=0.9,
                    label=f"P{pk.rank}  period={pk.period:.4g}",
                )
        else:
            sig_periods = summary.get("significant_periods", np.array([]))
            for sp in sig_periods:
                sf = 1.0 / sp
                if abs(sf - f_peak) > 1e-12 * max(f_peak, 1e-12):
                    ax.axvline(
                        sf,
                        color="darkorange",
                        lw=1.0,
                        ls=":",
                        alpha=0.8,
                    )

        # -- text annotation -----------------------------------------------
        if q is not None and np.isfinite(q):
            q_str = f"{q:.2f}"
        elif q is not None and np.isinf(q):
            q_str = "inf"
        else:
            q_str = "N/A"

        if interval is not None:
            p_lo, p_hi = interval
            int_str = f"[{p_lo:.4g}, {p_hi:.4g}]"
        else:
            int_str = "N/A"

        ann_lines = [
            f"Dominant period:   {p_dom:.6g}",
            f"Interval ({interval_label}): {int_str}",
            f"Q-factor:          {q_str}",
            f"Significant peaks: {n_sig}",
        ]
        ax.text(
            0.97,
            0.97,
            "\n".join(ann_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", alpha=0.8
            ),
        )

        if log_freq:
            ax.set_xscale("log")
        if has_psd and log_y:
            ax.set_yscale("log")
            _clamp_log_ylim(ax, psd)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("PSD" if has_psd else "")
        ax.set_title(f"Period summary ({method})")
        ax.legend(fontsize=8, loc="upper left")
        if annotate_provenance:
            self._plot_fit_history_provenance(
                ax,
                provenance_location=provenance_location,
            )

        if show:
            plt.show()
            return None
        return fig, ax

    # ------------------------------------------------------------------
    # High-level output-writing convenience
    # ------------------------------------------------------------------

    def _save_period_summary_figure(
        self,
        summary,
        filename,
        plot_kwargs=None,
        close_figure=True,
        dpi=150,
    ):
        """Internal helper for write_period_summary_outputs().

        Calls :meth:`plot_period_summary` with ``show=False``, saves the
        resulting figure, and optionally closes it.

        Parameters
        ----------
        summary : dict or PeriodSummaryResult
            Pre-computed period summary (passed straight through to
            :meth:`plot_period_summary`).
        filename : str or Path-like
            Destination path for the PNG (or any format supported by
            matplotlib's ``savefig``).
        plot_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :meth:`plot_period_summary`.
        close_figure : bool, optional
            If ``True`` (default), call ``plt.close(fig)`` after saving.
        dpi : int, optional
            Resolution in dots per inch, default ``150``.

        Returns
        -------
        pathlib.Path
            Path to the saved figure file (same as *filename* as a
            ``pathlib.Path``; may be relative).
        """
        from pathlib import Path

        if plot_kwargs is None:
            plot_kwargs = {}
        path = Path(filename)
        result = self.plot_period_summary(
            summary=summary, show=False, **plot_kwargs
        )
        # plot_period_summary returns None when show=True; that should not
        # happen here (we always pass show=False), but guard defensively.
        if result is None:
            return path
        fig, _ax = result
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if close_figure:
            plt.close(fig)
        return path

    def write_period_summary_outputs(
        self,
        text_file=None,
        png_file=None,
        json_file=None,
        summary=None,
        show=False,
        close_figure=True,
        include_components=True,
        include_peaks=True,
        include_psd_info=False,
        include_psd_in_json=False,
        include_fit_history=False,
        summary_kwargs=None,
        plot_kwargs=None,
    ):
        """Write period-summary outputs (text, PNG, JSON) to disk.

        This is a high-level **convenience wrapper** around:

        * :meth:`get_period_summary` — computes the summary if not supplied
        * :meth:`PeriodSummaryResult.write_text` — human-readable text report
        * :meth:`_save_period_summary_figure` — period-summary figure (PNG)
        * :meth:`PeriodSummaryResult.write_json` — machine-readable JSON export

        The method writes only the files whose paths are provided. Pass
        ``text_file``, ``png_file``, and/or ``json_file`` in any combination.

        Parameters
        ----------
        text_file : str, Path-like, or None, optional
            If given, the human-readable period-summary text is written here.
            The output is intended for direct reading by a researcher: it
            includes the dominant period, peak table, kernel-component
            diagnostics, and (optionally) PSD grid information.
        png_file : str, Path-like, or None, optional
            If given, the period-summary figure is saved here.  The PNG is a
            visualisation of the analyzed peak structure produced by
            :meth:`plot_period_summary`.
        json_file : str, Path-like, or None, optional
            If given, a machine-readable JSON export is written here.  The
            JSON contains the same information as the text report plus the
            raw array data (unless *include_psd_in_json* is ``False``).
        summary : dict or PeriodSummaryResult or None, optional
            A pre-computed period summary returned by
            :meth:`get_period_summary`.  If ``None`` (default) the summary is
            computed by calling ``get_period_summary(**summary_kwargs)``.
            Supplying a pre-computed summary avoids redundant computation when
            multiple output files are requested.
        show : bool, optional
            Passed through to :meth:`plot_period_summary`.  Ignored when
            *png_file* is ``None``.  Default is ``False``.
        close_figure : bool, optional
            If ``True`` (default), close the matplotlib figure after saving.
            Set to ``False`` to keep the figure in memory for further
            inspection.
        include_components : bool, optional
            Forwarded to :meth:`PeriodSummaryResult.write_text`.  Controls
            whether the kernel-component diagnostics block appears in the text
            output.  Default is ``True``.
        include_peaks : bool, optional
            Forwarded to :meth:`PeriodSummaryResult.write_text`.  Controls
            whether the analyzed-peaks block appears in the text output.
            Default is ``True``.
        include_psd_info : bool, optional
            Forwarded to :meth:`PeriodSummaryResult.write_text`.  Controls
            whether PSD grid statistics appear in the text output.  Default is
            ``False``.
        include_psd_in_json : bool, optional
            Forwarded to :meth:`PeriodSummaryResult.write_json`.  When
            ``True`` the full frequency grid and PSD arrays are embedded in
            the JSON file.  Default is ``False`` (arrays are omitted to keep
            the file small).
        include_fit_history : bool, optional
            Forwarded to :meth:`PeriodSummaryResult.write_json`.  When ``True``
            the current ``Lightcurve.fit_history`` is included in the exported
            JSON for reproducibility/debug provenance.  Default is ``False``.
        summary_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :meth:`get_period_summary`
            when *summary* is ``None``.  Ignored if *summary* is supplied.
        plot_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :meth:`plot_period_summary`
            (and thus to :meth:`_save_period_summary_figure`).  Ignored when
            *png_file* is ``None``.

        Returns
        -------
        PeriodSummaryResult or dict
            The period summary (computed or passed in).

        Examples
        --------
        Write all three output types in one call::

            lc.write_period_summary_outputs(
                text_file="results/summary.txt",
                png_file="results/summary.png",
                json_file="results/summary.json",
            )

        Reuse an existing summary to avoid recomputation::

            s = lc.get_period_summary()
            lc.write_period_summary_outputs(
                summary=s,
                text_file="results/summary.txt",
                png_file="results/summary.png",
            )
        """
        if summary_kwargs is None:
            summary_kwargs = {}
        if summary is None:
            summary = self.get_period_summary(**summary_kwargs)
        elif summary_kwargs:
            warnings.warn(
                "summary_kwargs are ignored because a pre-computed summary "
                "was supplied via the summary= argument.",
                UserWarning,
                stacklevel=2,
            )

        if text_file is not None:
            summary.write_text(
                text_file,
                include_components=include_components,
                include_peaks=include_peaks,
                include_psd_info=include_psd_info,
            )

        if json_file is not None:
            summary.write_json(
                json_file,
                include_psd=include_psd_in_json,
                include_fit_history=include_fit_history,
                fit_history=self.get_fit_history() if include_fit_history else None,
            )

        if png_file is not None:
            self._save_period_summary_figure(
                summary,
                png_file,
                plot_kwargs=plot_kwargs,
                close_figure=close_figure,
            )

        return summary

    def get_parameters(self, raw=False, transform=True):
        """
        Returns a dictionary of the parameters of the model, with the keys
        being the names of the parameters and the values being the values of
        the parameters. This is useful for getting the values of the parameters
        after training, for example.

        The routine is rather hacky, since there is no built-in way to get the
        unconstrained values of the parameters from the model without knowing
        exactly what they are ahead of time. This routine therefore gets the
        names of the raw parameters, and then uses those names with string
        manipulation and `__getattr__` to get the values of the constrained
        parameters.

        Parameters
        ----------
        raw : bool, default False
            If True, returns the raw values of the parameters, otherwise
            returns the constrained values of the parameters.

        Returns
        -------
        pars : dict
            A dictionary of the parameters of the model, with the keys
            being the names of the parameters and the values being the values
            of the parameters.
        """
        pars = {}
        pars_to_transform = {
            "x": ["mixture_means", "mixture_scales"],
            "y": ["noise", "mean_module"],
        }
        for param_name, param in self.model.named_parameters():
            comps = list(param_name.split("."))
            if not raw and "raw" in param_name:
                # This is a constrained parameter, so we need to get the
                # unconstrained value
                pn = ".".join([c.lstrip("raw_") for c in comps])
                tmp = self.model.__getattr__(comps[0])
                for i in range(1, len(comps)):
                    c = comps[i] if "raw" not in comps[i] else comps[i].lstrip("raw_")
                    try:
                        tmp = tmp.__getattr__(c)
                    except AttributeError:
                        tmp = tmp.__getattribute__(c)
                if (
                    any(p in pn for p in pars_to_transform["x"])
                    and transform
                    and self.xtransform is not None
                ):
                    d = 1 / self.xtransform.inverse(1 / tmp.data, shift=False)
                elif (
                    any(p in pn for p in pars_to_transform["y"])
                    and transform
                    and self.ytransform is not None
                ):
                    d = self.ytransform.inverse(tmp.data)
                else:
                    d = tmp.data
                pars[pn] = d
            else:
                # Either we actually want the raw values, or it's not a
                # constrained parameter
                if (
                    any(p in param_name for p in pars_to_transform["x"])
                    and transform
                    and self.xtransform is not None
                ):
                    d = 1 / self.xtransform.inverse(1 / param.data, shift=False)
                elif (
                    any(p in param_name for p in pars_to_transform["y"])
                    and transform
                    and self.ytransform is not None
                ):
                    d = self.ytransform.inverse(param.data)
                else:
                    d = param.data
                pars[param_name] = d
        return pars

    def print_parameters(self, raw=False):
        """
        Prints the parameters of the model, with the keys being the names of
        the parameters and the values being the values of the parameters. This
        is useful for getting the values of the parameters after training, for
        example.

        Parameters
        ----------
        raw : bool, default False
            If True, prints the raw values of the parameters, otherwise prints
            the constrained values of the parameters.

        """
        pars = self.get_parameters(raw=raw)
        for key, value in pars.items():
            print(f"{key}: {value}")

    def print_results(self):
        for key in self.results.keys():
            results_tmp = self.results[key][-1]
            results_tmp_shape = results_tmp.shape  # e.g. (4,1,1)
            results_tmp_shape_len = len(results_tmp.shape)
            if results_tmp_shape_len == 1:
                print(f"{key}: {results_tmp}")
            else:
                sum_over_shape = sum(j > 1 for j in results_tmp_shape)
                if sum_over_shape in [0, 1]:
                    print(f"{key}: {results_tmp.flatten()}")
                elif sum_over_shape == 2:
                    for i in range(results_tmp.shape[-1]):
                        print(f"{key}: {results_tmp[...,i].flatten()}")

    def plot_psd(
        self,
        freq=None,
        means=None,
        scales=None,
        weights=None,
        show=True,
        raw=False,
        log=(True, False),
        truncate_psd=True,
        logpsd=False,
        mcmc_samples=False,
        **kwargs,
    ):
        """Plot the power spectral density of the model

        Parameters
        ----------
        freq : array_like, optional
            The frequencies at which to compute the PSD, by default None. If
            None, the frequencies will be computed automatically.
        means : array_like, optional
            The means of the gaussians in the spectral mixture kernel, by
            default None. If None, the means from the model will be used.
        scales : array_like, optional
            The scales of the gaussians in the spectral mixture kernel, by
            default None. If None, the scales from the model will be used.
        weights : array_like, optional
            The weights of the gaussians in the spectral mixture kernel, by
            default None. If None, the weights from the model will be used.
        show : bool, optional
            Whether to show the plot, by default True.
        raw : bool, optional
            If True, the PSD will be computed in the space that the model was
            trained in, by default False. If False, the PSD will be computed
            in the original space of the data.
        log : tuple, optional
            A tuple of two booleans, indicating whether to plot the x-axis and
            y-axis on a log scale, respectively, by default (True, False).
        truncate_psd : float or bool, optional
            If not False, the PSD will be truncated at this value, by default
            True. This is useful for speeding up plotting when the frequency
            range is large. If logpsd is True, this value should be given in
            (natural) log space. If truncate_psd is True, the PSD will be
            truncated at 1e-6 times the maximum PSD for logpsd=False, and 1e-15
            of the maximum PSD (i.e. max(ln(psd)) - 34.5388) for logpsd=True.
        logpsd : bool, optional
            If True, the PSD will be plotted on a log scale, by default False.
            If True, truncate_psd must be given in (natural) log space.
        mcmc_samples : bool, optional
            If True, many sample PSDs will be plotted using the MCMC samples,
            by default False. This will only work if the model has been fitted
            using MCMC.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
            The figure and axes objects of the plot.
        """
        self._raise_if_fit_failed("PSD plot")

        if freq is None:
            if self.ndim == 1:
                if raw:
                    # our step size only needs to be small enough to resolve
                    # the width of the narrowest gaussian
                    step = self.model.sci_kernel.mixture_scales.min() / 5
                    # this isn't really the correct way to do this, but it will
                    # do for now
                    diffs = (
                        self._xdata_transformed.sort().values[1:]
                        - self._xdata_transformed.sort().values[:-1]
                    )
                    mindelta = (diffs[diffs > 0]).min().item()
                    freq = torch.arange(
                        1
                        / (
                            self._xdata_transformed.max()
                            - self._xdata_transformed.min()
                        ).item(),
                        1 / (mindelta),
                        step.item(),
                    )
                else:
                    # we have to transform the step size to the original space
                    # to get the correct frequency range
                    step = 1 / self.xtransform.inverse(
                        1 / (self.model.sci_kernel.mixture_scales.min() / 5),
                        shift=False,
                    )
                    # this isn't really the correct way to do this, but it will
                    # do for now
                    diffs = (
                        self._xdata_raw.sort().values[1:]
                        - self._xdata_raw.sort().values[:-1]
                    )
                    mindelta = (diffs[diffs > 0]).min().item()

                    # we want to sample a set of frequencies that are spaced
                    # in the range covered by the gaussian mixture, but we
                    # want to sample them densely enough to resolve the
                    # narrowest gaussian so we want a minimum frequency

                    freq = torch.arange(
                        1 / (self._xdata_raw.max() - self._xdata_raw.min()).item(),
                        1 / (mindelta / 2),
                        step.item(),
                    )

            elif self.ndim == 2:
                raise NotImplementedError(
                    """Plotting PSDs in more than 1 dimension is
                                          not currently supported. Please get in touch
                                          if you need this functionality!
                """
                )
            else:
                raise NotImplementedError(
                    """Plotting PSDs in more than 2 dimensions
                                          is not currently supported. Please get in
                                          touch if you need this functionality!
                """
                )

        if mcmc_samples:
            msg = (
                "MCMC is not currently exposed. "
                "It will be available in future releases."
            )
            raise NotImplementedError(msg)
            fig, ax = self._plot_psd_mcmc(
                freq,
                means=means,
                scales=scales,
                weights=weights,
                show=show,
                raw=raw,
                log=log,
                truncate_psd=truncate_psd,
                logpsd=logpsd,
                **kwargs,
            )
            return fig, ax
        # Computing the psd for frequencies f
        psd = self.compute_psd(
            freq,
            means=means,
            scales=scales,
            weights=weights,
            raw=raw,
            log=logpsd,
            **kwargs,
        )

        if truncate_psd is True:
            if logpsd:
                freq = freq[psd > psd.max() - 34.5388]
                psd = psd[psd > psd.max() - 34.5388]
            else:
                freq = freq[psd > 1e-6 * psd.max()]
                psd = psd[psd > 1e-6 * psd.max()]
        elif truncate_psd:
            freq = freq[psd > truncate_psd]
            psd = psd[psd > truncate_psd]

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # plotting psd
        ax.plot(freq, psd)
        if log[0]:
            ax.set_xscale("log")
        if log[1] and not logpsd:  # we don't need to double-log the Y axis (I hope!)
            ax.set_yscale("log")
        if show:
            plt.show()
        else:
            return fig, ax

    def _plot_psd_mcmc(
        self,
        freq,
        means=None,
        scales=None,
        weights=None,
        show=True,
        raw=False,
        log=(True, True),
        truncate_psd=True,
        logpsd=False,
        n_samples_to_plot=25,
        **kwargs,
    ):
        """Plot the power spectral density of the model using MCMC samples

        Parameters
        ----------
        freq : array_like
            The frequencies at which to compute the PSD
        means : array_like, optional
            The means of the gaussians in the spectral mixture kernel, by
            default None. If None, the means from the model will be used.
        scales : array_like, optional
            The scales of the gaussians in the spectral mixture kernel, by
            default None. If None, the scales from the model will be used.
        weights : array_like, optional
            The weights of the gaussians in the spectral mixture kernel, by
            default None. If None, the weights from the model will be used.
        show : bool, optional
            Whether to show the plot, by default True.
        raw : bool, optional
            If True, the PSD will be computed in the space that the model was
            trained in, by default False. If False, the PSD will be computed
            in the original space of the data.
        log : tuple, optional
            A tuple of two booleans, indicating whether to plot the x-axis and
            y-axis on a log scale, respectively, by default (True, False).
        truncate_psd : float or bool, optional
            If not False, the PSD will be truncated at this value, by default
            True. This is useful for speeding up plotting when the frequency
            range is large. If logpsd is True, this value should be given in
            (natural) log space. If truncate_psd is True, the PSD will be
            truncated at 1e-6 times the maximum PSD for logpsd=False, and 1e-15
            of the maximum PSD (i.e. max(ln(psd)) - 34.5388) for logpsd=True.
        logpsd : bool, optional
            If True, the PSD will be plotted on a log scale, by default False.
            If True, truncate_psd must be given in (natural) log space.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
            The figure and axes objects of the plot.
        """

        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        msg = "MCMC is not currently exposed. It will be available in future releases."
        raise NotImplementedError(msg)
        n_samples = min(self.num_samples, n_samples_to_plot)
        if means is None:
            # this approach is slightly bugged - if more than one chain is used,
            # it will only draw samples from the first chain
            # will change this to generate random indices instead
            # at some point!
            # right now, this will end up having shape (1, chains, samples) (I thinkk)
            means = (
                torch.as_tensor(self.inference_data.posterior["raw_frequencies"].values)
                .squeeze()[:n_samples]
                .unsqueeze(0)
            )
            # print(means.shape)
            # print(freq.shape)
        if scales is None:
            scales = (
                torch.as_tensor(
                    self.inference_data.posterior["raw_frequency_scales"].values
                )
                .squeeze()[:n_samples]
                .unsqueeze(0)
            )  # .unsqueeze(-1)
        if weights is None:
            weights = (
                torch.as_tensor(
                    self.inference_data.posterior[
                        "covar_module.mixture_weights_prior"
                    ].values
                )
                .squeeze()[:n_samples]
                .unsqueeze(0)
            )  # .unsqueeze(-1)

        # computing the psd for all samples simultaneously is very expensive,
        # so we're just going to loop over them and plot them individually
        # this means we have to do things in a differnet order to the other
        # plotting routines

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for i in range(n_samples):
            # Computing the psd for frequencies f
            psd = self.compute_psd(
                freq,
                means=means[..., i],
                scales=scales[..., i],
                weights=weights[..., i],
                raw=raw,
                log=logpsd,
                **kwargs,
            )

            if truncate_psd is True:
                mask = psd > psd.max() - 34.5388 if logpsd else psd > 1e-6 * psd.max()
            elif truncate_psd:
                mask = psd > truncate_psd
            # now we can plot it:
            ax.plot(freq[mask], psd[mask], alpha=0.2, color="b")

        # final plot formatting
        if log[0]:
            ax.set_xscale("log")
        if log[1] and not logpsd:  # we don't need to double-log the Y axis (I hope!)
            ax.set_yscale("log")
        if show:
            plt.show()
        return fig, ax

    def compute_psd(
        self,
        freq,
        means=None,
        scales=None,
        weights=None,
        raw=False,
        log=False,
        debug=False,
        **kwargs,
    ):
        """Compute the power spectral density for the model

        Parameters
        ----------
        freq : array_like or tuple(array_likes)
            The Fourier duals at which to compute the PSD
            If array_like, assumes only one dual present.
            If tuple, duals are unpacked from it.
        means : array_like, optional
            The means of the gaussians in the spectral mixture kernel, by
            default None. If None, the means from the model will be used.
        scales : array_like, optional
            The scales of the gaussians in the spectral mixture kernel, by
            default None. If None, the scales from the model will be used.
        weights : array_like, optional
            The weights of the gaussians in the spectral mixture kernel, by
            default None. If None, the weights from the model will be used.
        raw : bool, optional
            If True, the PSD will be computed in the space that the model was
            trained in, by default False. If False, the PSD will be computed
            in the original space of the data.
        **kwargs : dict, optional
            Any other keyword arguments to be passed.

        Returns
        -------
        psd : array_like
            The power spectral density of the model at the frequencies given
            by freq.
        """
        self._raise_if_fit_failed("PSD evaluation")
        if means is None:
            means = self.model.sci_kernel.mixture_means
            # now apply the transform too!
            if self.xtransform is not None and not raw:
                # there's probably an easier way to do this than converting to
                # a period and back, but this will do for now
                means = 1 / self.xtransform.inverse(1 / means, shift=False).detach()
        if scales is None:
            scales = self.model.sci_kernel.mixture_scales
            # now apply the transform too!
            if self.xtransform is not None and not raw:
                scales = 1 / (
                    2
                    * np.pi
                    * self.xtransform.inverse(
                        1 / (2 * torch.pi * scales), shift=False
                    ).detach()
                )
        if weights is None:
            weights = self.model.sci_kernel.mixture_weights.detach()  # .numpy()

        from torch.distributions import Normal as torchnorm

        # Computing the psd for frequencies f
        if debug:
            print(freq.shape, means.shape, scales.shape, weights.shape)
        norm = torchnorm(means, scales)
        if debug:
            print(norm)
        if self.ndim > 1:
            if not isinstance(freq, tuple):
                raise ValueError(
                    "freq must be a tuple of array_likes for "
                    "multidimensional light curves!"
                )
            if len(freq) > 2:
                raise NotImplementedError(
                    "PSD for more than two duals not implemented yet"
                )
            if len(freq) != self.ndim:
                raise ValueError(
                    "freq must have the same number of duals as the number "
                    "of light curve dimensions!"
                )
            f1, f2 = freq
            norm1 = torchnorm(means[..., -2], scales[..., -2])
            norm2 = torchnorm(means[..., -1], scales[..., -1])
            psd = norm1.log_prob(f1).unsqueeze(-1) + norm2.log_prob(f2).unsqueeze(1)
            try:
                psd_tot = torch.logsumexp(
                    torch.log(weights.unsqueeze(-1).unsqueeze(-1)) + psd, dim=-3
                )
            except RuntimeError as e:
                # chunk it
                print(f"{e}. Chunking not implemented yet in compute_psd.")
        else:
            if len(freq.shape) > 1:
                raise ValueError(
                    "array-like freq must be one-dimensional for 1D light " "curves!"
                )
            f1 = torch.as_tensor(freq)
            # marginalise over Fourier dual variables
            psd1 = norm.log_prob(f1.unsqueeze(-1)).sum(dim=-1)
            # marginalise over Fourier dual variables
            psd2 = norm.log_prob(-f1.unsqueeze(-1)).sum(dim=-1)
            psd = (
                torch.log(torch.Tensor([0.5]))
                + psd1
                + torch.log(1.0 + torch.exp(psd2 - psd1))
            )
            try:
                psd_tot = torch.logsumexp(
                    torch.log(weights.unsqueeze(-1)) + psd, dim=-2
                )
            except RuntimeError:  # logsumexp tries to allocate a large array and
                # then do the summation so let's do it in a loop instead and see
                # if that avoids the problem
                psd_tot = torch.zeros_like(f1)
                for i in range(len(freq[0])):
                    psd_tot[i] = torch.logsumexp(
                        torch.log(weights) + psd[..., i], dim=-1
                    )
        if debug:
            print(psd_tot.shape)
        if not log:
            psd_tot = psd_tot.exp().cpu().detach().numpy()
        return psd_tot

    def plot(
        self,
        ylim=None,
        yscale="auto",
        show=True,
        mcmc_samples=False,
        n_pred=1000,
        annotate_provenance=False,
        provenance_location="lower left",
        **kwargs,
    ):
        """Plot the model and data

        Parameters
        ----------
        ylim : list, optional
            The y-limits of the plot, by default None. If None, the y-limits
            will be set automatically. For 2-D (multiwavelength) data the
            limits are determined independently for each wavelength.
        yscale : str, optional
            The y-axis scale to use. Can be ``'auto'`` (default), ``'linear'``
            or ``'log'``. When ``'auto'``, log scale is chosen for a given
            wavelength if all its flux values are positive and the ratio of
            maximum to minimum flux exceeds 100; otherwise linear scale is
            used. For 2-D data the scale is decided independently per
            wavelength. Note that when ``mcmc_samples`` is ``True``, this
            parameter is currently ignored and the y-axis scale is set by the
            MCMC plotting routine.
        show : bool, optional
            Whether to show the plot, by default True.
        mcmc_samples : bool, optional
            Whether to plot the samples from the MCMC run, by default False.
            This will only work if the MCMC sampler has been run.
        n_pred : int, optional
            Number of prediction points used to construct the fine time grid
            for plotting. Lower values reduce memory usage and speed up
            plotting, especially for 2D light curves. Default is 1000.
        annotate_provenance : bool, optional
            If ``True``, annotate GP-fit plots with model/runtime/timestamp from
            the most recent fit-history entry. Default is ``False``.
        provenance_location : {"lower left", "lower right", "upper left",
            "upper right"}, optional
            Axes-relative location for provenance annotations when
            ``annotate_provenance=True``. Default is ``"lower left"``.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig : matplotlib.pyplot.Figure or list of matplotlib.pyplot.Figure
            The figure object of the plot.  For 2-D (multiwavelength) data a
            list of figures is returned, one per wavelength.
        """
        self._raise_if_fit_failed("GP fit plot")
        _VALID_YSCALES = ("auto", "linear", "log")
        if yscale not in _VALID_YSCALES:
            raise ValueError(
                f"yscale must be one of {_VALID_YSCALES!r}, got {yscale!r}"
            )
        if isinstance(n_pred, bool) or not isinstance(n_pred, (int, np.integer)):
            raise ValueError(
                f"n_pred must be an integer, got {type(n_pred).__name__!r}"
            )
        n_pred = int(n_pred)
        if n_pred < 2:
            raise ValueError(f"n_pred must be >= 2, got {n_pred}")
        if self.ndim > 2:
            raise NotImplementedError(
                "Plotting models and data in more than 2 dimensions is not "
                "currently supported. Please get in touch if you need this "
                "functionality!"
            )
        if ylim is None and self.ndim == 1:
            # ylim = [-3, 3]
            y_min = float(self.ydata.min())
            y_max = float(self.ydata.max())
            y_range = y_max - y_min
            if y_range != 0.0:
                padding = 0.1 * abs(y_range)
            else:
                # If all y values are identical, pad based on their magnitude,
                # or fall back to a small absolute padding.
                base = abs(y_max) if y_max != 0.0 else 1.0
                padding = 0.1 * base
            ylim = [y_min - padding, y_max + padding]
        if mcmc_samples:
            msg = (
                "MCMC is not currently exposed. "
                "It will be available in future releases."
                )
            raise NotImplementedError(msg)
            if self.__FITTED_MCMC:
                return self._plot_mcmc(ylim=ylim, show=show, **kwargs)
            else:
                raise RuntimeError("You must first run the MCMC sampler")
        elif not self.__FITTED_MAP:
            return self._plot_data_only(ylim=ylim, yscale=yscale, show=show)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get into evaluation (predictive posterior) mode
            # self.model.eval()
            # self.likelihood.eval()

            self._eval()

            target_dtype = self._xdata_transformed.dtype
            target_device = self._xdata_transformed.device
            self.model = self.model.to(dtype=target_dtype, device=target_device)
            self.likelihood = self.likelihood.to(dtype=target_dtype, device=target_device)
            self.model.prediction_strategy = None

            # Importing raw x and y training data from xdata and
            # ydata functions
            if self.ndim == 1:
                x_raw = self.xdata
            elif self.ndim == 2:
                x_raw = self.xdata[:, 0]
            # y_raw = self.ydata

            # creating array of test points across the range of the data
            x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), n_pred, dtype=x_raw.dtype, device=x_raw.device)

            if self.ndim == 1:
                fig = self._plot_1d(
                    x_fine_raw,
                    ylim=ylim,
                    yscale=yscale,
                    show=show,
                    annotate_provenance=annotate_provenance,
                    provenance_location=provenance_location,
                    **kwargs,
                )
            else:
                fig = self._plot_2d(
                    x_fine_raw,
                    ylim=ylim,
                    yscale=yscale,
                    show=show,
                    annotate_provenance=annotate_provenance,
                    provenance_location=provenance_location,
                    **kwargs,
                )
        return fig

    def _plot_mcmc(self, ylim=None, show=False, n_samples_to_plot=25, **kwargs):
        """Plot the model and data, including samples from the MCMC run

        Parameters
        ----------
        ylim : list, optional
            The y-limits of the plot, by default None. If None, the y-limits
            will be set automatically.
        show : bool, optional
            Whether to show the plot, by default True.
        n_samples_to_plot : int, optional
            The number of samples to plot, by default 25.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure object of the plot.
        """
        # Get into evaluation (predictive posterior) mode
        msg = "MCMC is not currently exposed. It will be available in future releases."
        raise NotImplementedError(msg)
        self._eval()

        if self.ndim > 1:
            raise NotImplementedError(
                """
            Plotting models and data in more than 1 dimension is not
            currently supported. Please get in touch if you need this
            functionality!
            """
            )
        # Importing raw x and y training data from xdata and
        # ydata functions
        x_raw = self.xdata
        # y_raw = self.ydata

        # creating array of 10000 test points across the range of the data
        x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), 10000, dtype=x_raw.dtype, device=x_raw.device).unsqueeze(-1)

        # transforming the x_fine_raw data to the space that the GP was
        # trained in (so it can predict)
        if self.xtransform is None:
            self.x_fine_transformed = x_fine_raw
        elif isinstance(self.xtransform, Transformer):
            self.x_fine_transformed = self.xtransform.transform(
                x_fine_raw.to(self.xtransform.min.device)
            )

        self.expanded_test_x = self.x_fine_transformed.unsqueeze(0).repeat(
            self.num_samples, 1, 1
        )  # .unsqueeze(0)
        output = self.model(self.expanded_test_x)
        with torch.no_grad():
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            for i in range(min(n_samples_to_plot, self.num_samples)):
                # Plot predictive samples as colored lines
                ax.plot(
                    x_fine_raw.cpu().numpy(),
                    output[i].sample().cpu().numpy(),
                    "b",
                    alpha=0.2,
                )

            # Plot training data as black filled circles (on top of model predictions)
            ax.plot(self.xdata.cpu().numpy(), self.ydata.cpu().numpy(), "ko")

            ax.legend(["Observed Data", "Sample means"])
            if ylim is not None:
                ax.set_ylim(ylim)
            if show:
                plt.show()
        return f

    @staticmethod
    def _yscale_and_ylim(y_vals, yscale, ylim):
        """Resolve the y-axis scale and limits for a single band.

        Parameters
        ----------
        y_vals : array-like
            Flux values for the band (must support ``min()``/``max()``).
        yscale : str
            One of ``'auto'``, ``'linear'``, or ``'log'``.  Values outside
            this set are passed through to ``ax.set_yscale()`` unchanged;
            callers should validate beforehand (``plot()`` does this).
        ylim : list or None
            Caller-supplied y-axis limits.  ``None`` means auto-compute.

        Returns
        -------
        scale : str
            Either ``'linear'`` or ``'log'``.
        lim : list or None
            Two-element list ``[y_lo, y_hi]``, or ``None`` when the limits
            should be left to matplotlib (e.g. log scale with non-positive
            data, or an explicit ``ylim`` that is incompatible with log scale).
        """
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))

        # Resolve scale
        if yscale == "auto":
            scale = (
                "log" if y_min > 0 and y_max / y_min > 100 else "linear"
            )
        else:
            scale = yscale

        # Resolve limits
        if ylim is None:
            if scale == "log" and y_min > 0:
                log_min = np.log10(y_min)
                log_max = np.log10(y_max)
                log_range = log_max - log_min
                padding = 0.1 * abs(log_range) if log_range != 0.0 else 0.1
                lim = [10 ** (log_min - padding), 10 ** (log_max + padding)]
            elif scale != "log":
                y_range = y_max - y_min
                if y_range != 0.0:
                    padding = 0.1 * abs(y_range)
                else:
                    base = abs(y_max) if y_max != 0.0 else 1.0
                    padding = 0.1 * base
                lim = [y_min - padding, y_max + padding]
            else:
                # Log scale forced/selected but data contains non-positive
                # values: let matplotlib choose an appropriate range.
                lim = None
        else:
            # Caller-supplied limits: skip setting them when they are
            # incompatible with a log axis (non-positive lower bound).
            lim = None if scale == "log" and ylim[0] <= 0 else ylim

        return scale, lim

    def _plot_data_only(self, ylim=None, yscale="auto", show=False):
        """Plot only the data, without any GP predictions.

        Used when the GP has not yet been fitted.

        For 1-D data, a single figure is returned.  For 2-D (multiband) data,
        a separate figure is created for each wavelength (matching the layout
        of :meth:`_plot_2d` used after fitting), and a list of figures is
        returned.
        """
        if self.ndim == 2:
            unique_values_axis2 = torch.unique(self.xdata[:, 1])
            figs = []
            for val in unique_values_axis2:
                mask = self.xdata[:, 1] == val
                x_plot = self.xdata[mask, 0].cpu().numpy()
                y_data_for_val = self.ydata[mask]
                y_plot = y_data_for_val.cpu().numpy()

                fig = plt.figure()
                ax = fig.add_subplot(111)

                if hasattr(self, "yerr") and self.yerr is not None:
                    ax.errorbar(
                        x_plot,
                        y_plot,
                        yerr=self.yerr[mask].cpu().numpy(),
                        fmt="ko",
                        label="Observed",
                    )
                else:
                    ax.plot(x_plot, y_plot, "ko", label="Observed")

                ax.set_ylabel("y")
                ax.set_xlabel("x")
                ax.set_title(f"y vs x for {val}")

                current_yscale, current_ylim = self._yscale_and_ylim(
                    y_plot, yscale, ylim
                )
                ax.set_yscale(current_yscale)
                if current_ylim is not None:
                    ax.set_ylim(current_ylim)
                ax.legend()

                if show:
                    plt.show()
                figs.append(fig)
            return figs

        # 1-D case
        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        x_plot = self.xdata.cpu().numpy()
        y_plot = self.ydata.cpu().numpy()
        if hasattr(self, "yerr") and self.yerr is not None:
            ax.errorbar(
                x_plot, y_plot, yerr=self.yerr.cpu().numpy(), fmt="ko", label="Observed"
            )
        else:
            ax.plot(x_plot, y_plot, "ko", label="Observed")
        current_yscale, current_ylim = self._yscale_and_ylim(y_plot, yscale, ylim)
        ax.set_yscale(current_yscale)
        if current_ylim is not None:
            ax.set_ylim(current_ylim)
        ax.legend()
        if show:
            plt.show()
        return f

    def _plot_1d(
        self,
        x_fine_raw,
        ylim=None,
        yscale="auto",
        show=False,
        save=True,
        annotate_provenance=False,
        provenance_location="lower left",
        **kwargs,
    ):
        # transforming the x_fine_raw data to the space that the GP was
        # trained in (so it can predict)
        if self.xtransform is None:
            x_fine_transformed = x_fine_raw
        elif isinstance(self.xtransform, Transformer):
            x_fine_transformed = self.xtransform.transform(
                x_fine_raw.to(self.xtransform.min.device)
            )

        # Make predictions
        observed_pred = self.likelihood(self.model(x_fine_transformed))

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

        # Plot predictive GP mean as blue line
        ax.plot(
            x_fine_raw.cpu().numpy(),
            observed_pred.mean.cpu().numpy(),
            "b",
            label="Mean",
        )

        # Shade between the lower and upper confidence bounds
        ax.fill_between(
            x_fine_raw.cpu().numpy(),
            lower.cpu().numpy(),
            upper.cpu().numpy(),
            alpha=0.5,
            label="Confidence",
        )

        # Plot training data as black filled circles (on top of model predictions)
        if self.yerr is not None:
            ax.errorbar(
                self.xdata.cpu().numpy(),
                self.ydata.cpu().numpy(),
                yerr=self.yerr.cpu().numpy(),
                fmt="ko",
                label="Observed",
            )
        else:
            ax.plot(
                self.xdata.cpu().numpy(),
                self.ydata.cpu().numpy(),
                "ko",
                label="Observed",
            )

        # Determine y-axis scale and limits using the shared helper
        current_yscale, current_ylim = self._yscale_and_ylim(
            self.ydata.cpu().numpy(), yscale, ylim
        )
        ax.set_yscale(current_yscale)
        if current_ylim is not None:
            ax.set_ylim(current_ylim)
        if annotate_provenance:
            self._plot_fit_history_provenance(
                ax,
                provenance_location=provenance_location,
            )
        ax.legend()
        if save:
            plt.savefig(f"{self.name}_fit.png")
        if show:
            plt.show()
        return f

    def _plot_2d(
        self,
        x_fine_raw,
        ylim=None,
        yscale="auto",
        show=False,
        save=True,
        annotate_provenance=False,
        provenance_location="lower left",
        **kwargs,
    ):
        if self.xtransform is None:
            x_fine_transformed = x_fine_raw
        elif isinstance(self.xtransform, Transformer):
            x_fine_transformed = self.xtransform.transform(
                x_fine_raw.to(self.xtransform.min.device),
                apply_to=(0, 0),
            )
        unique_values_axis2 = torch.unique(self.xdata[:, 1])
        figs = []
        for val in unique_values_axis2:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            vals = torch.ones_like(x_fine_transformed) * val
            x_fine_tmp = torch.cat((x_fine_transformed[:, None], vals[:, None]), dim=1)

            observed_pred = self.likelihood(self.model(x_fine_tmp))
            ax.plot(x_fine_raw.cpu().numpy(), observed_pred.mean.cpu().numpy(),
                    "b", label = "Mean")

            lower, upper = observed_pred.confidence_region()
            ax.fill_between(
                x_fine_raw.cpu().numpy(),
                lower.cpu().numpy(),
                upper.cpu().numpy(),
                alpha=0.5,
                label = "Confidence"
            )

            # Plot training data as black filled circles (on top of model predictions)
            mask = self.xdata[:, 1] == val
            x_data_for_val = self.xdata[mask, 0]
            y_data_for_val = self.ydata[mask]
            if hasattr(self, "yerr") and self.yerr is not None:
                y_err_for_val = self.yerr[mask]
                ax.errorbar(
                    x_data_for_val,
                    y_data_for_val,
                    yerr=y_err_for_val,
                    fmt="ko",
                    label="Observed Data",
                )
            else:
                ax.plot(
                    x_data_for_val,
                    y_data_for_val,
                    "ko",
                    label="Observed Data",
                )
            ax.legend()

            ax.set_ylabel("y")
            ax.set_xlabel("x")
            ax.set_title(f"y vs x for {val}")

            # Set x-axis limits to the data range for this wavelength so that
            # the plot is centred on that wavelength's observations, even when
            # the fit is evaluated over the combined time grid of all bands.
            x_min = x_data_for_val.min().item()
            x_max = x_data_for_val.max().item()
            x_padding = 0.05 * (x_max - x_min) if x_max != x_min else 0.5
            ax.set_xlim(x_min - x_padding, x_max + x_padding)

            # Determine y-axis scale and limits for this wavelength
            # independently, using the shared helper.
            current_yscale, current_ylim = self._yscale_and_ylim(
                y_data_for_val.cpu().numpy(), yscale, ylim
            )
            ax.set_yscale(current_yscale)
            if current_ylim is not None:
                ax.set_ylim(current_ylim)
            if annotate_provenance:
                self._plot_fit_history_provenance(
                    ax,
                    provenance_location=provenance_location,
                )

            if save:
                plt.savefig(f"{self.name}_{val}_fit.png")

            if show:
                plt.show()
            figs.append(fig)
        return figs

    def _plot_nd(self):
        raise NotImplementedError(
            """
        Plotting models and data in more than 2 dimensions is not currently supported.
        Please get in touch if you need this functionality!
        """
        )

    def plot_results(self):
        for key, value in self.results.item():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            with contextlib.suppress(ValueError):
                ax.plot(value, "-")
            ax.set_ylabel(key)
            ax.set_xlabel("Iteration")

            if "means" in key:
                self.value_inversed = self.xtransform.inverse(value)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(torch.Tensor(self.value_reversed), "-")
                ax.set_ylabel(key)
                ax.set_xlabel("Iteration")
        plt.show()

    def to_table(self):
        """Create an astropy table with the results.

        Parameters
        ----------
        none

        Returns
        -------
        tab_results : astropy.table.Table
            Astropy table with the results.
        """
        from astropy.table import Table

        t = Table()
        t["x"] = [np.asarray(self.xdata.cpu())]
        t["y"] = [np.asarray(self.ydata.cpu())]
        if hasattr(self, "yerr"):
            t["yerr"] = [np.asarray(self.yerr.cpu())]
        if self.__FITTED_MCMC or self.__FITTED_MAP:
            # These outputs can only be produced if a fit has been run.
            periods, weights, scales = self.get_periods()
            t["period"] = [np.asarray(periods)]
            try:
                t["weights"] = [np.asarray(weights)]
            except RuntimeError:
                t["weights"] = [torch.as_tensor(weights).cpu().detach().numpy()]
            try:
                t["scales"] = [np.asarray(scales)]
            except RuntimeError:
                t["scales"] = [torch.as_tensor(scales).cpu().detach().numpy()]
            for key, value in self.results.items():
                try:
                    t[key] = [np.asarray(value)]
                except RuntimeError:
                    t[key] = [torch.as_tensor(value).cpu().detach().numpy()]
            if self.__FITTED_MAP:
                # Loss isn't relevant for MCMC, I think
                t["loss"] = [np.asarray(self.results["loss"])]
            # Now we want the model predictions for the input times:
            if self.__FITTED_MAP:
                self._eval()
                
                target_dtype = self._xdata_transformed.dtype
                target_device = self._xdata_transformed.device
                self.model = self.model.to(dtype=target_dtype, device=target_device)
                self.likelihood = self.likelihood.to(dtype=target_dtype, device=target_device)
                self.model.prediction_strategy = None
                
                with torch.no_grad():
                    observed_pred = self.likelihood(self.model(self._xdata_transformed))
                    t["y_pred_mean_obs"] = [np.asarray(observed_pred.mean.cpu())]
                    t["y_pred_lower_obs"] = [
                        np.asarray(observed_pred.confidence_region()[0].cpu())
                    ]
                    t["y_pred_upper_obs"] = [
                        np.asarray(observed_pred.confidence_region()[1].cpu())
                    ]

                    if self.ndim == 1:
                        x_raw = self.xdata
                    elif self.ndim == 2:
                        x_raw = self.xdata[:, 0]
                    # y_raw = self.ydata

                    # creating array of 10000 test points across the range of the data
                    x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), 10000, dtype=x_raw.dtype, device=x_raw.device)
                    if self.xtransform is None:
                        x_fine_transformed = x_fine_raw
                    elif isinstance(self.xtransform, Transformer):
                        x_fine_transformed = self.xtransform.transform(
                            x_fine_raw.to(self.xtransform.min.device)
                        )

                    # Make predictions
                    observed_pred = self.likelihood(self.model(x_fine_transformed))
                    t["x_fine"] = [np.asarray(x_fine_raw.cpu())]
                    t["y_pred_mean"] = [np.asarray(observed_pred.mean.cpu())]
                    t["y_pred_lower"] = [
                        np.asarray(observed_pred.confidence_region()[0].cpu())
                    ]
                    t["y_pred_upper"] = [
                        np.asarray(observed_pred.confidence_region()[1].cpu())
                    ]
            elif self.__FITTED_MCMC:
                raise NotImplementedError("MCMC predictions not yet implemented")
                # with torch.no_grad():

        return t

    def to_csv(self, filepath: str | Path = "lightcurve.csv") -> None:
        """Write lightcurve data to a CSV file.

        The output always includes ``time``, ``wavelength``, and ``flux``
        columns.  ``flux_error`` and ``band`` are included when present on the
        instance. For 1-D lightcurves (time-only ``xdata``), ``wavelength`` is
        written as ``0.0`` for each row.

        Parameters
        ----------
        filepath : str or pathlib.Path, optional
            Destination CSV path. The file is created or overwritten.
            Default is ``"lightcurve.csv"``.

        """
        def _tensor_to_numpy(tensor):
            return tensor.detach().cpu().numpy()

        path = Path(filepath)
        x_np = _tensor_to_numpy(self._xdata_raw)
        y_np = _tensor_to_numpy(self._ydata_raw)
        yerr_np = (
            _tensor_to_numpy(self._yerr_raw)
            if hasattr(self, "_yerr_raw") and self._yerr_raw is not None
            else None
        )

        if x_np.ndim == 1:
            time_np = x_np
            wavelength_np = np.zeros_like(time_np)
        elif x_np.ndim == 2 and x_np.shape[1] == 2:
            time_np = x_np[:, 0]
            wavelength_np = x_np[:, 1]
        else:
            raise ValueError(
                "xdata must be 1-dimensional (N,) or 2-dimensional "
                "with exactly two columns (N, 2) to export to CSV."
            )

        n_rows = len(time_np)
        if y_np.ndim != 1:
            raise ValueError("ydata must be 1-dimensional with shape (N,).")
        if len(y_np) != n_rows:
            raise ValueError(
                f"Length mismatch between xdata ({n_rows}) and ydata ({len(y_np)})."
            )
        if yerr_np is not None:
            if yerr_np.ndim != 1:
                raise ValueError("yerr must be 1-dimensional with shape (N,).")
            if len(yerr_np) != n_rows:
                raise ValueError(
                    "Length mismatch between xdata "
                    f"({n_rows}) and yerr ({len(yerr_np)})."
                )

        band_np = None
        if self.band is not None:
            band_np = np.asarray(self.band, dtype=str)
            if band_np.size == 1 and n_rows > 1:
                band_np = np.repeat(band_np, n_rows)
            elif band_np.size != n_rows:
                raise ValueError(
                    f"Length of 'band' ({band_np.size}) does not match "
                    f"number of rows ({n_rows})."
                )

        headers = ["time", "wavelength", "flux"]
        if yerr_np is not None:
            headers.append("flux_error")
        if band_np is not None:
            headers.append("band")

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for i in range(n_rows):
                row = [time_np[i], wavelength_np[i], y_np[i]]
                if yerr_np is not None:
                    row.append(yerr_np[i])
                if band_np is not None:
                    row.append(band_np[i])
                writer.writerow(row)

    def write_votable(self, filename):
        """Write the results to a votable file.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        """
        t = self.to_table()
        t.write(filename, format="votable", overwrite=True)

    # ------------------------------------------------------------------
    # Private helpers for merge / concat
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_lc_input(cls, item):
        """Resolve *item* to a :class:`Lightcurve`.

        Accepts a :class:`Lightcurve`, or a ``str``/``pathlib.Path``
        pointing to a CSV file.  Any other type raises :class:`TypeError`.

        Parameters
        ----------
        item : Lightcurve, str, or pathlib.Path
            The input to resolve.

        Returns
        -------
        Lightcurve

        Raises
        ------
        TypeError
            If *item* is not a :class:`Lightcurve`, ``str``, or
            ``pathlib.Path``.
        """
        if isinstance(item, cls):
            return item
        if isinstance(item, (str, Path)):
            return cls.from_csv(item)
        raise TypeError(
            f"Expected a Lightcurve, str, or pathlib.Path; "
            f"got {type(item).__name__!r}."
        )

    @staticmethod
    def _validate_band_wavelength_mapping(band_arr, wavelength_arr):
        """Check that band↔wavelength mapping is strictly one-to-one.

        Parameters
        ----------
        band_arr : numpy.ndarray of str
            Per-row band labels.
        wavelength_arr : numpy.ndarray of float
            Per-row wavelength values (from ``xdata[:, 1]``).

        Returns
        -------
        None
            Raises :class:`ValueError` if any mapping violation is found;
            returns without a value when all mappings are consistent.

        Raises
        ------
        ValueError
            If any band maps to more than one wavelength, or any wavelength
            maps to more than one band.
        """
        for b in np.unique(band_arr):
            wls = np.unique(wavelength_arr[band_arr == b])
            if len(wls) != 1:
                raise ValueError(
                    f"Band {b!r} maps to multiple wavelengths: {wls.tolist()}. "
                    "Each band must correspond to exactly one wavelength."
                )
        for wl in np.unique(wavelength_arr):
            bs = np.unique(band_arr[wavelength_arr == wl])
            if len(bs) != 1:
                raise ValueError(
                    f"Wavelength {wl} maps to multiple bands: "
                    f"{bs.tolist()}. "
                    "Each wavelength must correspond to exactly one band."
                )

    @staticmethod
    def _get_scalar_wavelength_for_1d(lc):
        """Extract a scalar wavelength from a 1-D :class:`Lightcurve`.

        Checks the attributes ``wavelength``, ``wave``, and ``lambda_`` in
        priority order.  The first one found is used.

        Parameters
        ----------
        lc : Lightcurve
            A 1-D lightcurve whose wavelength metadata should be read.

        Returns
        -------
        float
            The scalar wavelength value.

        Raises
        ------
        ValueError
            If none of the expected attributes exists, if the value is
            non-scalar, or if it cannot be converted to :class:`float`.
        """
        for attr in ("wavelength", "wave", "lambda_"):
            val = getattr(lc, attr, None)
            if val is None:
                continue
            # Reject non-scalar array-like values
            try:
                import numpy as _np

                arr = _np.asarray(val)
                if arr.ndim != 0 and arr.size != 1:
                    raise ValueError(
                        f"1-D input to concat(): attribute {attr!r} "
                        f"is non-scalar ({arr.shape}). "
                        "A single numeric wavelength value is required."
                    )
                return float(arr.flat[0])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"1-D input to concat(): attribute {attr!r} "
                    f"cannot be converted to a float: {val!r}."
                ) from exc
        raise ValueError(
            "1-D input to concat() cannot be promoted to 2-D: no wavelength "
            "metadata is available. Set lc.wavelength, lc.wave, or "
            "lc.lambda_ to a scalar numeric value before calling concat()."
        )

    # ------------------------------------------------------------------
    # merge
    # ------------------------------------------------------------------

    def merge(self, other, *, band=None, wavelength=None, on_conflict="raise"):
        """Merge *other* into this light curve, appending new band(s).

        Parameters
        ----------
        other : Lightcurve, str, or pathlib.Path
            The light curve (or CSV path) to merge.  Must not be a list;
            for multiple inputs use :meth:`concat`.
        band : str or array-like of str, optional
            Band label(s) to assign to *other* when *other* has no
            ``band`` attribute set.  For a 1-D *other* this must be a
            scalar string; for a 2-D *other* this must have the same
            length as *other*.
        wavelength : float, optional
            Wavelength to assign when *other* is 1-D.  Must not be
            provided for 2-D inputs.
        on_conflict : {"raise", "skip"}, optional
            Action taken when a band label or wavelength in *other*
            already exists in ``self``.  ``"raise"`` (default) raises a
            :class:`ValueError`; ``"skip"`` silently drops that
            constituent band and emits a :class:`UserWarning`.

        Returns
        -------
        Lightcurve
            A new 2-D :class:`Lightcurve` containing the rows of
            ``self`` followed by the non-conflicting rows from *other*.
            The result is always unfitted (no model or posterior state).

        Raises
        ------
        TypeError
            If *other* is a list (use :meth:`concat` instead), or if
            *other* is not a :class:`Lightcurve`, ``str``, or
            ``pathlib.Path``.
        ValueError
            If ``self`` is not 2-D.
        ValueError
            If a conflict is detected and *on_conflict* is ``"raise"``.
        ValueError
            If *wavelength* is provided for a 2-D *other*.
        ValueError
            If *other* is 1-D and *wavelength* is not provided.
        ValueError
            If *other* is 1-D but *band* does not resolve to exactly
            one unique label.
        """
        if isinstance(other, list):
            raise TypeError(
                "'other' must be a single Lightcurve or CSV path, not a list. "
                "To merge multiple inputs use Lightcurve.concat()."
            )

        if on_conflict not in ("raise", "skip"):
            raise ValueError(
                f"on_conflict must be 'raise' or 'skip'; got {on_conflict!r}."
            )

        if self.ndim < 2:
            raise ValueError(
                "merge() requires 'self' to be a 2-D lightcurve "
                "(xdata must have shape (N, 2))."
            )

        other = self._resolve_lc_input(other)

        # ------------------------------------------------------------------
        # 1-D vs 2-D promotion of *other*
        # ------------------------------------------------------------------
        if other.ndim < 2:
            if wavelength is None:
                raise ValueError(
                    "When 'other' is a 1-D lightcurve, 'wavelength' must be "
                    "provided as a scalar float."
                )
            if not np.isscalar(wavelength):
                raise ValueError(
                    "'wavelength' must be a scalar when 'other' is 1-D; "
                    f"got {type(wavelength).__name__!r}."
                )
            # Determine band label for the 1-D input
            if other.band is not None:
                if band is not None:
                    warnings.warn(
                        "'band' was supplied but 'other' already has a band "
                        "attribute; the supplied value will be ignored.",
                        UserWarning,
                        stacklevel=2,
                    )
                resolved_band = np.asarray(other.band).astype(str)
            else:
                if band is None:
                    raise ValueError(
                        "'band' must be supplied when 'other' is 1-D and has "
                        "no band attribute."
                    )
                resolved_band = np.atleast_1d(
                    np.asarray(band).astype(str)
                )

            unique_bands = np.unique(resolved_band)
            if len(unique_bands) != 1:
                raise ValueError(
                    "A 1-D 'other' must map to exactly one band label; "
                    f"got {unique_bands.tolist()}."
                )
            n_other = len(other._xdata_raw)
            if len(resolved_band) == 1:
                resolved_band = np.full(n_other, resolved_band[0], dtype=object)
            elif len(resolved_band) != n_other:
                raise ValueError(
                    f"Length of 'band' ({len(resolved_band)}) does not match "
                    f"the number of rows in 'other' ({n_other})."
                )

            # Promote to 2-D by stacking time + wavelength
            wl_col = torch.full(
                (n_other,),
                float(wavelength),
                dtype=other._xdata_raw.dtype,
                device=other._xdata_raw.device,
            )
            other_x = torch.stack([other._xdata_raw.reshape(-1), wl_col], dim=1)
            other_y = other._ydata_raw
            other_yerr = (
                other._yerr_raw if hasattr(other, "_yerr_raw") else None
            )
            other_band = resolved_band

        else:
            # other is already 2-D
            if wavelength is not None:
                raise ValueError(
                    "'wavelength' must not be provided when 'other' is "
                    "already a 2-D lightcurve."
                )
            other_x = other._xdata_raw
            other_y = other._ydata_raw
            other_yerr = (
                other._yerr_raw if hasattr(other, "_yerr_raw") else None
            )

            # Resolve band for 2-D other
            if other.band is not None:
                if band is not None:
                    warnings.warn(
                        "'band' was supplied but 'other' already has a band "
                        "attribute; the supplied value will be ignored.",
                        UserWarning,
                        stacklevel=2,
                    )
                other_band = np.asarray(other.band).astype(str)
            else:
                if band is None:
                    raise ValueError(
                        "'band' must be supplied when 'other' is 2-D and has "
                        "no band attribute."
                    )
                other_band = np.atleast_1d(np.asarray(band, dtype=object))
                if len(other_band) == 1:
                    other_band = np.full(
                        len(other_x), other_band[0], dtype=object
                    )
                elif len(other_band) != len(other_x):
                    raise ValueError(
                        f"Length of 'band' ({len(other_band)}) does not match "
                        f"the number of rows in 'other' ({len(other_x)})."
                    )

        # ------------------------------------------------------------------
        # Validate band arrays on self
        # ------------------------------------------------------------------
        if self.band is None:
            raise ValueError(
                "'self' must have a 'band' attribute set for merge()."
            )
        self_band = np.asarray(self.band).astype(str)
        self_x = self._xdata_raw

        # Validate band↔wavelength mapping on self
        self._validate_band_wavelength_mapping(
            self_band, self_x[:, 1].detach().cpu().numpy()
        )

        # Validate band↔wavelength mapping on other
        self._validate_band_wavelength_mapping(
            other_band, other_x[:, 1].detach().cpu().numpy()
        )

        # ------------------------------------------------------------------
        # Build per-band groups from self (for conflict checking)
        # ------------------------------------------------------------------
        self_bands_set = set(np.unique(self_band).tolist())
        self_wl_set = set(np.unique(self_x[:, 1].detach().cpu().numpy()).tolist())

        # ------------------------------------------------------------------
        # Process each constituent band in other
        # ------------------------------------------------------------------
        keep_x = [self_x]
        keep_y = [self._ydata_raw]
        keep_yerr = (
            [self._yerr_raw] if hasattr(self, "_yerr_raw") else None
        )
        keep_band = [self_band]

        other_constituent_bands = {}
        for b in np.unique(other_band):
            other_constituent_bands[b] = np.where(other_band == b)[0]

        for b, idx in other_constituent_bands.items():
            b_wl = float(other_x[idx[0], 1].item())
            conflict_reason = None
            if b in self_bands_set:
                conflict_reason = (
                    f"band {b!r} already exists in 'self'."
                )
            elif b_wl in self_wl_set:
                conflict_reason = (
                    f"wavelength {b_wl} already exists in 'self'."
                )

            if conflict_reason is not None:
                if on_conflict == "raise":
                    raise ValueError(
                        f"Conflict detected: {conflict_reason} "
                        "Use on_conflict='skip' to skip conflicting bands."
                    )
                else:
                    warnings.warn(
                        f"Skipping band {b!r} from 'other': {conflict_reason}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

            keep_x.append(other_x[idx])
            keep_y.append(other_y[idx])
            if keep_yerr is not None and other_yerr is not None:
                keep_yerr.append(other_yerr[idx])
            elif keep_yerr is not None:
                keep_yerr = None
            keep_band.append(other_band[idx])

            # Track newly added band/wavelength to avoid double-merging
            self_bands_set.add(b)
            self_wl_set.add(b_wl)

        # ------------------------------------------------------------------
        # Assemble output arrays
        # ------------------------------------------------------------------
        new_x = torch.cat(keep_x, dim=0)
        new_y = torch.cat(keep_y, dim=0)
        new_yerr = (
            torch.cat(keep_yerr, dim=0) if keep_yerr is not None else None
        )
        new_band = np.concatenate(keep_band, axis=0)

        return type(self)(
            new_x,
            new_y,
            yerr=new_yerr,
            band=new_band,
            xtransform=self.xtransform,
            ytransform=self.ytransform,
            name=self.name,
        )

    # ------------------------------------------------------------------
    # concat
    # ------------------------------------------------------------------

    @classmethod
    def concat(cls, items, on_conflict="raise", **kwargs):
        """Concatenate multiple light curves into one 2-D :class:`Lightcurve`.

        Parameters
        ----------
        items : iterable of Lightcurve, str, or pathlib.Path
            Input light curves (or CSV paths) to concatenate.  A bare
            ``str`` is treated as a single item, **not** as an iterable
            of characters.
        on_conflict : {"raise", "skip"}, optional
            Action taken when a band label or wavelength appears in more
            than one input.  ``"raise"`` (default) raises a
            :class:`ValueError`; ``"skip"`` drops the offending
            constituent and emits a :class:`UserWarning`.
        **kwargs
            Additional keyword arguments forwarded to the
            :class:`Lightcurve` constructor (e.g. ``xtransform``,
            ``ytransform``, ``name``).  They are **not** forwarded to
            :meth:`from_csv`.

        Returns
        -------
        Lightcurve
            A new 2-D :class:`Lightcurve` built from all non-conflicting
            constituent bands across *items*, in input order.  Always
            unfitted.

        Raises
        ------
        TypeError
            If any element of *items* is not a :class:`Lightcurve`,
            ``str``, or ``pathlib.Path``.
        ValueError
            If *items* is empty.
        ValueError
            If some inputs have band information but others do not.
        ValueError
            If a 1-D input maps to more than one band or wavelength.
        ValueError
            If a conflict is detected and *on_conflict* is ``"raise"``.
        """
        # Treat a bare Lightcurve, str, or Path as a single-element list
        if isinstance(items, (cls, str, Path)):
            items = [items]

        items = list(items)
        if not items:
            raise ValueError(
                "concat() requires at least one item; got an empty iterable."
            )

        if on_conflict not in ("raise", "skip"):
            raise ValueError(
                f"on_conflict must be 'raise' or 'skip'; got {on_conflict!r}."
            )
        # ------------------------------------------------------------------
        # Resolve all inputs to Lightcurve objects
        # ------------------------------------------------------------------
        lcs = [cls._resolve_lc_input(item) for item in items]

        # ------------------------------------------------------------------
        # Global band requirement: all inputs must carry band information.
        # ------------------------------------------------------------------
        has_band = [lc.band is not None for lc in lcs]
        if not any(has_band):
            raise ValueError(
                "concat() requires band information on all inputs; "
                "none of the supplied inputs has a 'band' attribute."
            )
        if any(has_band) and not all(has_band):
            raise ValueError(
                "All inputs must have band information if any one of them "
                "does. Found a mix of inputs with and without 'band'."
            )

        # ------------------------------------------------------------------
        # Promote each lightcurve to 2-D + resolve band arrays
        # ------------------------------------------------------------------
        resolved = []  # list of (x_2d, y, yerr, band_arr)
        for lc in lcs:
            if lc.ndim < 2:
                # 1-D input: validate band info, then recover scalar wavelength.
                band_arr = np.asarray(lc.band).astype(str)
                unique_b = np.unique(band_arr)
                if len(unique_b) != 1:
                    raise ValueError(
                        "A 1-D input to concat() must map to exactly one "
                        f"band label; got {unique_b.tolist()}."
                    )
                # Recover scalar wavelength from lc.wavelength / .wave / .lambda_
                wl_scalar = cls._get_scalar_wavelength_for_1d(lc)
                # Build 2D xdata: (N, 2) with col0=time, col1=wavelength
                t_col = lc._xdata_raw.reshape(-1)
                wl_col = torch.full(
                    (t_col.shape[0],),
                    wl_scalar,
                    dtype=t_col.dtype,
                    device=t_col.device,
                )
                x_2d = torch.stack([t_col, wl_col], dim=1)
                y = lc._ydata_raw
                yerr = lc._yerr_raw if hasattr(lc, "_yerr_raw") else None
                # Expand single-label band to one entry per data row
                n_rows = lc._xdata_raw.shape[0]
                band_arr = np.full(n_rows, unique_b[0], dtype=object)
                resolved.append((x_2d, y, yerr, band_arr))
            else:
                # 2-D
                x_2d = lc._xdata_raw
                y = lc._ydata_raw
                yerr = lc._yerr_raw if hasattr(lc, "_yerr_raw") else None

                band_arr = np.asarray(lc.band).astype(str)

                cls._validate_band_wavelength_mapping(
                    band_arr, x_2d[:, 1].detach().cpu().numpy()
                )
                resolved.append((x_2d, y, yerr, band_arr))
        # ------------------------------------------------------------------
        global_bands: set[str] = set()
        global_wls: set[float] = set()

        final_x_parts = []
        final_y_parts = []
        final_yerr_parts: list = []
        have_yerr = True  # set to False the moment any band lacks yerr
        final_band_parts = []

        for x_2d, y, yerr, band_arr in resolved:
            # Process each constituent band within this input
            for b in np.unique(band_arr):
                idx = np.where(band_arr == b)[0]
                b_wl = float(x_2d[idx[0], 1].item())

                conflict_reason = None
                if b in global_bands:
                    conflict_reason = (
                        f"band {b!r} appears in more than one input."
                    )
                elif b_wl in global_wls:
                    conflict_reason = (
                        f"wavelength {b_wl} appears in more than one input."
                    )

                if conflict_reason is not None:
                    if on_conflict == "raise":
                        raise ValueError(
                            f"Conflict detected: {conflict_reason} "
                            "Use on_conflict='skip' to skip conflicting "
                            "bands."
                        )
                    else:
                        warnings.warn(
                            f"Skipping band {b!r}: {conflict_reason}",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue

                final_x_parts.append(x_2d[idx])
                final_y_parts.append(y[idx])
                if have_yerr:
                    if yerr is not None:
                        final_yerr_parts.append(yerr[idx])
                    else:
                        # First band without yerr — drop all collected so far
                        have_yerr = False
                        final_yerr_parts = None
                final_band_parts.append(band_arr[idx])

                global_bands.add(b)
                global_wls.add(b_wl)

        if not final_x_parts:
            raise ValueError(
                "All constituent bands were skipped due to conflicts; "
                "the resulting lightcurve would be empty."
            )

        new_x = torch.cat(final_x_parts, dim=0)
        new_y = torch.cat(final_y_parts, dim=0)
        new_yerr = (
            torch.cat(final_yerr_parts, dim=0)
            if final_yerr_parts is not None
            else None
        )
        new_band = np.concatenate(final_band_parts, axis=0)

        return cls(new_x, new_y, yerr=new_yerr, band=new_band, **kwargs)
