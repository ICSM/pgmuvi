"""Kernel-specific constraint helpers for pgmuvi.

This module provides convenience functions for constructing
gpytorch constraints that are physically motivated and based on
properties of the observed data, such as sampling cadence and
observation span.
"""

import copy

import gpytorch  # noqa: F401
from gpytorch.constraints import Interval, GreaterThan, LessThan, Positive


def period_constraint(data_span, min_period_fraction=0.05, max_factor=1.0):
    """Constraint on a period parameter based on the data time span.

    Restricts the period to be at least ``min_period_fraction * data_span``
    and at most ``max_factor * data_span``.

    The default minimum-period fraction of 0.05 (5 % of the data span) is
    physically motivated by the typical observational requirements for cool
    evolved stars — the primary target class for pgmuvi — whose periods are
    generally tens of days or longer.  For a 500-day dataset this implies a
    lower bound of ~25 days; for a 1000-day dataset ~50 days.  Adjust
    ``min_period_fraction`` for other target classes or shorter baselines.

    Parameters
    ----------
    data_span : float
        Total duration of the observations (max time - min time), in whatever
        units the input time axis uses (days, normalised, etc.).
    min_period_fraction : float, optional
        Minimum period as a fraction of ``data_span``, by default 0.05.
    max_factor : float, optional
        Maximum period as a multiple of ``data_span``, by default 1.0.

    Returns
    -------
    gpytorch.constraints.Interval
        Constraint suitable for registering on a period parameter.

    Examples
    --------
    >>> from pgmuvi.constraints import period_constraint
    >>> c = period_constraint(data_span=100.0)
    >>> c.lower_bound < c.upper_bound
    True
    """
    lower = data_span * min_period_fraction
    upper = data_span * max_factor
    # Safety: ensure a valid (non-empty) interval even for very short baselines
    if lower >= upper:
        lower = upper * 0.01
    return Interval(lower_bound=lower, upper_bound=upper)


def lengthscale_constraint(span, min_fraction=0.01, max_fraction=2.0):
    """Constraint on a kernel lengthscale, expressed as fractions of a span.

    Works for both temporal and wavelength lengthscales — pass the relevant
    axis range (e.g. ``data_span`` for time, ``wl_span`` for wavelength) and
    the bounds are computed as fractions of that span.

    Parameters
    ----------
    span : float
        Total range of the axis being constrained (e.g. max_time - min_time,
        or max_wavelength - min_wavelength).
    min_fraction : float, optional
        Minimum lengthscale as a fraction of ``span``, by default 0.01.
        Must be less than ``max_fraction``.
    max_fraction : float, optional
        Maximum lengthscale as a multiple of ``span``, by default 2.0.

    Returns
    -------
    gpytorch.constraints.Interval
        Constraint suitable for registering on a lengthscale parameter.

    Raises
    ------
    ValueError
        If ``min_fraction >= max_fraction``, or if ``span <= 0``.

    Examples
    --------
    Temporal lengthscale with a 100-day dataset:

    >>> from pgmuvi.constraints import lengthscale_constraint
    >>> c = lengthscale_constraint(span=100.0)
    >>> c.lower_bound < c.upper_bound
    True

    Wavelength lengthscale with a 500 nm range:

    >>> c = lengthscale_constraint(span=500.0, min_fraction=0.01, max_fraction=10.0)
    >>> c.lower_bound < c.upper_bound
    True
    """
    if span <= 0:
        raise ValueError(f"span must be positive, got {span}")
    if min_fraction >= max_fraction:
        raise ValueError(
            f"min_fraction ({min_fraction}) must be less than "
            f"max_fraction ({max_fraction})"
        )
    lower = max(span * min_fraction, 1e-4)
    upper = span * max_fraction
    return Interval(lower_bound=lower, upper_bound=upper)


def wavelength_constraint(wl_span, min_fraction=0.01):
    """Constraint on a wavelength kernel lengthscale.

    Alias for :func:`lengthscale_constraint` with ``max_fraction=10.0``,
    kept for backward compatibility.

    Parameters
    ----------
    wl_span : float
        Total wavelength range (max wavelength - min wavelength).
    min_fraction : float, optional
        Minimum lengthscale as a fraction of the wavelength span, by default
        0.01.

    Returns
    -------
    gpytorch.constraints.Interval
        Constraint suitable for a wavelength kernel lengthscale parameter.

    Examples
    --------
    >>> from pgmuvi.constraints import wavelength_constraint
    >>> c = wavelength_constraint(wl_span=500.0)
    >>> c.lower_bound < c.upper_bound
    True

    See Also
    --------
    lengthscale_constraint : General-purpose lengthscale constraint.
    """
    return lengthscale_constraint(wl_span, min_fraction=min_fraction, max_fraction=10.0)


def positive_constraint():
    """Return a simple positivity constraint.

    Returns
    -------
    gpytorch.constraints.Positive
    """
    return Positive()


def outputscale_constraint(data_std, min_factor=0.001, max_factor=100.0):
    """Constraint on an output scale (amplitude) parameter.

    Parameters
    ----------
    data_std : float
        Standard deviation of the observed data.
    min_factor : float, optional
        Minimum output scale as a multiple of ``data_std``, by default 0.001.
    max_factor : float, optional
        Maximum output scale as a multiple of ``data_std``, by default 100.0.

    Returns
    -------
    gpytorch.constraints.Interval

    Examples
    --------
    >>> from pgmuvi.constraints import outputscale_constraint
    >>> c = outputscale_constraint(data_std=1.0)
    >>> c.lower_bound < c.upper_bound
    True
    """
    lower = max(data_std * min_factor, 1e-6)
    upper = data_std * max_factor
    return Interval(lower_bound=lower, upper_bound=upper)


# Pre-defined constraint sets for common source types.
#
# Each key is a source-type label (e.g. ``"LPV"``).  The value is a
# ``dict`` keyed by parameter name.  For the ``"period"`` parameter the
# entry is another ``dict`` with ``"lower"`` and ``"upper"`` keys, each
# holding a ``(value, active)`` tuple where *value* is the bound in the
# same units as the raw time axis / ``Lightcurve.xdata`` (typically days,
# or ``None`` when the limit is not applicable) and *active* is a
# ``bool`` that flags whether the limit should be enforced.
#
# Currently defined sets
# ----------------------
# LPV
#     Long-Period Variable stars.  Only a lower period limit of **20**
#     (in the same units as the time axis, typically days) is enforced
#     (``lower=(20.0, True)``).  The upper limit is inactive
#     (``upper=(None, False)``).
CONSTRAINT_SETS = {
    "LPV": {
        "period": {
            "lower": (20.0, True),
            "upper": (None, False),
        },
    },
}


def get_constraint_set(name):
    """Return the constraint-set dict for *name*.

    Parameters
    ----------
    name : str
        Name of the constraint set (e.g. ``"LPV"``).

    Returns
    -------
    dict
        Mapping of parameter names to their bound specifications.

    Raises
    ------
    ValueError
        If *name* is not a recognised constraint set.

    Examples
    --------
    >>> from pgmuvi.constraints import get_constraint_set
    >>> cs = get_constraint_set("LPV")
    >>> cs["period"]["lower"]
    (20.0, True)
    >>> cs["period"]["upper"]
    (None, False)
    """
    if name not in CONSTRAINT_SETS:
        raise ValueError(
            f"Unknown constraint_set {name!r}. "
            f"Available sets: {sorted(CONSTRAINT_SETS.keys())}"
        )
    return copy.deepcopy(CONSTRAINT_SETS[name])


__all__ = [
    "CONSTRAINT_SETS",
    "GreaterThan",
    "Interval",
    "LessThan",
    "Positive",
    "get_constraint_set",
    "lengthscale_constraint",
    "outputscale_constraint",
    "period_constraint",
    "positive_constraint",
    "wavelength_constraint",
]
