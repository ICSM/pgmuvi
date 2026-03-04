"""Kernel-specific constraint helpers for pgmuvi.

This module provides convenience functions for constructing
gpytorch constraints that are physically motivated and based on
properties of the observed data, such as sampling cadence and
observation span.
"""

import gpytorch  # noqa: F401
from gpytorch.constraints import Interval, GreaterThan, LessThan, Positive


def period_constraint(data_span, min_periods=2, max_factor=1.0):
    """Constraint on a period parameter based on the data time span.

    Restricts the period to be at least ``data_span / (10 * min_periods)``
    and at most ``max_factor * data_span``.

    Parameters
    ----------
    data_span : float
        Total duration of the observations (max time - min time).
    min_periods : int, optional
        Minimum number of periods that must fit within the data span,
        by default 2.
    max_factor : float, optional
        Maximum period as a multiple of the data span, by default 1.0.

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
    lower = data_span / (10.0 * min_periods)
    upper = data_span * max_factor
    return Interval(lower_bound=lower, upper_bound=upper)


def lengthscale_constraint(typical_spacing, data_span):
    """Constraint on a temporal lengthscale.

    Restricts the lengthscale to be between the typical sampling cadence
    and the full data span.

    Parameters
    ----------
    typical_spacing : float
        Median sampling interval (time between consecutive observations).
    data_span : float
        Total duration of the observations.

    Returns
    -------
    gpytorch.constraints.Interval
        Constraint suitable for registering on a lengthscale parameter.

    Examples
    --------
    >>> from pgmuvi.constraints import lengthscale_constraint
    >>> c = lengthscale_constraint(typical_spacing=1.0, data_span=100.0)
    >>> c.lower_bound < c.upper_bound
    True
    """
    lower = max(typical_spacing, 1e-4)
    upper = data_span * 2.0
    return Interval(lower_bound=lower, upper_bound=upper)


def wavelength_constraint(wl_span, min_fraction=0.01):
    """Constraint on a wavelength kernel lengthscale.

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
        Constraint suitable for a wavelength RBF lengthscale parameter.

    Examples
    --------
    >>> from pgmuvi.constraints import wavelength_constraint
    >>> c = wavelength_constraint(wl_span=500.0)
    >>> c.lower_bound < c.upper_bound
    True
    """
    lower = max(wl_span * min_fraction, 1e-3)
    upper = wl_span * 10.0
    return Interval(lower_bound=lower, upper_bound=upper)


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


__all__ = [
    "GreaterThan",
    "Interval",
    "LessThan",
    "Positive",
    "lengthscale_constraint",
    "outputscale_constraint",
    "period_constraint",
    "positive_constraint",
    "wavelength_constraint",
]
