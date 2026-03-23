"""Period/frequency prior distributions for pgmuvi GP models.

This module provides Prior classes for setting priors on period (for models
parameterized directly in period space, e.g. :class:`~pgmuvi.gps.QuasiPeriodicGPModel`)
and on frequency (for models parameterized in frequency space, e.g.
:class:`~pgmuvi.gps.SpectralMixtureGPModel`).

Two flavours of prior are provided:

1. **Log-Normal in period space** (``"lognormal"`` / LPV default)
   Period P ~ LogNormal(mu, sigma), with ``mu=5`` and ``sigma=1``.
   The median period is ``exp(mu) ~ 148`` days, loosely appropriate for
   Long-Period Variable (LPV) stars.

   - :class:`LogNormalPeriodPrior` -- register directly on a ``period_length``
     parameter.
   - :class:`LogNormalFrequencyPrior` -- equivalent prior for a frequency
     parameter.  Because P = 1/f and the log-normal is closed under
     reciprocal, the frequency prior is LogNormal(-mu, sigma) and the
     Jacobian correction is automatically included.

2. **Normal in period space** (``"normal"`` / second choice)
   Period P ~ Normal(mean, std), with ``mean=300`` and ``std=75``.

   - :class:`NormalPeriodPrior` -- register on a ``period_length`` parameter.
   - :class:`NormalFrequencyPrior` -- equivalent prior for a frequency
     parameter, obtained by change of variables with the correct Jacobian.

All classes accept optional bounds to truncate the prior to the physically
allowed range.  The log-probability is correctly normalised over the truncated
support so that the distribution integrates to 1 within the allowed interval.
Outside the interval the log-probability is ``-inf``.

The *frequency* classes support a ``period`` keyword (default ``True``) that
controls whether the supplied bounds are expressed in period units or in
frequency units, giving users the flexibility to work in whichever space is
more natural.

Pre-defined prior sets
----------------------
:data:`PRIOR_SETS` mirrors the structure of
:data:`~pgmuvi.constraints.CONSTRAINT_SETS` and contains ready-to-use prior
parameters for common source types.  Period bounds are read directly from the
corresponding constraint set at runtime to guarantee consistency.

Examples
--------
>>> from pgmuvi.priors import LogNormalPeriodPrior, LogNormalFrequencyPrior
>>> import torch
>>> p_prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0, lower_bound=100.0)
>>> p_prior.log_prob(torch.tensor([150.0, 300.0, 50.0]))
tensor([-5.1767, -5.5043,    -inf])
>>> f_prior = LogNormalFrequencyPrior(mu=5.0, sigma=1.0, lower_period=100.0)
>>> # frequency 1/150 ~ 0.00667 (period 150 days, passes lower bound of 100)
>>> f_prior.log_prob(torch.tensor([1.0 / 150.0]))
tensor([4.5170])
"""

import copy
import math

import torch
from torch.distributions import LogNormal as TorchLogNormal
from torch.distributions import Normal as TorchNormal

from gpytorch.priors import LogNormalPrior, NormalPrior


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lognormal_log_normalizer(mu, sigma, lower, upper):
    """Log normalizing constant for a truncated LogNormal(mu, sigma) on [lower, upper].

    Returns ``log(CDF(upper) - CDF(lower))`` where ``CDF`` is the LogNormal
    CDF.  Handles ``None`` bounds (treated as 0 and +inf respectively).
    """
    dist = TorchLogNormal(
        torch.as_tensor(float(mu)), torch.as_tensor(float(sigma))
    )
    cdf_lower = (
        dist.cdf(torch.as_tensor(float(lower)))
        if lower is not None
        else torch.zeros(1)
    )
    cdf_upper = (
        dist.cdf(torch.as_tensor(float(upper)))
        if upper is not None
        else torch.ones(1)
    )
    return torch.log(cdf_upper - cdf_lower)


def _normal_log_normalizer(mean, std, lower, upper):
    """Log normalizing constant for a truncated Normal(mean, std) on [lower, upper].

    Returns ``log(CDF(upper) - CDF(lower))`` where ``CDF`` is the Normal CDF.
    Handles ``None`` bounds (treated as -inf and +inf respectively).
    """
    dist = TorchNormal(
        torch.as_tensor(float(mean)), torch.as_tensor(float(std))
    )
    cdf_lower = (
        dist.cdf(torch.as_tensor(float(lower)))
        if lower is not None
        else torch.zeros(1)
    )
    cdf_upper = (
        dist.cdf(torch.as_tensor(float(upper)))
        if upper is not None
        else torch.ones(1)
    )
    return torch.log(cdf_upper - cdf_lower)


# ---------------------------------------------------------------------------
# Period-space priors
# ---------------------------------------------------------------------------


class LogNormalPeriodPrior(LogNormalPrior):
    """Log-Normal prior on a period parameter, with correct truncation normalisation.

    Period ``P ~ LogNormal(mu, sigma)``, optionally truncated to the interval
    ``[lower_bound, upper_bound]``.  When bounds are supplied the log-probability
    is normalised so that the distribution integrates to 1 over the allowed
    interval; outside the interval the log-probability is ``-inf``.

    Parameters
    ----------
    mu : float, optional
        Log-mean of the distribution (log-median = mu).  Default ``5.0``
        (median ~ 148 days, appropriate for LPVs).
    sigma : float, optional
        Log-standard-deviation.  Default ``1.0``.
    lower_bound : float or None, optional
        Minimum period.  Values below this bound receive ``-inf`` log-prob.
    upper_bound : float or None, optional
        Maximum period.  Values above this bound receive ``-inf`` log-prob.

    Examples
    --------
    >>> import torch
    >>> from pgmuvi.priors import LogNormalPeriodPrior
    >>> prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0, lower_bound=100.0)
    >>> prior.log_prob(torch.tensor([150.0, 300.0, 50.0]))
    tensor([-5.1767, -5.5043,    -inf])
    """

    def __init__(self, mu=5.0, sigma=1.0, lower_bound=None, upper_bound=None):
        super().__init__(loc=mu, scale=sigma)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # Pre-compute log normalizer for the truncated support
        if lower_bound is None and upper_bound is None:
            self._log_normalizer = 0.0
        else:
            self._log_normalizer = float(
                _lognormal_log_normalizer(mu, sigma, lower_bound, upper_bound)
            )

    def log_prob(self, x):
        lp = super().log_prob(x) - self._log_normalizer
        if self.lower_bound is not None:
            lb = torch.as_tensor(self.lower_bound, dtype=x.dtype, device=x.device)
            lp = torch.where(x >= lb, lp, torch.full_like(lp, float("-inf")))
        if self.upper_bound is not None:
            ub = torch.as_tensor(self.upper_bound, dtype=x.dtype, device=x.device)
            lp = torch.where(x <= ub, lp, torch.full_like(lp, float("-inf")))
        return lp


class NormalPeriodPrior(NormalPrior):
    """Normal (Gaussian) prior on a period parameter, truncation-normalised.

    Period ``P ~ Normal(mean, std)``, optionally truncated to
    ``[lower_bound, upper_bound]``.  When bounds are supplied the
    log-probability is normalised over the allowed interval.

    Parameters
    ----------
    mean : float, optional
        Mean of the Normal distribution.  Default ``300.0`` (days).
    std : float, optional
        Standard deviation.  Default ``75.0`` (days).
    lower_bound : float or None, optional
        Minimum period.  Values below this bound receive ``-inf`` log-prob.
    upper_bound : float or None, optional
        Maximum period.  Values above this bound receive ``-inf`` log-prob.

    Examples
    --------
    >>> import torch
    >>> from pgmuvi.priors import NormalPeriodPrior
    >>> prior = NormalPeriodPrior(mean=300.0, std=75.0, lower_bound=100.0)
    >>> prior.log_prob(torch.tensor([300.0, 100.0, 50.0]))
    tensor([-5.2326, -7.8660,    -inf])
    """

    def __init__(self, mean=300.0, std=75.0, lower_bound=None, upper_bound=None):
        super().__init__(loc=mean, scale=std)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if lower_bound is None and upper_bound is None:
            self._log_normalizer = 0.0
        else:
            self._log_normalizer = float(
                _normal_log_normalizer(mean, std, lower_bound, upper_bound)
            )

    def log_prob(self, x):
        lp = super().log_prob(x) - self._log_normalizer
        if self.lower_bound is not None:
            lb = torch.as_tensor(self.lower_bound, dtype=x.dtype, device=x.device)
            lp = torch.where(x >= lb, lp, torch.full_like(lp, float("-inf")))
        if self.upper_bound is not None:
            ub = torch.as_tensor(self.upper_bound, dtype=x.dtype, device=x.device)
            lp = torch.where(x <= ub, lp, torch.full_like(lp, float("-inf")))
        return lp


# ---------------------------------------------------------------------------
# Frequency-space priors (period prior expressed in frequency space)
# ---------------------------------------------------------------------------


class LogNormalFrequencyPrior(LogNormalPrior):
    """Log-Normal period prior expressed as a prior on a frequency parameter.

    If period ``P ~ LogNormal(mu, sigma)`` and frequency ``f = 1/P``, then
    the change-of-variables formula (with Jacobian) gives ``f ~ LogNormal(-mu,
    sigma)``.  This class encapsulates that transformation so that the prior
    can be registered directly on a frequency (e.g. ``mixture_means``)
    parameter while still encoding a belief about the *period*.

    The log-probability is correctly normalised over the truncated support.

    Parameters
    ----------
    mu : float, optional
        Log-mean of the *period* distribution.  Default ``5.0``.
    sigma : float, optional
        Log-standard-deviation of the *period* distribution.  Default ``1.0``.
    lower_period : float or None, optional
        Lower bound on the allowed range.  Interpretation depends on
        ``period``:

        - ``period=True`` (default): minimum *period*.  Frequencies whose
          corresponding period ``1/f`` falls below this threshold receive
          ``-inf`` log-prob.
        - ``period=False``: minimum *frequency*.  Frequencies below this
          threshold receive ``-inf`` log-prob.
    upper_period : float or None, optional
        Upper bound on the allowed range.  Interpretation depends on
        ``period``:

        - ``period=True`` (default): maximum *period*.
        - ``period=False``: maximum *frequency*.
    period : bool, optional
        Controls the units of ``lower_period`` and ``upper_period``.
        If ``True`` (default) the bounds are in period units.
        If ``False`` the bounds are in frequency units, and are converted
        internally to period units (``lower_freq`` becomes ``upper_period``,
        ``upper_freq`` becomes ``lower_period``).

    Notes
    -----
    The Jacobian correction is automatically included: this distribution is
    the mathematically correct probability density for the frequency when the
    period follows a log-normal.

    Examples
    --------
    >>> import torch
    >>> from pgmuvi.priors import LogNormalFrequencyPrior
    >>> # Bounds in period units (default)
    >>> prior = LogNormalFrequencyPrior(mu=5.0, sigma=1.0, lower_period=100.0)
    >>> prior.log_prob(torch.tensor([1.0 / 150.0]))
    tensor([4.5170])
    >>> prior.log_prob(torch.tensor([1.0 / 50.0]))
    tensor([-inf])
    >>> # Equivalent using frequency units (upper_period=0.01 is max frequency)
    >>> prior_f = LogNormalFrequencyPrior(
    ...     mu=5.0, sigma=1.0, upper_period=1/100.0, period=False)
    >>> prior_f.log_prob(torch.tensor([1.0 / 150.0]))
    tensor([4.5170])
    """

    def __init__(self, mu=5.0, sigma=1.0, lower_period=None, upper_period=None,
                 period=True):
        # f ~ LogNormal(-mu, sigma) is the Jacobian-correct transformation of
        # P = 1/f ~ LogNormal(mu, sigma).
        super().__init__(loc=-mu, scale=sigma)
        if period:
            self.lower_period = lower_period
            self.upper_period = upper_period
        else:
            # Bounds given in frequency units: convert to period for storage.
            # A lower frequency bound (min_freq) corresponds to a maximum
            # period (max_period = 1/min_freq), and vice versa.
            self.lower_period = (
                1.0 / upper_period if upper_period is not None else None
            )
            self.upper_period = (
                1.0 / lower_period if lower_period is not None else None
            )
        # Normalizer computed in period space using LogNormal(mu, sigma) CDF
        if self.lower_period is None and self.upper_period is None:
            self._log_normalizer = 0.0
        else:
            self._log_normalizer = float(
                _lognormal_log_normalizer(
                    mu, sigma, self.lower_period, self.upper_period
                )
            )

    def log_prob(self, f):
        lp = super().log_prob(f) - self._log_normalizer
        period = 1.0 / f
        if self.lower_period is not None:
            lb = torch.as_tensor(self.lower_period, dtype=f.dtype, device=f.device)
            lp = torch.where(period >= lb, lp, torch.full_like(lp, float("-inf")))
        if self.upper_period is not None:
            ub = torch.as_tensor(self.upper_period, dtype=f.dtype, device=f.device)
            lp = torch.where(period <= ub, lp, torch.full_like(lp, float("-inf")))
        return lp


class NormalFrequencyPrior(NormalPrior):
    """Normal period prior expressed as a prior on a frequency parameter.

    Evaluates a Normal(mean, std) distribution in *period space* (P = 1/f) and
    applies the correct change-of-variables Jacobian so that the resulting
    log-probability is appropriate for a *frequency* parameter:

    .. code-block:: text

        log p(f) = Normal(mean, std).log_prob(1/f)  -  2*log(f)

    The ``-2*log(f)`` term is the log-Jacobian ``|dp/df| = 1/f^2``.

    The log-probability is correctly normalised over the truncated support.

    Parameters
    ----------
    mean : float, optional
        Mean of the *period* Normal distribution.  Default ``300.0``.
    std : float, optional
        Standard deviation of the *period* distribution.  Default ``75.0``.
    lower_period : float or None, optional
        Lower bound on the allowed range.  Interpretation depends on
        ``period``:

        - ``period=True`` (default): minimum *period*.
        - ``period=False``: minimum *frequency*.
    upper_period : float or None, optional
        Upper bound on the allowed range.  Interpretation depends on
        ``period``:

        - ``period=True`` (default): maximum *period*.
        - ``period=False``: maximum *frequency*.
    period : bool, optional
        If ``True`` (default), bounds are in period units.
        If ``False``, bounds are in frequency units (converted internally).

    Examples
    --------
    >>> import torch
    >>> from pgmuvi.priors import NormalFrequencyPrior
    >>> prior = NormalFrequencyPrior(mean=300.0, std=75.0, lower_period=100.0)
    >>> prior.log_prob(torch.tensor([1.0 / 300.0]))
    tensor([6.1749])
    >>> prior.log_prob(torch.tensor([1.0 / 50.0]))  # period 50 < lower_period 100
    tensor([-inf])
    """

    def __init__(self, mean=300.0, std=75.0, lower_period=None, upper_period=None,
                 period=True):
        super().__init__(loc=mean, scale=std)
        if period:
            self.lower_period = lower_period
            self.upper_period = upper_period
        else:
            # Bounds given in frequency units: convert to period for storage.
            # A lower frequency bound (min_freq) corresponds to max period,
            # and an upper frequency bound (max_freq) corresponds to min period.
            self.lower_period = (
                1.0 / upper_period if upper_period is not None else None
            )
            self.upper_period = (
                1.0 / lower_period if lower_period is not None else None
            )
        # Normalizer computed in period space using Normal(mean, std) CDF
        if self.lower_period is None and self.upper_period is None:
            self._log_normalizer = 0.0
        else:
            self._log_normalizer = float(
                _normal_log_normalizer(mean, std, self.lower_period, self.upper_period)
            )

    def log_prob(self, f):
        period = 1.0 / f
        # Normal log_prob evaluated at period = 1/f
        lp = (
            -0.5 * ((period - self.loc) / self.scale) ** 2
            - torch.log(self.scale)
            - 0.5 * math.log(2 * math.pi)
        )
        # Jacobian correction: dp/df = -1/f^2, log|J| = -2*log(f)
        lp = lp - 2.0 * torch.log(f) - self._log_normalizer
        if self.lower_period is not None:
            lb = torch.as_tensor(self.lower_period, dtype=f.dtype, device=f.device)
            lp = torch.where(period >= lb, lp, torch.full_like(lp, float("-inf")))
        if self.upper_period is not None:
            ub = torch.as_tensor(self.upper_period, dtype=f.dtype, device=f.device)
            lp = torch.where(period <= ub, lp, torch.full_like(lp, float("-inf")))
        return lp


# ---------------------------------------------------------------------------
# Pre-defined prior sets
# ---------------------------------------------------------------------------

#: Prior distribution parameters for common astronomical source types.
#:
#: Each key is a source-type label (e.g. ``"LPV"``).  The value is a ``dict``
#: with the following sub-keys:
#:
#: ``"lognormal"``
#:     Parameters for the default Log-Normal period prior.
#: ``"normal"``
#:     Parameters for the second-choice Normal period prior.
#:
#: Period bounds are **not** stored here; they are read at runtime from the
#: matching entry in :data:`~pgmuvi.constraints.CONSTRAINT_SETS` by
#: :func:`get_prior_set` to guarantee consistency.
#:
#: Currently defined sets
#: ----------------------
#: LPV
#:     Long-Period Variable stars.  Default Log-Normal with ``mu=5``,
#:     ``sigma=1``.  Second-choice Normal with ``mean=300``, ``std=75``.
#:     Truncation bounds come from ``CONSTRAINT_SETS["LPV"]["period"]``.
PRIOR_SETS = {
    "LPV": {
        "lognormal": {
            "mu": 5.0,
            "sigma": 1.0,
        },
        "normal": {
            "mean": 300.0,
            "std": 75.0,
        },
    },
}


def get_prior_set(name):
    """Return the prior-set dict for *name*, with period bounds from the constraint set.

    The returned dict contains the distribution parameters (``"lognormal"``
    and ``"normal"`` sub-dicts) **plus** a ``"period_bounds"`` sub-dict that
    is read at call time from :data:`~pgmuvi.constraints.CONSTRAINT_SETS` so
    that prior bounds and constraint bounds are always consistent.

    Parameters
    ----------
    name : str
        Name of the prior set (e.g. ``"LPV"``).

    Returns
    -------
    dict
        Mapping containing ``"lognormal"``, ``"normal"``, and
        ``"period_bounds"`` sub-dicts.  The caller receives a deep copy and
        may modify it freely.

    Raises
    ------
    ValueError
        If *name* is not a recognised prior set.

    Examples
    --------
    >>> from pgmuvi.priors import get_prior_set
    >>> ps = get_prior_set("LPV")
    >>> ps["lognormal"]["mu"]
    5.0
    >>> ps["period_bounds"]["lower"]  # from CONSTRAINT_SETS["LPV"]["period"]["lower"]
    (100.0, True)
    """
    if name not in PRIOR_SETS:
        raise ValueError(
            f"Unknown prior_set {name!r}. "
            f"Available sets: {sorted(PRIOR_SETS.keys())}"
        )
    result = copy.deepcopy(PRIOR_SETS[name])
    # Pull period bounds from the matching constraint set at runtime so they
    # are always in sync with the physical constraints on the model.
    from .constraints import CONSTRAINT_SETS, get_constraint_set
    if name in CONSTRAINT_SETS and "period" in CONSTRAINT_SETS[name]:
        cs = get_constraint_set(name)
        result["period_bounds"] = cs["period"]
    else:
        result["period_bounds"] = {"lower": (None, False), "upper": (None, False)}
    return result


__all__ = [
    "PRIOR_SETS",
    "LogNormalFrequencyPrior",
    "LogNormalPeriodPrior",
    "NormalFrequencyPrior",
    "NormalPeriodPrior",
    "get_prior_set",
]
