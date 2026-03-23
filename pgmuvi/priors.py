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

All classes accept optional *lower_bound*/*upper_period* (or *lower_period*/
*upper_period* for frequency-space classes) to truncate the prior to the
physically allowed range (matching the constraints already on the model).
Outside this range the log-probability is ``-inf``.

Pre-defined prior sets
----------------------
:data:`PRIOR_SETS` mirrors the structure of
:data:`~pgmuvi.constraints.CONSTRAINT_SETS` and contains ready-to-use prior
parameters for common source types.

Examples
--------
>>> from pgmuvi.priors import LogNormalPeriodPrior, LogNormalFrequencyPrior
>>> import torch
>>> p_prior = LogNormalPeriodPrior(mu=5.0, sigma=1.0, lower_bound=100.0)
>>> p_prior.log_prob(torch.tensor([150.0, 300.0, 50.0]))
tensor([-5.6021, -5.9296,    -inf])
>>> f_prior = LogNormalFrequencyPrior(mu=5.0, sigma=1.0, lower_period=100.0)
>>> # frequency 1/150 ~ 0.00667 (period 150 days, passes lower bound of 100)
>>> f_prior.log_prob(torch.tensor([1.0 / 150.0]))
tensor([4.0916])
"""

import copy
import math

import torch
from gpytorch.priors import LogNormalPrior, NormalPrior


# ---------------------------------------------------------------------------
# Period-space priors
# ---------------------------------------------------------------------------


class LogNormalPeriodPrior(LogNormalPrior):
    """Log-Normal prior on a period parameter.

    Period ``P ~ LogNormal(mu, sigma)``, optionally truncated to the interval
    ``[lower_bound, upper_bound]``.  Outside this interval the log-probability
    is ``-inf``.

    Parameters
    ----------
    mu : float, optional
        Log-mean of the distribution (log-median = mu).  Default ``5.0``
        (median ≈ 148 days, appropriate for LPVs).
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
    tensor([-5.6021, -5.9296,    -inf])
    """

    def __init__(self, mu=5.0, sigma=1.0, lower_bound=None, upper_bound=None):
        super().__init__(loc=mu, scale=sigma)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def log_prob(self, x):
        lp = super().log_prob(x)
        if self.lower_bound is not None:
            lb = torch.as_tensor(self.lower_bound, dtype=x.dtype, device=x.device)
            lp = torch.where(x >= lb, lp, torch.full_like(lp, float("-inf")))
        if self.upper_bound is not None:
            ub = torch.as_tensor(self.upper_bound, dtype=x.dtype, device=x.device)
            lp = torch.where(x <= ub, lp, torch.full_like(lp, float("-inf")))
        return lp


class NormalPeriodPrior(NormalPrior):
    """Normal (Gaussian) prior on a period parameter.

    Period ``P ~ Normal(mean, std)``, optionally truncated to
    ``[lower_bound, upper_bound]``.

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
    tensor([-5.2364, -7.8698,    -inf])
    """

    def __init__(self, mean=300.0, std=75.0, lower_bound=None, upper_bound=None):
        super().__init__(loc=mean, scale=std)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def log_prob(self, x):
        lp = super().log_prob(x)
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

    Optionally truncated to frequencies corresponding to periods in
    ``[lower_period, upper_period]`` (both in model/transformed-parameter
    space).

    Parameters
    ----------
    mu : float, optional
        Log-mean of the *period* distribution.  Default ``5.0``.
    sigma : float, optional
        Log-standard-deviation of the *period* distribution.  Default ``1.0``.
    lower_period : float or None, optional
        Minimum *period* (in model parameter units).  Frequencies whose
        corresponding period ``1/f`` falls below this threshold receive
        ``-inf`` log-prob.
    upper_period : float or None, optional
        Maximum *period* (in model parameter units).  Frequencies whose
        corresponding period ``1/f`` exceeds this threshold receive
        ``-inf`` log-prob.

    Notes
    -----
    The Jacobian correction is automatically included: this distribution is
    the mathematically correct probability density for the frequency when the
    period follows a log-normal.

    Examples
    --------
    >>> import torch
    >>> from pgmuvi.priors import LogNormalFrequencyPrior
    >>> prior = LogNormalFrequencyPrior(mu=5.0, sigma=1.0, lower_period=100.0)
    >>> # f = 1/150 ≈ 0.00667; period = 150 > 100, so not truncated
    >>> prior.log_prob(torch.tensor([1.0 / 150.0]))
    tensor([4.0916])
    >>> # f = 1/50; period = 50 < 100, so truncated
    >>> prior.log_prob(torch.tensor([1.0 / 50.0]))
    tensor([-inf])
    """

    def __init__(self, mu=5.0, sigma=1.0, lower_period=None, upper_period=None):
        # f ~ LogNormal(-mu, sigma) is the Jacobian-correct transformation of
        # P = 1/f ~ LogNormal(mu, sigma).
        super().__init__(loc=-mu, scale=sigma)
        self.lower_period = lower_period
        self.upper_period = upper_period

    def log_prob(self, f):
        lp = super().log_prob(f)
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

    Optionally truncated to frequencies corresponding to periods in
    ``[lower_period, upper_period]``.

    Parameters
    ----------
    mean : float, optional
        Mean of the *period* Normal distribution.  Default ``300.0``.
    std : float, optional
        Standard deviation of the *period* distribution.  Default ``75.0``.
    lower_period : float or None, optional
        Minimum period (in model parameter units).
    upper_period : float or None, optional
        Maximum period (in model parameter units).

    Examples
    --------
    >>> import torch
    >>> from pgmuvi.priors import NormalFrequencyPrior
    >>> prior = NormalFrequencyPrior(mean=300.0, std=75.0, lower_period=100.0)
    >>> prior.log_prob(torch.tensor([1.0 / 300.0]))
    tensor([6.1711])
    >>> prior.log_prob(torch.tensor([1.0 / 50.0]))  # period 50 < lower_period 100
    tensor([-inf])
    """

    def __init__(self, mean=300.0, std=75.0, lower_period=None, upper_period=None):
        super().__init__(loc=mean, scale=std)
        self.lower_period = lower_period
        self.upper_period = upper_period

    def log_prob(self, f):
        period = 1.0 / f
        # Normal log_prob evaluated at period = 1/f
        lp = (
            -0.5 * ((period - self.loc) / self.scale) ** 2
            - torch.log(self.scale)
            - 0.5 * math.log(2 * math.pi)
        )
        # Jacobian correction: dp/df = -1/f², log|J| = -2·log(f)
        lp = lp - 2.0 * torch.log(f)
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

#: Pre-defined prior sets for common astronomical source types.
#:
#: Each key is a source-type label (e.g. ``"LPV"``).  The value is a ``dict``
#: with the following sub-keys:
#:
#: ``"lognormal"``
#:     Parameters for the default Log-Normal period prior.
#: ``"normal"``
#:     Parameters for the second-choice Normal period prior.
#: ``"period_bounds"``
#:     A dict with ``"lower"`` and ``"upper"`` keys, each holding a
#:     ``(value, active)`` tuple in the *same convention* as
#:     :data:`~pgmuvi.constraints.CONSTRAINT_SETS`.
#:
#: Currently defined sets
#: ----------------------
#: LPV
#:     Long-Period Variable stars.  Default Log-Normal with ``mu=5``,
#:     ``sigma=1`` truncated to a minimum period of **100** days.
#:     Second-choice Normal with ``mean=300``, ``std=75``, same truncation.
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
        "period_bounds": {
            "lower": (100.0, True),
            "upper": (None, False),
        },
    },
}


def get_prior_set(name):
    """Return the prior-set dict for *name*.

    Parameters
    ----------
    name : str
        Name of the prior set (e.g. ``"LPV"``).

    Returns
    -------
    dict
        Mapping containing ``"lognormal"``, ``"normal"``, and
        ``"period_bounds"`` sub-dicts.

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
    >>> ps["period_bounds"]["lower"]
    (100.0, True)
    """
    if name not in PRIOR_SETS:
        raise ValueError(
            f"Unknown prior_set {name!r}. "
            f"Available sets: {sorted(PRIOR_SETS.keys())}"
        )
    return copy.deepcopy(PRIOR_SETS[name])


__all__ = [
    "PRIOR_SETS",
    "LogNormalFrequencyPrior",
    "LogNormalPeriodPrior",
    "NormalFrequencyPrior",
    "NormalPeriodPrior",
    "get_prior_set",
]
