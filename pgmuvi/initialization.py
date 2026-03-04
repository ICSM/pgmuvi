"""Smart initialization routines for GP kernel hyperparameters.

This module provides functions to automatically initialize kernel
hyperparameters from data using periodogram analysis and data statistics.
"""

import numpy as np
import torch


def initialize_quasi_periodic_from_data(train_x, train_y, yerr=None):
    """Initialize quasi-periodic kernel parameters from data via Lomb-Scargle.

    Computes the Lomb-Scargle periodogram on the input data and returns
    sensible initial hyperparameter values based on the dominant period.

    Parameters
    ----------
    train_x : torch.Tensor or array-like
        Time stamps (1D).
    train_y : torch.Tensor or array-like
        Observed values (1D).
    yerr : torch.Tensor or array-like, optional
        Measurement uncertainties. If provided, the data are weighted during
        the periodogram computation.

    Returns
    -------
    dict
        Dictionary with keys ``'period'``, ``'lengthscale'``, ``'decay'``,
        and ``'outputscale'``.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from pgmuvi.initialization import initialize_quasi_periodic_from_data
    >>> t = torch.linspace(0, 20, 100)
    >>> y = torch.sin(2 * np.pi * t / 5.0)
    >>> params = initialize_quasi_periodic_from_data(t, y)
    >>> abs(params['period'] - 5.0) < 1.0  # rough check
    True
    """
    try:
        from astropy.timeseries import LombScargle
    except ImportError:
        return _fallback_init(train_x, train_y)

    x_np = _to_numpy(train_x).ravel()
    y_np = _to_numpy(train_y).ravel()

    span = x_np.max() - x_np.min()
    diffs = np.diff(np.sort(x_np))
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size > 0:
        typical_spacing = np.median(positive_diffs)
    else:
        return _fallback_init(train_x, train_y)

    min_freq = 1.0 / span if span > 0 else 1e-3
    max_freq = 1.0 / (2.0 * typical_spacing) if typical_spacing > 0 else 10.0

    if max_freq <= min_freq:
        return _fallback_init(train_x, train_y)

    dy = _to_numpy(yerr).ravel() if yerr is not None else None

    try:
        ls = LombScargle(x_np, y_np, dy=dy)
        frequency, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
        )
    except Exception:
        return _fallback_init(train_x, train_y)

    if len(power) == 0 or power.max() < 0.01:
        return _fallback_init(train_x, train_y)

    peak_idx = np.argmax(power)
    period = float(1.0 / frequency[peak_idx])

    lengthscale = 0.5
    decay = period * 5.0
    outputscale = float(np.std(y_np)) if np.std(y_np) > 0 else 1.0

    return {
        "period": period,
        "lengthscale": lengthscale,
        "decay": decay,
        "outputscale": outputscale,
    }


def initialize_separable_from_data(train_x, train_y, yerr=None):
    """Initialize separable 2D kernel parameters from multiwavelength data.

    Uses per-band Lomb-Scargle periodograms to determine the dominant temporal
    period in each band and checks whether the variability is achromatic
    (consistent periods) or chromatic (varying periods).

    Parameters
    ----------
    train_x : torch.Tensor or array-like
        Input data of shape ``(n, 2)`` where column 0 is time and column 1 is
        wavelength/band.
    train_y : torch.Tensor or array-like
        Observed values (1D tensor of length n).
    yerr : torch.Tensor or array-like, optional
        Measurement uncertainties (1D).

    Returns
    -------
    dict
        Dictionary with keys:
        - ``'period'`` — mean period across bands.
        - ``'is_achromatic'`` — True if periods agree within 10 %.
        - ``'wavelength_lengthscale'`` — half the wavelength range.
        - ``'periods_per_band'`` — list of per-band peak periods.
        - ``'outputscale'`` — std of the observed data.

    Examples
    --------
    >>> import torch
    >>> from pgmuvi.initialization import initialize_separable_from_data
    >>> t = torch.linspace(0, 10, 40)
    >>> wl = torch.cat([torch.ones(20) * 500.0, torch.ones(20) * 700.0])
    >>> x = torch.stack([t, wl], dim=1)
    >>> y = torch.sin(2 * 3.14159 * t / 3.0)
    >>> params = initialize_separable_from_data(x, y)
    >>> 'period' in params
    True
    """
    try:
        from astropy.timeseries import LombScargle
    except ImportError:
        return _fallback_separable_init(train_x, train_y)

    x_np = _to_numpy(train_x)
    y_np = _to_numpy(train_y).ravel()
    dy_np = _to_numpy(yerr).ravel() if yerr is not None else None

    times = x_np[:, 0]
    wavelengths = x_np[:, 1]

    unique_wls = np.unique(wavelengths)
    periods_per_band = []

    wl_span = float(wavelengths.max() - wavelengths.min())
    wavelength_lengthscale = max(wl_span / 2.0, 1.0)

    span = times.max() - times.min()
    diffs = np.diff(np.sort(times))
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size > 0:
        typical_spacing = np.median(positive_diffs)
    else:
        typical_spacing = 1.0

    min_freq = 1.0 / span if span > 0 else 1e-3
    max_freq = 1.0 / (2.0 * typical_spacing) if typical_spacing > 0 else 10.0

    for wl in unique_wls:
        mask = wavelengths == wl
        t_band = times[mask]
        y_band = y_np[mask]
        dy_band = dy_np[mask] if dy_np is not None else None

        if len(t_band) < 5 or max_freq <= min_freq:
            continue

        try:
            ls = LombScargle(t_band, y_band, dy=dy_band)
            frequency, power = ls.autopower(
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
            )
            if len(power) > 0 and power.max() > 0.01:
                peak_idx = np.argmax(power)
                periods_per_band.append(float(1.0 / frequency[peak_idx]))
        except Exception:
            continue

    if len(periods_per_band) == 0:
        return _fallback_separable_init(train_x, train_y)

    period_mean = float(np.mean(periods_per_band))
    period_std = float(np.std(periods_per_band))
    is_achromatic = (period_std / period_mean) < 0.1 if period_mean > 0 else True

    return {
        "period": period_mean,
        "is_achromatic": is_achromatic,
        "wavelength_lengthscale": wavelength_lengthscale,
        "periods_per_band": periods_per_band,
        "outputscale": float(np.std(y_np)) if np.std(y_np) > 0 else 1.0,
    }


def initialize_from_physics(period, lengthscale=None, decay=None, outputscale=1.0):
    """Create initial hyperparameters from user-supplied physical parameters.

    Parameters
    ----------
    period : float
        Known or expected period of the source (same units as input time).
    lengthscale : float, optional
        Periodic lengthscale. Defaults to ``0.5`` (moderate periodicity).
    decay : float, optional
        Long-term decay timescale. Defaults to ``5 * period`` (slow decay).
    outputscale : float, optional
        Amplitude of the variability. Defaults to 1.0.

    Returns
    -------
    dict
        Dictionary suitable for passing to ``QuasiPeriodicGPModel`` or
        ``pgmuvi.lightcurve.Lightcurve.fit(guess=...)``.

    Examples
    --------
    >>> from pgmuvi.initialization import initialize_from_physics
    >>> params = initialize_from_physics(period=10.0, outputscale=0.5)
    >>> params['period']
    10.0
    """
    if lengthscale is None:
        lengthscale = 0.5
    if decay is None:
        decay = period * 5.0

    return {
        "period": float(period),
        "lengthscale": float(lengthscale),
        "decay": float(decay),
        "outputscale": float(outputscale),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(x):
    """Convert a tensor or array-like to a NumPy array."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _fallback_init(train_x, train_y):
    """Return default initialization when periodogram fails."""
    x_np = _to_numpy(train_x).ravel()
    y_np = _to_numpy(train_y).ravel()
    span = float(x_np.max() - x_np.min()) if len(x_np) > 1 else 1.0
    period = span / 2.0
    return {
        "period": period,
        "lengthscale": 0.5,
        "decay": period * 5.0,
        "outputscale": float(np.std(y_np)) if np.std(y_np) > 0 else 1.0,
    }


def _fallback_separable_init(train_x, train_y):
    """Return default separable initialization when per-band analysis fails."""
    x_np = _to_numpy(train_x)
    y_np = _to_numpy(train_y).ravel()
    times = x_np[:, 0] if x_np.ndim == 2 else x_np
    wavelengths = x_np[:, 1] if x_np.ndim == 2 else np.ones_like(x_np)

    span = float(times.max() - times.min()) if len(times) > 1 else 1.0
    wl_span = (
        float(wavelengths.max() - wavelengths.min()) if len(wavelengths) > 1 else 1.0
    )

    return {
        "period": span / 2.0,
        "is_achromatic": True,
        "wavelength_lengthscale": max(wl_span / 2.0, 1.0),
        "periods_per_band": [],
        "outputscale": float(np.std(y_np)) if np.std(y_np) > 0 else 1.0,
    }
