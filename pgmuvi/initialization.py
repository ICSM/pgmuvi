"""Smart initialization routines for GP kernel hyperparameters.

This module provides functions to automatically initialize kernel
hyperparameters from data using periodogram analysis and data statistics.
"""

import numpy as np
import torch

# Fraction of the relevant span (data span or period) used as the default
# PeriodicKernel lengthscale.  10% keeps the shape parameter resolved by
# typical observing cadences without being so large that the prior becomes
# uninformative.
_DEFAULT_LENGTHSCALE_FRACTION = 0.1


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

    # Initialise the PeriodicKernel lengthscale as a fraction of the data
    # span rather than a fixed value in data units.  Using 10% of the span
    # means the kernel shape is resolved by the observations while remaining
    # well below the total baseline.
    lengthscale = span * _DEFAULT_LENGTHSCALE_FRACTION
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

    Uses :class:`~pgmuvi.multiband_ls_significance.MultibandLSWithSignificance`
    to compute a single multiband Lomb-Scargle periodogram over all bands
    simultaneously and to assess the significance of the dominant period.
    Per-band single-band periodograms are then used to check whether the
    variability is achromatic (consistent periods across bands) or chromatic.

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

        - ``'period'`` — dominant multiband period (or per-band mean if
          the multiband approach fails).
        - ``'is_significant'`` — True if the peak FAP < 0.05.
        - ``'is_achromatic'`` — True if per-band periods agree within 10 %.
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
        from .multiband_ls_significance import MultibandLSWithSignificance
    except ImportError:
        return _fallback_separable_init(train_x, train_y)

    x_np = _to_numpy(train_x)
    y_np = _to_numpy(train_y).ravel()
    dy_np = _to_numpy(yerr).ravel() if yerr is not None else None

    times = x_np[:, 0]
    wavelengths = x_np[:, 1]

    unique_wls = np.unique(wavelengths)

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

    if max_freq <= min_freq:
        return _fallback_separable_init(train_x, train_y)

    outputscale = float(np.std(y_np)) if np.std(y_np) > 0 else 1.0

    # --- Step 1: multiband periodogram for the dominant period ---
    period_multiband = None
    is_significant = False
    try:
        ls_mb = MultibandLSWithSignificance(
            times, y_np, wavelengths, dy=dy_np
        )
        freq_grid = ls_mb.autofrequency(
            minimum_frequency=min_freq, maximum_frequency=max_freq
        )
        power = ls_mb.power(freq_grid)
        if len(power) > 0:
            peak_idx = int(np.argmax(power))
            period_multiband = float(1.0 / freq_grid[peak_idx])
            fap = ls_mb.false_alarm_probability(
                float(power[peak_idx]), method="analytical"
            )
            is_significant = bool(fap < 0.05)
    except Exception:
        period_multiband = None

    # --- Step 2: per-band periodograms for achromatic vs chromatic ---
    periods_per_band = []
    for wl in unique_wls:
        mask = wavelengths == wl
        t_band = times[mask]
        y_band = y_np[mask]
        dy_band = dy_np[mask] if dy_np is not None else None

        if len(t_band) < 5:
            continue

        try:
            ls = LombScargle(t_band, y_band, dy=dy_band)
            frequency, power_band = ls.autopower(
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
            )
            if len(power_band) > 0 and power_band.max() > 0.01:
                peak_idx = np.argmax(power_band)
                periods_per_band.append(float(1.0 / frequency[peak_idx]))
        except Exception:
            continue

    # Decide period: prefer multiband estimate; fall back to per-band mean
    if period_multiband is not None:
        period_out = period_multiband
    elif len(periods_per_band) > 0:
        period_out = float(np.mean(periods_per_band))
    else:
        return _fallback_separable_init(train_x, train_y)

    # Achromatic check: per-band periods should agree within 10 %
    if len(periods_per_band) >= 2:
        period_std = float(np.std(periods_per_band))
        period_mean = float(np.mean(periods_per_band))
        is_achromatic = (period_std / period_mean) < 0.1 if period_mean > 0 else True
    else:
        is_achromatic = True

    return {
        "period": period_out,
        "is_significant": is_significant,
        "is_achromatic": is_achromatic,
        "wavelength_lengthscale": wavelength_lengthscale,
        "periods_per_band": periods_per_band,
        "outputscale": outputscale,
    }


def initialize_from_physics(period, lengthscale=None, decay=None, outputscale=1.0):
    """Create initial hyperparameters from user-supplied physical parameters.

    Parameters
    ----------
    period : float
        Known or expected period of the source (same units as input time).
    lengthscale : float, optional
        Periodic lengthscale (PeriodicKernel). Defaults to ``0.1 * period``
        (10% of the period), which gives a moderately smooth periodic shape
        that is resolved by typical observing cadences.
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
        lengthscale = period * _DEFAULT_LENGTHSCALE_FRACTION
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
        "is_significant": False,
        "is_achromatic": True,
        "wavelength_lengthscale": max(wl_span / 2.0, 1.0),
        "periods_per_band": [],
        "outputscale": float(np.std(y_np)) if np.std(y_np) > 0 else 1.0,
    }
