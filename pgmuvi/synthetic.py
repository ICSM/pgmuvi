"""Synthetic light-curve generators for testing and demonstration.

This module provides a uniform interface for generating synthetic astronomical
light curves.  Four model families are provided:

(a) :func:`make_simple_sinusoid_1d`
    A 1-D light curve consisting of a single sinusoidal component plus
    optional Gaussian noise.

(b) :func:`make_multi_sinusoid_1d`
    A 1-D light curve with three (or more) sinusoidal components that each
    peak at a different phase.

(c) :func:`make_chromatic_sinusoid_2d`
    A 2-D (time x wavelength) light curve whose amplitude and phase both
    have a wavelength dependence.  Two amplitude laws are supported:

    * ``"linear"`` - amplitude scales linearly with wavelength.
    * ``"extinction"`` - amplitude follows a power-law dust-extinction law,
      ``A(wl) = overall_amplitude * exp(-tau * wl**alpha)``, mimicking the
      wavelength-dependent attenuation discussed elsewhere in the code.

(d) :func:`make_multi_sinusoid_chromatic_2d`
    A 2-D light curve with multiple sinusoidal components, each of which
    carries its own wavelength-dependent amplitude and phase.

All functions return a :class:`~pgmuvi.lightcurve.Lightcurve` object so that
they slot directly into the normal pgmuvi workflow.  The raw time, wavelength
and flux arrays can be recovered via ``lc.xdata`` and ``lc.ydata``.

Examples
--------
Generate a simple 1-D sinusoid::

    from pgmuvi.synthetic import make_simple_sinusoid_1d
    lc = make_simple_sinusoid_1d(period=5.0, noise_level=0.1, seed=42)

Generate a chromatic 2-D light curve with extinction-law amplitude::

    from pgmuvi.synthetic import make_chromatic_sinusoid_2d
    lc = make_chromatic_sinusoid_2d(
        period=400.0,
        wavelengths=[0.8, 1.2, 2.2],
        amplitude_law="extinction",
        tau=2.0,
        alpha=1.7,
        seed=0,
    )
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pgmuvi.lightcurve import Lightcurve


__all__ = [
    "make_chromatic_sinusoid_2d",
    "make_multi_sinusoid_1d",
    "make_multi_sinusoid_chromatic_2d",
    "make_simple_sinusoid_1d",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rng(seed: int | None) -> np.random.Generator:
    """Return a NumPy random Generator seeded with *seed* (or random if None)."""
    return np.random.default_rng(seed)


def _resolve_n_per_band(
    n_per_band: int | tuple[int, int] | list[int],
    n_bands: int,
    rng: np.random.Generator,
) -> list[int]:
    """Resolve the ``n_per_band`` argument to a per-band list of counts.

    Three forms are accepted:

    * **int** - every band gets the same number of observations.
    * **tuple (min, max)** - each band independently draws a random count
      from ``randint(min, max)`` (inclusive on both ends), so different
      bands may have different numbers of observations.
    * **list[int]** - explicit per-band counts; length must equal *n_bands*.

    Parameters
    ----------
    n_per_band:
        The user-supplied value.
    n_bands:
        Number of wavelength bands.
    rng:
        The random-number generator to use when *n_per_band* is a tuple.

    Returns
    -------
    list[int]
        A list of length *n_bands* with the resolved observation count for
        each band.

    Raises
    ------
    ValueError
        If *n_per_band* is a list whose length does not match *n_bands*, or
        if the tuple range is invalid (min > max or either value < 1).
    """
    if isinstance(n_per_band, int):
        return [n_per_band] * n_bands
    if isinstance(n_per_band, tuple):
        if len(n_per_band) != 2:  # type: ignore[arg-type]
            raise ValueError(
                "When n_per_band is a tuple it must have exactly two elements "
                f"(min, max), got {len(n_per_band)}."
            )
        lo, hi = int(n_per_band[0]), int(n_per_band[1])
        if lo < 1:
            raise ValueError(
                f"n_per_band minimum ({lo}) must be at least 1."
            )
        if hi < lo:
            raise ValueError(
                f"n_per_band range ({lo}, {hi}) is invalid: "
                "max must be >= min."
            )
        return [int(rng.integers(lo, hi + 1)) for _ in range(n_bands)]
    # list
    result = list(n_per_band)
    if len(result) != n_bands:
        raise ValueError(
            f"Length of n_per_band ({len(result)}) must match "
            f"length of wavelengths ({n_bands})."
        )
    return result


def _make_times(
    n: int,
    t_min: float,
    t_span: float,
    irregular: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a sorted array of *n* observation times."""
    if irregular:
        t = np.sort(rng.uniform(t_min, t_min + t_span, n))
    else:
        t = np.linspace(t_min, t_min + t_span, n)
    return t


def _extinction_amplitude(
    wavelengths: np.ndarray,
    overall_amplitude: float,
    tau: float,
    alpha: float,
    offset: float,
) -> np.ndarray:
    """Per-wavelength amplitude from a dust-extinction power law.

    ``A(wl) = overall_amplitude * exp(-tau * wl**alpha) + offset``

    Parameters
    ----------
    wavelengths:
        1-D array of wavelength values (same units as *tau* / *alpha*).
    overall_amplitude:
        Pre-extinction flux amplitude.
    tau:
        Dust optical depth scale.
    alpha:
        Extinction power-law index (typical ISM value: 1.7).
    offset:
        Additive background offset.
    """
    wl = np.asarray(wavelengths, dtype=float)
    extinction = tau * wl ** alpha
    return overall_amplitude * np.exp(-extinction) + offset


def _linear_amplitude(
    wavelengths: np.ndarray,
    base_amplitude: float,
    amplitude_slope: float,
    wl_ref: float,
) -> np.ndarray:
    """Per-wavelength amplitude from a linear law.

    ``A(λ) = base_amplitude * (1 + amplitude_slope * (λ - wl_ref))``
    """
    wl = np.asarray(wavelengths, dtype=float)
    return base_amplitude * (1.0 + amplitude_slope * (wl - wl_ref))


def _linear_phase(
    wavelengths: np.ndarray,
    phase_ref: float,
    phase_slope: float,
    wl_ref: float,
) -> np.ndarray:
    """Per-wavelength phase offset from a linear law.

    ``φ(λ) = phase_ref + phase_slope * (λ - wl_ref)``
    """
    wl = np.asarray(wavelengths, dtype=float)
    return phase_ref + phase_slope * (wl - wl_ref)


_VALID_NOISE_TYPES = ("gaussian", "poisson")


def _validate_noise_type(noise_type: str | None) -> None:
    """Raise ``ValueError`` if *noise_type* is not a recognised value."""
    if noise_type is not None and noise_type not in _VALID_NOISE_TYPES:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            f"Choose one of {list(_VALID_NOISE_TYPES)} or None."
        )


def _apply_noise(
    y_signal: np.ndarray,
    noise_level: float,
    noise_type: str | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply noise to *y_signal* and return the noisy array and uncertainties.

    Three noise modes are supported:

    * ``None`` - no noise is added and no uncertainties are produced.
      The signal is returned unchanged.
    * ``"gaussian"`` - constant-sigma Gaussian noise.
      ``noise ~ N(0, noise_level**2)`` at every data point.
    * ``"poisson"`` (default) - shot-noise approximation where the noise
      standard deviation scales as the square root of the local flux,
      mimicking Poisson statistics.  Brighter data points have higher
      *absolute* noise but lower *fractional* noise, consistent with the
      behaviour seen for real astronomical sources.  The signal is shifted
      so that it is always positive; the noise level at the *mean* flux
      equals *noise_level*.

    Parameters
    ----------
    y_signal:
        The noiseless signal array (1-D, float64).
    noise_level:
        Overall noise scale.  If ``<= 0``, the signal is returned
        unchanged and no uncertainties are produced.
    noise_type:
        ``None``, ``"gaussian"``, or ``"poisson"``.  Pass ``None`` to
        explicitly produce a noise-free output with no uncertainties.
    rng:
        The random-number generator to use.

    Returns
    -------
    y_noisy : np.ndarray
        Signal with noise added (a new array; *y_signal* is not modified).
        If *noise_level* is ``<= 0`` or *noise_type* is ``None`` this is
        a copy of *y_signal*.
    y_err : np.ndarray or None
        Per-point 1-sigma uncertainties corresponding to the applied noise.
        ``None`` when no noise is added.

    Raises
    ------
    ValueError
        If *noise_type* is not ``None``, ``"gaussian"``, or ``"poisson"``.
    """
    _validate_noise_type(noise_type)
    if noise_type is None or noise_level <= 0:
        return y_signal.copy(), None
    n = len(y_signal)
    if noise_type == "gaussian":
        y_err = np.full(n, noise_level)
        return y_signal + rng.standard_normal(n) * noise_level, y_err
    # noise_type == "poisson"
    # Shift to strictly positive: floor = 1 % of peak signal (or eps)
    peak = float(np.abs(y_signal).max())
    y_floor = peak * 0.01 + 1e-10
    y_pos = y_signal - float(y_signal.min()) + y_floor
    y_ref = float(y_pos.mean())
    noise_std = noise_level * np.sqrt(y_pos / y_ref)
    return y_signal + rng.standard_normal(n) * noise_std, noise_std


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_simple_sinusoid_1d(
    n_obs: int = 80,
    period: float = 150.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
    noise_level: float = 0.1,
    noise_type: str | None = "poisson",
    t_min: float = 0.0,
    t_span: float | None = None,
    irregular: bool = False,
    seed: int | None = None,
) -> Lightcurve:
    """Create a 1-D Lightcurve from a simple sinusoidal signal.

    The observed flux is::

        y(t) = A * sin(2*pi*t/P + phi) + noise

    Parameters
    ----------
    n_obs:
        Number of observations.
    period:
        Period of the sinusoid, in the same units as *t_span* (typically
        days). Defaults to ``150.0``.
    amplitude:
        Peak amplitude *A*.
    phase:
        Initial phase in radians.
    noise_level:
        Overall noise scale.  Interpretation depends on *noise_type*.
        Set to ``0`` for a noise-free light curve.
    noise_type:
        ``"poisson"`` (default) - shot-noise approximation where the noise
        standard deviation scales as the square root of the local flux
        (brighter = lower fractional noise).  ``"gaussian"`` - constant-sigma
        Gaussian noise with ``sigma = noise_level``.  ``None`` - no noise is
        added and ``yerr`` is not populated.
    t_min:
        Start time of the observation window.
    t_span:
        Total time span of the observations.  If ``None`` (default) the span
        is set to ``2.3 * period``.
    irregular:
        If ``True`` the observation times are drawn uniformly at random from
        ``[t_min, t_min + t_span]`` and then sorted.  If ``False`` (default)
        the times are equally spaced.
    seed:
        Optional integer seed for the random-number generator, for
        reproducibility.

    Returns
    -------
    Lightcurve
        A 1-D light curve object.

    Examples
    --------
    >>> lc = make_simple_sinusoid_1d(period=5.0, noise_level=0.1, seed=0)
    >>> lc.ndim
    1
    """
    from pgmuvi.lightcurve import Lightcurve

    if t_span is None:
        t_span = 2.3 * period
    rng = _rng(seed)
    t = _make_times(n_obs, t_min, t_span, irregular, rng)
    y = amplitude * np.sin(2 * math.pi * t / period + phase)
    y, y_err = _apply_noise(y, noise_level, noise_type, rng)

    t_tensor = torch.as_tensor(t, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    yerr_tensor = None if y_err is None else torch.as_tensor(y_err, dtype=torch.float32)
    return Lightcurve(t_tensor, y_tensor, yerr=yerr_tensor)


def make_multi_sinusoid_1d(
    n_obs: int = 80,
    components: list[dict] | None = None,
    noise_level: float = 0.1,
    noise_type: str | None = "poisson",
    t_min: float = 0.0,
    t_span: float | None = None,
    irregular: bool = False,
    seed: int | None = None,
) -> Lightcurve:
    """Create a 1-D Lightcurve with multiple sinusoidal components.

    Each component is characterised by its own period, amplitude, and phase,
    so the components peak at different times.  The default configuration uses
    three components spread across the phase circle, providing a signal that is
    clearly non-sinusoidal.

    Parameters
    ----------
    n_obs:
        Number of observations.
    components:
        List of component dictionaries.  Each dictionary must contain:

        ``"period"`` *(float)*
            Period in the same units as *t_span*.
        ``"amplitude"`` *(float)*
            Peak amplitude.
        ``"phase"`` *(float)*
            Initial phase in radians.

        If ``None`` the default three-component model is used::

            [
                {"period": 5.0, "amplitude": 1.0, "phase": 0.0},
                {"period": 3.0, "amplitude": 0.5, "phase": math.pi / 3},
                {"period": 7.0, "amplitude": 0.3, "phase": 2 * math.pi / 3},
            ]

    noise_level:
        Overall noise scale.  Interpretation depends on *noise_type*.
    noise_type:
        ``"poisson"`` (default) - shot-noise approximation (brighter = lower
        fractional noise).  ``"gaussian"`` - constant-sigma Gaussian noise.
        ``None`` - no noise is added and ``yerr`` is not populated.
    t_min:
        Start time.
    t_span:
        Total time span.  If ``None`` (default) the span is set to
        ``2.3 * max(period)`` across all components.
    irregular:
        If ``True`` observation times are randomly sampled.
    seed:
        Optional random seed.

    Returns
    -------
    Lightcurve
        A 1-D light curve object.

    Raises
    ------
    ValueError
        If any component dictionary is missing the required ``"period"``,
        ``"amplitude"``, or ``"phase"`` keys, or if *noise_type* is not a
        recognised value.

    Examples
    --------
    >>> lc = make_multi_sinusoid_1d(seed=0)
    >>> lc.ndim
    1
    """
    from pgmuvi.lightcurve import Lightcurve

    _validate_noise_type(noise_type)

    if components is None:
        components = [
            {"period": 5.0, "amplitude": 1.0, "phase": 0.0},
            {"period": 3.0, "amplitude": 0.5, "phase": math.pi / 3},
            {"period": 7.0, "amplitude": 0.3, "phase": 2 * math.pi / 3},
        ]

    for i, comp in enumerate(components):
        missing = [k for k in ("period", "amplitude", "phase") if k not in comp]
        if missing:
            raise ValueError(
                f"Component {i} is missing required key(s): {missing}. "
                "Each component dict must have 'period', 'amplitude', "
                "and 'phase' keys."
            )

    rng = _rng(seed)
    if t_span is None:
        t_span = 2.3 * max(c["period"] for c in components)
    t = _make_times(n_obs, t_min, t_span, irregular, rng)

    y = np.zeros(n_obs)
    for comp in components:
        y = y + comp["amplitude"] * np.sin(
            2 * math.pi * t / comp["period"] + comp["phase"]
        )

    y, y_err = _apply_noise(y, noise_level, noise_type, rng)

    t_tensor = torch.as_tensor(t, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    yerr_tensor = None if y_err is None else torch.as_tensor(y_err, dtype=torch.float32)
    return Lightcurve(t_tensor, y_tensor, yerr=yerr_tensor)


def make_chromatic_sinusoid_2d(
    n_per_band: int | tuple[int, int] | list[int] = 50,
    period: float = 150.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
    wavelengths: list[float] | None = None,
    amplitude_law: str = "linear",
    amplitude_slope: float = 0.3,
    wl_ref: float = 0.0,
    overall_amplitude: float = 5.0,
    tau: float = 2.0,
    alpha: float = 1.7,
    offset: float = 0.0,
    phase_law: str = "none",
    phase_slope: float = 0.1,
    noise_level: float = 0.1,
    noise_type: str | None = "poisson",
    t_min: float = 0.0,
    t_span: float | None = None,
    irregular: bool = True,
    seed: int | None = None,
) -> Lightcurve:
    """Create a 2-D Lightcurve with wavelength-dependent amplitude and phase.

    The light curve is observed at several wavelength bands; within each band
    the underlying signal is sinusoidal, but both its amplitude and phase vary
    with wavelength.

    Two amplitude laws are provided:

    * ``"linear"`` (default) -
      ``A(wl) = amplitude * (1 + amplitude_slope * (wl - wl_ref))``
    * ``"extinction"`` -
      ``A(wl) = overall_amplitude * exp(-tau * wl**alpha) + offset``
      (dust power-law extinction, matching the DustMean mean function).

    Two phase laws are provided:

    * ``"none"`` (default) - constant phase.
    * ``"linear"`` - ``phi(wl) = phase + phase_slope * (wl - wl_ref)``

    Parameters
    ----------
    n_per_band:
        Number of observations per wavelength band.  Three forms are accepted:

        * **int** - every band gets the same number of observations.
        * **tuple (min, max)** - each band independently draws a random
          count from ``randint(min, max)`` (inclusive), so different bands
          may have different numbers of observations.  Useful for simulating
          realistic uneven coverage across wavelengths.
        * **list[int]** - explicit count per band; length must equal the
          number of wavelengths.
    period:
        Period of the sinusoid.  Defaults to ``150.0`` days.
    amplitude:
        Base amplitude used by the ``"linear"`` law.
    phase:
        Reference phase (radians), used by both phase laws.
    wavelengths:
        List of wavelength values (in whatever units the model uses).  If
        ``None`` defaults to ``[450.0, 600.0, 750.0]`` (nm scale).
    amplitude_law:
        ``"linear"`` or ``"extinction"``.
    amplitude_slope:
        Slope of the linear amplitude law.
    wl_ref:
        Reference wavelength for the linear laws.
    overall_amplitude:
        Pre-extinction amplitude used by the extinction law.
    tau:
        Dust optical depth for the extinction law.
    alpha:
        Extinction power-law index (typical ISM value: 1.7).
    offset:
        Additive offset applied by the extinction law.
    phase_law:
        ``"none"`` or ``"linear"``.
    phase_slope:
        Slope of the linear phase law (radians per wavelength unit).
    noise_level:
        Overall noise scale.  Interpretation depends on *noise_type*.
    noise_type:
        ``"poisson"`` (default) - shot-noise approximation (brighter = lower
        fractional noise), applied independently per band.  ``"gaussian"`` -
        constant-sigma Gaussian noise.  ``None`` - no noise is added and
        ``yerr`` is not populated.
    t_min:
        Start time.
    t_span:
        Total time span.  If ``None`` (default) the span is set to
        ``2.3 * period``.
    irregular:
        If ``True`` observation times within each band are randomly sampled.
    seed:
        Optional random seed.

    Returns
    -------
    Lightcurve
        A 2-D (time x wavelength) light curve object.

    Examples
    --------
    Extinction-law amplitude, linear phase::

        lc = make_chromatic_sinusoid_2d(
            period=400.0,
            wavelengths=[0.8, 1.2, 2.2],
            amplitude_law="extinction",
            tau=2.0, alpha=1.7,
            phase_law="linear", phase_slope=0.2,
            seed=0,
        )
    """
    from pgmuvi.lightcurve import Lightcurve

    _validate_noise_type(noise_type)

    if wavelengths is None:
        wavelengths = [450.0, 600.0, 750.0]
    n_bands = len(wavelengths)

    if t_span is None:
        t_span = 2.3 * period

    rng = _rng(seed)
    n_per_band_list = _resolve_n_per_band(n_per_band, n_bands, rng)

    # Compute per-band amplitudes
    wl_arr = np.asarray(wavelengths, dtype=float)
    if amplitude_law == "linear":
        amplitudes = _linear_amplitude(wl_arr, amplitude, amplitude_slope, wl_ref)
    elif amplitude_law == "extinction":
        amplitudes = _extinction_amplitude(
            wl_arr, overall_amplitude, tau, alpha, offset
        )
    else:
        raise ValueError(
            f"Unknown amplitude_law '{amplitude_law}'. "
            "Choose 'linear' or 'extinction'."
        )

    # Compute per-band phases
    if phase_law == "none":
        phases = np.full(n_bands, phase)
    elif phase_law == "linear":
        phases = _linear_phase(wl_arr, phase, phase_slope, wl_ref)
    else:
        raise ValueError(
            f"Unknown phase_law '{phase_law}'. Choose 'none' or 'linear'."
        )

    t_list, wl_list, y_list, yerr_list = [], [], [], []
    for i, (wl, n, amp, ph) in enumerate(
        zip(wavelengths, n_per_band_list, amplitudes, phases, strict=False)
    ):
        t_band = _make_times(n, t_min, t_span, irregular, rng)
        y_band = amp * np.sin(2 * math.pi * t_band / period + ph)
        y_band, y_err_band = _apply_noise(y_band, noise_level, noise_type, rng)

        t_list.append(t_band)
        wl_list.append(np.full(n, wl))
        y_list.append(y_band)
        if y_err_band is not None:
            yerr_list.append(y_err_band)

    t_all = np.concatenate(t_list)
    wl_all = np.concatenate(wl_list)
    y_all = np.concatenate(y_list)

    x = torch.tensor(
        np.column_stack([t_all, wl_all]), dtype=torch.float32
    )
    y = torch.tensor(y_all, dtype=torch.float32)
    yerr = (
        torch.tensor(np.concatenate(yerr_list), dtype=torch.float32)
        if yerr_list
        else None
    )
    return Lightcurve(x, y, yerr=yerr)


def make_multi_sinusoid_chromatic_2d(
    n_per_band: int | tuple[int, int] | list[int] = 50,
    components: list[dict] | None = None,
    wavelengths: list[float] | None = None,
    amplitude_law: str = "extinction",
    amplitude_slope: float = 0.3,
    wl_ref: float = 0.0,
    overall_amplitude: float = 5.0,
    tau: float = 2.0,
    alpha: float = 1.7,
    offset: float = 0.0,
    phase_law: str = "linear",
    phase_slope: float = 0.1,
    noise_level: float = 0.1,
    noise_type: str | None = "poisson",
    t_min: float = 0.0,
    t_span: float | None = None,
    irregular: bool = True,
    seed: int | None = None,
) -> Lightcurve:
    """Create a 2-D Lightcurve with multiple chromatic sinusoidal components.

    Each sinusoidal component has its own wavelength-dependent amplitude and
    phase, following the laws described in
    :func:`make_chromatic_sinusoid_2d`.  Within each band the total signal is
    the sum of all components.

    Parameters
    ----------
    n_per_band:
        Number of observations per wavelength band.  Three forms are accepted:

        * **int** - every band gets the same number of observations.
        * **tuple (min, max)** - each band independently draws a random
          count from ``randint(min, max)`` (inclusive), producing realistic
          uneven coverage across wavelengths.
        * **list[int]** - explicit count per band; length must equal the
          number of wavelengths.
    components:
        List of component dictionaries.  Each dictionary may contain:

        ``"period"`` *(float)*
            Period in the same units as *t_span*.
        ``"amplitude_fraction"`` *(float)*
            Amplitude of this component relative to the band mean amplitude
            (computed from the chosen law).  Defaults to ``1.0``.
        ``"phase"`` *(float)*
            Reference phase in radians.  Defaults to ``0.0``.

        If ``None`` the default two-component model is used, mimicking the
        fundamental and first harmonic of a Mira-like LPV::

            [
                {"period": 400.0, "amplitude_fraction": 0.4, "phase": 0.0},
                {"period": 200.0, "amplitude_fraction": 0.1,
                 "phase": math.pi / 2},
            ]

        which mimics the fundamental and first harmonic of a pulsating star.

    wavelengths:
        List of wavelength values.  Defaults to ``[0.8, 1.2, 2.2]`` (µm).
    amplitude_law:
        ``"linear"`` or ``"extinction"``.  Defaults to ``"extinction"``.
    amplitude_slope:
        Slope of the linear amplitude law.
    wl_ref:
        Reference wavelength for the linear laws.
    overall_amplitude:
        Pre-extinction amplitude used by the extinction law.
    tau:
        Dust optical depth.
    alpha:
        Extinction power-law index.
    offset:
        Additive offset for the extinction law.
    phase_law:
        ``"none"`` or ``"linear"``.  Defaults to ``"linear"``.
    phase_slope:
        Slope of the linear phase law (radians per wavelength unit).
    noise_level:
        Overall noise scale.  Interpretation depends on *noise_type*.
    noise_type:
        ``"poisson"`` (default) - shot-noise approximation (brighter = lower
        fractional noise), applied independently per band.  ``"gaussian"`` -
        constant-sigma Gaussian noise.  ``None`` - no noise is added and
        ``yerr`` is not populated.
    t_min:
        Start time.
    t_span:
        Total time span.  If ``None`` (default) the span is set to
        ``2.3 * max(period)`` across all components.
    irregular:
        If ``True`` observation times within each band are randomly sampled.
    seed:
        Optional random seed.

    Returns
    -------
    Lightcurve
        A 2-D (time x wavelength) light curve object.

    Examples
    --------
    Mira-like pulsator with dust attenuation::

        lc = make_multi_sinusoid_chromatic_2d(
            n_per_band=60,
            t_span=3 * 400.0,
            components=[
                {"period": 400.0, "amplitude_fraction": 0.4, "phase": 0.0},
                {"period": 200.0, "amplitude_fraction": 0.1,
                 "phase": math.pi / 2},
            ],
            wavelengths=[0.8, 1.2, 2.2],
            amplitude_law="extinction",
            tau=2.0, alpha=1.7, overall_amplitude=5.0, offset=0.2,
            phase_law="linear", phase_slope=0.05,
            noise_level=0.05,
            seed=0,
        )
    """
    from pgmuvi.lightcurve import Lightcurve

    _validate_noise_type(noise_type)

    if components is None:
        components = [
            {"period": 400.0, "amplitude_fraction": 0.4, "phase": 0.0},
            {"period": 200.0, "amplitude_fraction": 0.1, "phase": math.pi / 2},
        ]

    if wavelengths is None:
        wavelengths = [0.8, 1.2, 2.2]
    n_bands = len(wavelengths)

    if t_span is None:
        if not components:
            raise ValueError(
                "When 't_span' is None, 'components' must be a non-empty sequence "
                "of dicts with a numeric 'period' key."
            )
        for idx, comp in enumerate(components):
            if "period" not in comp:
                raise ValueError(
                    "Each component must define a 'period' key when 't_span' is "
                    f"None (missing in component index {idx})."
                )
            period = comp["period"]
            if not isinstance(period, (int, float)) or not math.isfinite(period):
                raise ValueError(
                    "Each component 'period' must be a finite numeric value when "
                    f"'t_span' is None (invalid in component index {idx})."
                )
        t_span = 2.3 * max(c["period"] for c in components)

    rng = _rng(seed)
    n_per_band_list = _resolve_n_per_band(n_per_band, n_bands, rng)

    # Compute per-band mean amplitudes (used to scale each component)
    wl_arr = np.asarray(wavelengths, dtype=float)
    if amplitude_law == "linear":
        band_amplitudes = _linear_amplitude(
            wl_arr, overall_amplitude, amplitude_slope, wl_ref
        )
    elif amplitude_law == "extinction":
        band_amplitudes = _extinction_amplitude(
            wl_arr, overall_amplitude, tau, alpha, offset
        )
    else:
        raise ValueError(
            f"Unknown amplitude_law '{amplitude_law}'. "
            "Choose 'linear' or 'extinction'."
        )

    # Compute per-band phase offsets
    if phase_law == "none":
        band_phases = np.zeros(n_bands)
    elif phase_law == "linear":
        band_phases = _linear_phase(wl_arr, 0.0, phase_slope, wl_ref)
    else:
        raise ValueError(
            f"Unknown phase_law '{phase_law}'. Choose 'none' or 'linear'."
        )

    t_list, wl_list, y_list, yerr_list = [], [], [], []
    for wl, n, band_amp, band_ph in zip(
        wavelengths, n_per_band_list, band_amplitudes, band_phases, strict=False
    ):
        t_band = _make_times(n, t_min, t_span, irregular, rng)
        y_band = np.zeros(n)

        for comp in components:
            comp_period = comp["period"]
            comp_amp_frac = comp.get("amplitude_fraction", 1.0)
            comp_phase = comp.get("phase", 0.0)
            y_band = y_band + band_amp * comp_amp_frac * np.sin(
                2 * math.pi * t_band / comp_period + comp_phase + band_ph
            )

        y_band, y_err_band = _apply_noise(y_band, noise_level, noise_type, rng)

        t_list.append(t_band)
        wl_list.append(np.full(n, wl))
        y_list.append(y_band)
        if y_err_band is not None:
            yerr_list.append(y_err_band)

    t_all = np.concatenate(t_list)
    wl_all = np.concatenate(wl_list)
    y_all = np.concatenate(y_list)

    x = torch.tensor(
        np.column_stack([t_all, wl_all]), dtype=torch.float32
    )
    y = torch.tensor(y_all, dtype=torch.float32)
    yerr = (
        torch.tensor(np.concatenate(yerr_list), dtype=torch.float32)
        if yerr_list
        else None
    )
    return Lightcurve(x, y, yerr=yerr)
