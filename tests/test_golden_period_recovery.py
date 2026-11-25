"""Seeded period-recovery regression tests (roadmap 1.4).

A noisy sinusoid with P = 37 d is fitted with each model and the period
reported by ``get_period_summary()`` (``dominant_period`` and
``period_interval``) must lie within 2 % of 37 d, with the interval covering
the truth. For the spectral-mixture models this is the peak of the summed
PSD, not a raw component mean.

Data: 80 irregular epochs over 370 d (10 cycles), amplitude 1, Gaussian
noise sigma = 0.1 (S/N 10 per point). For the 1D data the least-squares
sinusoid period is 37.03 d (brute-force grid in the scratch check that chose
these settings), so the data pin the period far more tightly than 2 %; a
miss is a fitting failure, not a data limit. The 2D data have the same
period at wavelengths 0.5, 1 and 2 um, 40 epochs each, amplitude rising
linearly with wavelength.

Initialisation follows what the code supports today:

* ``"1D"``: MLS initialisation (``use_mls_init=True``, the default).
* ``"1DQuasiPeriodic"`` / ``"1DPeriodicStochastic"``: ``fit`` applies MLS
  initialisation only to spectral-mixture models, so the period is seeded
  from ``fit_LS`` via ``guess``. Without it the period stays near its
  default (~170 d); ``TestUnseededQuasiPeriodic`` records that as strict
  xfails.
* ``"2D"``: ``use_best_band_init=True`` (the only route that seeds 2D
  mixture means) plus ``mixture_scales`` seeded at (1/T, 1/wavelength span).
  With the default scale (softplus(0) = 0.69 d^-1, a ~1 d coherence time)
  the temporal component drifts to periods of 200-300 d.
* ``"2DSeparable"``: the default time kernel is a Matern kernel with no
  period at all; the test asserts that. Period recovery is tested with a
  quasi-periodic time kernel passed through ``time_kernel``.

Learning rates: Adam at ``lr=0.1`` (the ``fit`` default) moves
``raw_mixture_means`` by about 10 % of the frequency per step, far wider than
the 1/T likelihood peak, and over 300 steps the 1D SM period wanders off the
peak for some seeds. The SM tests therefore use ``lr=0.02``.

Precision: ``Lightcurve`` stores data as float32 today and the synthetic
generators return float32, so float64 cannot be requested yet; revisit the
iteration counts when roadmap 1.8 lands.
"""

import contextlib
import io
import math

import gpytorch
import pytest
import torch

from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d

PERIOD = 37.0
T_SPAN = 10 * PERIOD
WAVELENGTHS = [0.5, 1.0, 2.0]
REL_TOL = 0.02

QP_PERIOD_KEY = {
    "1DQuasiPeriodic": "covar_module.base_kernel.kernels.0.period_length",
    "1DPeriodicStochastic": (
        "covar_module.kernels.0.base_kernel.kernels.0.period_length"
    ),
}


@pytest.fixture(autouse=True)
def single_thread():
    """Run torch single-threaded: these are small matrices, and several
    threads per pytest-xdist worker oversubscribe the CPU and make the fits
    several times slower."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _lc_1d():
    return make_simple_sinusoid_1d(
        n_obs=80,
        period=PERIOD,
        t_span=T_SPAN,
        noise_level=0.1,
        noise_type="gaussian",
        irregular=True,
        seed=0,
    )


def _lc_2d():
    return make_chromatic_sinusoid_2d(
        n_per_band=40,
        period=PERIOD,
        wavelengths=WAVELENGTHS,
        t_span=T_SPAN,
        noise_level=0.1,
        noise_type="gaussian",
        irregular=True,
        seed=0,
    )


def _ls_period(lc):
    with contextlib.redirect_stdout(io.StringIO()):
        freqs, _ = lc.fit_LS(num_peaks=1)
    return float(1.0 / freqs[0])


def _fit(lc, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        lc.fit(**kwargs)
    return lc.get_period_summary()


def _assert_recovered(summary):
    period = summary["dominant_period"]
    lo, hi = summary["period_interval"]
    assert period == pytest.approx(PERIOD, rel=REL_TOL), period
    assert lo <= PERIOD <= hi, (lo, hi)


def test_1d_spectral_mixture():
    lc = _lc_1d()
    summary = _fit(
        lc,
        model="1D",
        num_mixtures=1,
        use_mls_init=True,
        training_iter=200,
        lr=0.02,
    )
    _assert_recovered(summary)


@pytest.mark.parametrize("model", ["1DQuasiPeriodic", "1DPeriodicStochastic"])
def test_1d_quasi_periodic(model):
    lc = _lc_1d()
    guess = {QP_PERIOD_KEY[model]: torch.tensor(_ls_period(lc))}
    summary = _fit(lc, model=model, guess=guess, training_iter=100, lr=0.1)
    _assert_recovered(summary)


class TestUnseededQuasiPeriodic:
    """``fit`` does not seed ``period_length`` for these models, so with the
    defaults the period stays near its initial value and 37 d is not found.
    When MLS seeding is extended to them these tests XPASS and fail; then
    drop the ``guess`` in ``test_1d_quasi_periodic``."""

    @pytest.mark.xfail(
        strict=True, raises=AssertionError, reason="period_length is not MLS-seeded"
    )
    @pytest.mark.parametrize(
        "model", ["1DQuasiPeriodic", "1DPeriodicStochastic"]
    )
    def test_default_init(self, model):
        summary = _fit(_lc_1d(), model=model, training_iter=100, lr=0.1)
        _assert_recovered(summary)


def test_2d_spectral_mixture():
    lc = _lc_2d()
    wl_span = max(WAVELENGTHS) - min(WAVELENGTHS)
    guess = {
        "covar_module.mixture_scales": torch.tensor(
            [[[1.0 / T_SPAN, 1.0 / wl_span]]]
        )
    }
    summary = _fit(
        lc,
        model="2D",
        num_mixtures=1,
        use_mls_init=True,
        use_best_band_init=True,
        guess=guess,
        training_iter=200,
        lr=0.02,
    )
    _assert_recovered(summary)


def test_2d_separable_default_has_no_period():
    lc = _lc_2d()
    summary = _fit(lc, model="2DSeparable", training_iter=20, lr=0.1)
    assert summary["dominant_period"] is None
    assert summary["time_kernel_family"] == "MaternKernel"


def test_2d_separable_quasi_periodic_time_kernel():
    lc = _lc_2d()
    periodic = gpytorch.kernels.PeriodicKernel()
    periodic.period_length = _ls_period(lc)
    envelope = gpytorch.kernels.RBFKernel()
    envelope.lengthscale = T_SPAN
    time_kernel = gpytorch.kernels.ScaleKernel(periodic * envelope)
    summary = _fit(
        lc, model="2DSeparable", time_kernel=time_kernel, training_iter=100, lr=0.1
    )
    _assert_recovered(summary)
    assert math.isfinite(summary["q_factor"])
