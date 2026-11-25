"""Fixed-hyperparameter golden snapshots (roadmap 1.4).

No training happens here. Each test builds a seeded synthetic light curve,
sets the model, fixes every hyperparameter with ``set_hypers`` and compares a
derived quantity with a JSON file in ``tests/golden/``:

* the exact marginal log-likelihood of the training data,
* the log power spectral density from ``compute_psd`` on a fixed grid,
* ``get_period_summary().as_dict()`` for the 1D spectral-mixture model.

Before the snapshot is taken, the MLL and the 1D PSD are also checked against
an independent float64 NumPy computation of the same formula. The 2D PSD is
not: pgmuvi's 1D branch returns log(1/2 [S(f) + S(-f)]), but its 2D branch
returns log S(f1, f2) with no mirrored component and no factor 1/2, so the
2D snapshot pins today's behaviour and is not known to be the correct density.
A strict xfail asserts the symmetrised formula (roadmap 1.12).

Precision: ``Lightcurve`` currently stores its data as float32
(``lightcurve.py`` coerces inputs; roadmap 1.8 flips the default to float64)
and the synthetic generators return float32 tensors, so there is no float64
switch to request yet. The snapshots are float32 values compared with
``rtol=1e-4``; regenerate them (``PGMUVI_UPDATE_GOLDEN=1``) when 1.8 lands.

Hyperparameters (raw data units, no transforms are active, days and um):

* 1D SM, one component: mean frequency 1/37 d^-1, scale 0.005 d^-1,
  weight 1, constant mean 0.
* 2D SM, one component: mean frequencies (1/37 d^-1, 0.2 um^-1), scales
  (0.005 d^-1, 0.5 um^-1), weight 1, constant mean 0.

The noise is the fixed per-point ``yerr**2`` (= 0.01) of the
``FixedNoiseGaussianLikelihood`` the light curves get by default.
"""

import math
import warnings

import gpytorch
import numpy as np
import pytest
import torch

from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d

PERIOD = 37.0
T_SPAN = 10 * PERIOD
WAVELENGTHS = [0.5, 1.0, 2.0]
RTOL = 1e-4  # float32 quantities

HYPERS_1D = {
    "mean_module.constant": torch.tensor(0.0),
    "covar_module.mixture_weights": torch.tensor([1.0]),
    "covar_module.mixture_means": torch.tensor([[[1.0 / PERIOD]]]),
    "covar_module.mixture_scales": torch.tensor([[[0.005]]]),
}
HYPERS_2D = {
    "mean_module.constant": torch.tensor(0.0),
    "covar_module.mixture_weights": torch.tensor([1.0]),
    "covar_module.mixture_means": torch.tensor([[[1.0 / PERIOD, 0.2]]]),
    "covar_module.mixture_scales": torch.tensor([[[0.005, 0.5]]]),
}

# 64 log-spaced temporal frequencies from 1e-3 to 0.5 d^-1 (Nyquist of a
# 1 d cadence); 8 log-spaced wavelength frequencies for the 2D model.
FREQ_T = np.logspace(-3, math.log10(0.5), 64)
FREQ_WL = np.logspace(-2, 0, 8)


@pytest.fixture
def lc_1d():
    lc = make_simple_sinusoid_1d(
        n_obs=80,
        period=PERIOD,
        t_span=T_SPAN,
        noise_level=0.1,
        noise_type="gaussian",
        irregular=True,
        seed=0,
    )
    lc.set_model("1D", num_mixtures=1)
    lc.set_hypers(dict(HYPERS_1D))
    return lc


@pytest.fixture
def lc_2d():
    lc = make_chromatic_sinusoid_2d(
        n_per_band=40,
        period=PERIOD,
        wavelengths=WAVELENGTHS,
        t_span=T_SPAN,
        noise_level=0.1,
        noise_type="gaussian",
        irregular=True,
        seed=0,
    )
    lc.set_model("2D", num_mixtures=1)
    lc.set_hypers(dict(HYPERS_2D))
    return lc


def _exact_mll(lc):
    """MLL as the trainer computes it (trainers.py:114-116): train mode.

    In train mode ``model(train_x)`` is the GP prior at the training inputs,
    which is what ``ExactMarginalLogLikelihood`` expects; in eval mode it
    would be the posterior and the number would not be the evidence.
    ``ExactMarginalLogLikelihood`` divides by the number of data points.
    """
    lc.model.train()
    lc.likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lc.likelihood, lc.model)
    x = lc.model.train_inputs[0]
    y = lc.model.train_targets
    with torch.no_grad():
        return float(mll(lc.model(x), y))


def _sm_kernel_np(tau, weights, means, scales):
    """Spectral-mixture covariance for lags ``tau`` of shape (..., d), float64.

    k(tau) = sum_q w_q prod_d exp(-2 pi^2 tau_d^2 s_qd^2) cos(2 pi mu_qd tau_d)
    """
    k = 0.0
    for w, mu, s in zip(weights, means, scales, strict=True):
        k = k + w * np.prod(
            np.exp(-2 * np.pi**2 * tau**2 * s**2) * np.cos(2 * np.pi * mu * tau),
            axis=-1,
        )
    return k


def _gaussian_logpdf_np(y, mean, cov):
    """log N(y | mean, cov) in float64 via a Cholesky factor."""
    chol = np.linalg.cholesky(cov)
    alpha = np.linalg.solve(chol, y - mean)
    return (
        -0.5 * alpha @ alpha
        - np.log(np.diag(chol)).sum()
        - 0.5 * len(y) * np.log(2 * np.pi)
    )


def _brute_force_mll(lc, hypers):
    x = lc.model.train_inputs[0].double().numpy()
    if x.ndim == 1:
        x = x[:, None]
    y = lc.model.train_targets.double().numpy()
    noise = lc.likelihood.noise.double().numpy()
    weights = hypers["covar_module.mixture_weights"].double().numpy()
    means = hypers["covar_module.mixture_means"].double().numpy().reshape(
        len(weights), -1
    )
    scales = hypers["covar_module.mixture_scales"].double().numpy().reshape(
        len(weights), -1
    )
    tau = x[:, None, :] - x[None, :, :]
    cov = _sm_kernel_np(tau, weights, means, scales) + np.diag(noise)
    const = float(hypers["mean_module.constant"])
    return _gaussian_logpdf_np(y, const, cov) / len(y)


def _norm_logpdf_np(x, mu, s):
    return -0.5 * ((x - mu) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))


class TestMarginalLogLikelihood:
    def test_mll_1d(self, lc_1d, golden):
        value = _exact_mll(lc_1d)
        expected = _brute_force_mll(lc_1d, HYPERS_1D)
        assert math.isclose(value, expected, rel_tol=RTOL)
        golden("snapshot_mll_1d", value, rtol=RTOL)

    def test_mll_2d(self, lc_2d, golden):
        value = _exact_mll(lc_2d)
        expected = _brute_force_mll(lc_2d, HYPERS_2D)
        assert math.isclose(value, expected, rel_tol=RTOL)
        golden("snapshot_mll_2d", value, rtol=RTOL)


class TestPSD:
    # The log PSD is snapshotted: the PSD spans many decades, and an
    # absolute tolerance of 1e-4 on its log is a relative one of 1e-4 on it.

    def test_psd_1d(self, lc_1d, golden):
        freq = torch.as_tensor(FREQ_T, dtype=torch.float32)
        log_psd = lc_1d.compute_psd(freq, log=True).detach().double().numpy()
        # Symmetrised Gaussian: S(f) = w [N(f; mu, s) + N(-f; mu, s)] / 2.
        mu, s = 1.0 / PERIOD, 0.005
        expected = np.logaddexp(
            _norm_logpdf_np(FREQ_T, mu, s), _norm_logpdf_np(-FREQ_T, mu, s)
        ) + np.log(0.5)
        np.testing.assert_allclose(log_psd, expected, rtol=RTOL, atol=1e-4)
        assert FREQ_T[np.argmax(log_psd)] == pytest.approx(mu, rel=0.1)
        golden("snapshot_psd_1d", log_psd, rtol=RTOL, atol=1e-4)

    def test_psd_2d(self, lc_2d, golden):
        freq = (
            torch.as_tensor(FREQ_T, dtype=torch.float32),
            torch.as_tensor(FREQ_WL, dtype=torch.float32),
        )
        log_psd = lc_2d.compute_psd(freq, log=True).detach().double().numpy()
        assert log_psd.shape == (len(FREQ_T), len(FREQ_WL))
        # Regression pin of what compute_psd returns today (no mirror term).
        golden("snapshot_psd_2d", log_psd, rtol=RTOL, atol=1e-4)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="2D compute_psd omits the mirrored component and the 1/2 factor "
        "that the 1D branch applies (roadmap 1.12)",
    )
    def test_psd_2d_symmetrised(self, lc_2d):
        freq = (
            torch.as_tensor(FREQ_T, dtype=torch.float32),
            torch.as_tensor(FREQ_WL, dtype=torch.float32),
        )
        log_psd = lc_2d.compute_psd(freq, log=True).detach().double().numpy()
        # Consistent with the 1D branch: 1/2 [S(f1, f2) + S(-f1, -f2)], weight 1.
        def log_s(f1, f2):
            return (
                _norm_logpdf_np(f1, 1.0 / PERIOD, 0.005)[:, None]
                + _norm_logpdf_np(f2, 0.2, 0.5)[None, :]
            )

        expected = np.logaddexp(
            log_s(FREQ_T, FREQ_WL), log_s(-FREQ_T, -FREQ_WL)
        ) + np.log(0.5)
        np.testing.assert_allclose(log_psd, expected, rtol=RTOL, atol=1e-4)


class TestPeriodSummary:
    def test_period_summary_1d(self, lc_1d, golden):
        # n_grid is reduced from 5000 so that the snapshot stays small; the
        # summary still contains its freq_grid and psd arrays.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary = lc_1d.get_period_summary(n_grid=200).as_dict()
        # The input frequency is 1/37 d^-1; the summary's dominant period is
        # the PSD peak on its grid, so it agrees to the grid resolution.
        assert summary["dominant_period"] == pytest.approx(PERIOD, rel=0.02)
        # Non-numeric fields (method, notes, backend, kernel_family,
        # interval_definition, ...) are compared exactly: a wording change
        # in the notes shows up here and needs a deliberate golden update.
        golden("snapshot_period_summary_1d", summary, rtol=RTOL, atol=1e-6)
