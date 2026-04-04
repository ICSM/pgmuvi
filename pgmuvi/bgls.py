
"""
Torch-based Bayesian Generalized Lomb-Scargle (BGLS) utilities for PGMUVI.

This module is intentionally insulated from the rest of PGMUVI.  It can be
imported independently and used either with raw arrays/tensors or with a
PGMUVI ``Lightcurve`` object.  The single-band implementation follows the
Bayesian GLS logic of analytically marginalizing a weighted sinusoid-plus-offset
model with Gaussian errors and flat priors on the linear coefficients
(Mortier et al. 2015; Zechmeister & Kürster 2009).  The multiband combination
implemented here is *not* a full joint hierarchical multiband Bayesian model:
it is a tempered composite-likelihood approximation, motivated by composite-
likelihood theory and Bayesian adjustment arguments (Lindsay 1988;
Varin, Reid & Firth 2011; Ribatet, Cooley & Davison 2012).

References
----------
Lomb, N. R. 1976, Ap&SS, 39, 447
Scargle, J. D. 1982, ApJ, 263, 835
Zechmeister, M., & Kürster, M. 2009, A&A, 496, 577
Mortier, A., Faria, J. P., Correia, C. M., Santerne, A., & Santos, N. C.
    2015, A&A, 573, A101
VanderPlas, J. T., & Ivezić, Ž. 2015, ApJ, 812, 18
Lindsay, B. G. 1988, Contemporary Mathematics, 80, 221
Varin, C., Reid, N., & Firth, D. 2011, Statistica Sinica, 21, 5
Ribatet, M., Cooley, D., & Davison, A. C. 2012, Statistica Sinica, 22, 813

Assumptions
-----------
1. Within each band, the signal model is a single sinusoid plus a constant
   offset at each trial frequency.
2. Measurement errors are Gaussian and either supplied explicitly or replaced
   by a user-chosen homoscedastic scale.
3. In the multiband combination, the frequency is shared across bands.
4. Amplitudes, phases, and offsets are allowed to differ by band because each
   band is marginalized independently.
5. The cross-band combination assumes conditional independence given frequency
   unless a tempering adjustment is applied.  The tempering does not restore
   a full covariance model; it only mitigates overconfidence.

Design goals
------------
- Standalone use outside ``pgmuvi.lightcurve``.
- PyTorch implementation for speed and optional GPU acceleration.
- Public API kept small and explicit.
- Private helpers prefixed with ``_`` so integration remains contained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import math
import warnings

import numpy as np
import torch


__all__ = [
    "BGLSResult",
    "BGLSBandResult",
    "BGLSMultibandResult",
    "extract_lightcurve_arrays",
    "build_log_frequency_grid",
    "bgls_single_band",
    "bgls_per_band",
    "combine_bgls_bands",
    "fit_bgls",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class BGLSResult:
    """Single-band BGLS output.

    Attributes
    ----------
    frequency : torch.Tensor
        Shared frequency grid in cycles per input time unit.
    log_marginal_likelihood : torch.Tensor
        Log marginal likelihood over frequency after analytically marginalizing
        the sinusoid coefficients and floating offset.
    posterior : torch.Tensor
        Normalized posterior over frequency after applying the chosen prior.
    peak_frequency : float
        Frequency corresponding to the highest posterior value.
    peak_period : float
        Reciprocal of ``peak_frequency``.
    metadata : dict
        Human-readable implementation details, assumptions, and diagnostics.
    """

    frequency: torch.Tensor
    log_marginal_likelihood: torch.Tensor
    posterior: torch.Tensor
    peak_frequency: float
    peak_period: float
    metadata: Dict[str, Any]


@dataclass
class BGLSBandResult(BGLSResult):
    """Single-band result with band identity attached."""
    band_value: float | int | str = "band"


@dataclass
class BGLSMultibandResult:
    """Combined multiband BGLS output."""
    frequency: torch.Tensor
    log_composite_likelihood: torch.Tensor
    posterior: torch.Tensor
    peak_frequency: float
    peak_period: float
    weights: Dict[Any, float]
    band_results: Dict[Any, BGLSBandResult]
    calibration: Dict[str, Any]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _as_1d_tensor(x: Any, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert input to a contiguous 1D tensor."""
    if isinstance(x, torch.Tensor):
        out = x.detach()
        if device is not None:
            out = out.to(device=device, dtype=dtype)
        else:
            out = out.to(dtype=dtype)
    else:
        out = torch.as_tensor(x, dtype=dtype, device=device)
    return out.reshape(-1).contiguous()


def _as_2d_tensor(x: Any, *, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert input to a contiguous 2D tensor."""
    if isinstance(x, torch.Tensor):
        out = x.detach()
        if device is not None:
            out = out.to(device=device, dtype=dtype)
        else:
            out = out.to(dtype=dtype)
    else:
        out = torch.as_tensor(x, dtype=dtype, device=device)
    if out.ndim != 2:
        raise ValueError(f"Expected a 2D tensor/array, got shape {tuple(out.shape)}")
    return out.contiguous()


def _safe_var(x: torch.Tensor) -> torch.Tensor:
    """Variance with a floor to avoid degenerate scaling."""
    if x.numel() < 2:
        return torch.tensor(1.0, dtype=x.dtype, device=x.device)
    v = torch.var(x)
    return torch.clamp(v, min=torch.finfo(x.dtype).tiny)


def _normalize_log_weights(logw: torch.Tensor) -> torch.Tensor:
    """Convert log unnormalized weights to normalized probabilities."""
    shift = torch.max(logw)
    p = torch.exp(logw - shift)
    return p / torch.sum(p)


def _compute_default_sigma(y: torch.Tensor) -> torch.Tensor:
    """Fallback homoscedastic error scale if no ``yerr`` is supplied."""
    sigma = torch.sqrt(_safe_var(y))
    return torch.clamp(sigma, min=torch.tensor(1e-12, dtype=y.dtype, device=y.device))


def _prepare_1d_data(time: Any, flux: Any, flux_err: Any = None, *, dtype: torch.dtype = torch.float64,
                     device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare single-band arrays, drop non-finite values, and sort by time."""
    t = _as_1d_tensor(time, dtype=dtype, device=device)
    y = _as_1d_tensor(flux, dtype=dtype, device=device)
    if flux_err is None:
        dy = torch.full_like(y, _compute_default_sigma(y))
    else:
        dy = _as_1d_tensor(flux_err, dtype=dtype, device=device)
    if not (t.shape == y.shape == dy.shape):
        raise ValueError("time, flux, and flux_err must have the same length")

    finite = torch.isfinite(t) & torch.isfinite(y) & torch.isfinite(dy) & (dy > 0)
    t = t[finite]
    y = y[finite]
    dy = dy[finite]
    if t.numel() < 4:
        raise ValueError("Need at least 4 finite observations for BGLS")

    order = torch.argsort(t)
    return t[order], y[order], dy[order]


def _prepare_multiband_data(xdata: Any, ydata: Any, yerr: Any = None, *, dtype: torch.dtype = torch.float64,
                            device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare 2D xdata=(time, band) with 1D ydata/yerr."""
    x = _as_2d_tensor(xdata, dtype=dtype, device=device)
    if x.shape[1] < 2:
        raise ValueError("Multiband xdata must have at least two columns: time and band")
    y = _as_1d_tensor(ydata, dtype=dtype, device=device)
    if yerr is None:
        dy = torch.full_like(y, _compute_default_sigma(y))
    else:
        dy = _as_1d_tensor(yerr, dtype=dtype, device=device)
    if not (x.shape[0] == y.shape[0] == dy.shape[0]):
        raise ValueError("xdata, ydata, and yerr must have matching row counts")
    t = x[:, 0]
    b = x[:, 1]
    finite = torch.isfinite(t) & torch.isfinite(b) & torch.isfinite(y) & torch.isfinite(dy) & (dy > 0)
    return x[finite], y[finite], dy[finite]


def _unique_bands(band: torch.Tensor) -> torch.Tensor:
    """Return sorted unique band values."""
    return torch.unique(band, sorted=True)


def _split_by_band(xdata: torch.Tensor, ydata: torch.Tensor, yerr: torch.Tensor) -> Dict[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Split multiband data into per-band arrays keyed by band value."""
    bands = _unique_bands(xdata[:, 1])
    out: Dict[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for b in bands:
        mask = xdata[:, 1] == b
        t, y, dy = _prepare_1d_data(xdata[mask, 0], ydata[mask], yerr[mask], dtype=xdata.dtype, device=xdata.device)
        out[float(b.item())] = (t, y, dy)
    return out


def _weighted_band_information(yerr: torch.Tensor) -> float:
    """Cheap proxy for per-band information content.

    This is used in weight option B because it is fast and monotonic in both the
    number of points and the inverse noise variance.  It is not a full Fisher
    information calculation.
    """
    return float(torch.sum(1.0 / torch.square(yerr)).item())


def _band_effective_sample_size(time: torch.Tensor, *, min_sep_factor: float = 0.02) -> float:
    """Heuristic effective sample size based on temporal clustering.

    The routine downweights near-duplicate timestamps by counting a new
    effectively independent point only when the time separation exceeds a small
    fraction of the total baseline.  This is intentionally cheap because option
    B is meant to be the faster default.
    """
    if time.numel() <= 1:
        return float(time.numel())
    baseline = max(float((time[-1] - time[0]).item()), 1e-12)
    threshold = min_sep_factor * baseline
    diffs = torch.diff(time)
    eff = 1.0 + torch.sum((diffs > threshold).to(time.dtype)).item()
    return max(eff, 1.0)


def _estimate_curvature_from_log_posterior(freq: torch.Tensor, logp: torch.Tensor) -> float:
    """Estimate local curvature near the dominant peak in log-frequency space.

    This is used by weight option C as a fast curvature proxy.  The result is
    positive when the peak is concave.
    """
    idx = int(torch.argmax(logp).item())
    if idx == 0 or idx == freq.numel() - 1:
        return 0.0
    x = torch.log(freq[idx - 1: idx + 2])
    y = logp[idx - 1: idx + 2]
    dx1 = x[1] - x[0]
    dx2 = x[2] - x[1]
    if abs(float(dx1.item())) < 1e-15 or abs(float(dx2.item())) < 1e-15:
        return 0.0
    second = 2.0 * (
        y[0] / (dx1 * (dx1 + dx2))
        - y[1] / (dx1 * dx2)
        + y[2] / (dx2 * (dx1 + dx2))
    )
    return float(max((-second).item(), 0.0))


def _resample_blocks(t: torch.Tensor, y: torch.Tensor, dy: torch.Tensor, *, block_size: int,
                     generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Circular moving-block bootstrap for one band."""
    n = t.numel()
    if n <= 2 or block_size <= 1:
        idx = torch.randint(0, n, size=(n,), generator=generator, device=t.device)
        return t[idx], y[idx], dy[idx]
    block_size = min(block_size, n)
    n_blocks = math.ceil(n / block_size)
    starts = torch.randint(0, n, size=(n_blocks,), generator=generator, device=t.device)
    indices = []
    ar = torch.arange(block_size, device=t.device)
    for s in starts:
        indices.append((s + ar) % n)
    idx = torch.cat(indices)[:n]
    return _prepare_1d_data(t[idx], y[idx], dy[idx], dtype=t.dtype, device=t.device)


def _make_prior_log_density(freq: torch.Tensor, prior: str | Mapping[str, Any] = "log_uniform") -> torch.Tensor:
    """Construct a log prior density on the supplied frequency grid.

    Supported priors
    ----------------
    - ``"uniform"`` or ``"frequency_uniform"``: flat in frequency.
    - ``"log_uniform"``: Jeffreys-like flat prior in log frequency.
    - dict with keys ``name`` and optional parameters.
      Currently implemented:
      ``{"name": "truncated_normal", "mu": ..., "sigma": ...}``
      in frequency space.
    """
    if isinstance(prior, str):
        name = prior.lower()
        if name in {"uniform", "frequency_uniform"}:
            return torch.zeros_like(freq)
        if name == "log_uniform":
            return -torch.log(freq)
        raise ValueError(f"Unsupported prior string: {prior!r}")

    if not isinstance(prior, Mapping):
        raise TypeError("prior must be a string or mapping")
    name = str(prior.get("name", "log_uniform")).lower()
    if name in {"uniform", "frequency_uniform"}:
        return torch.zeros_like(freq)
    if name == "log_uniform":
        return -torch.log(freq)
    if name == "truncated_normal":
        mu = float(prior["mu"])
        sigma = float(prior["sigma"])
        if sigma <= 0:
            raise ValueError("sigma must be positive for truncated_normal prior")
        return -0.5 * ((freq - mu) / sigma) ** 2
    raise ValueError(f"Unsupported prior mapping: {prior}")


def _choose_chunk_size(n_time: int, *, target_megabytes: float = 192.0, dtype: torch.dtype = torch.float64) -> int:
    """Choose a frequency chunk size to cap memory use.

    The most memory-hungry temporary arrays scale like ``chunk_size * n_time``.
    """
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    denom = max(n_time * bytes_per * 8, 1)  # loose safety factor
    chunk = int((target_megabytes * 1024**2) / denom)
    return max(chunk, 128)


def _evaluate_single_band_log_marginal(time: torch.Tensor, flux: torch.Tensor, flux_err: torch.Tensor,
                                       frequency: torch.Tensor, *, center_time: bool = True,
                                       ridge_factor: float = 1e-10,
                                       chunk_size: Optional[int] = None) -> torch.Tensor:
    """Evaluate the single-band log marginal likelihood over a frequency grid.

    Notes
    -----
    For each trial frequency the model is

    ``y(t) = a cos(2π f t) + b sin(2π f t) + c + ε``,

    with independent Gaussian errors ``ε ~ N(0, diag(sigma_i^2))``.
    Analytic marginalization over ``(a, b, c)`` with a flat prior yields a
    frequency-dependent log marginal likelihood proportional to

    ``0.5 * bᵀ G⁻¹ b - 0.5 log |G|``,

    where ``G = Xᵀ W X`` and ``b = Xᵀ W y``.  This is the linear-Gaussian
    version of the BGLS formalism of Mortier et al. (2015), expressed directly
    in matrix form.
    """
    if center_time:
        time = time - torch.mean(time)
    w = 1.0 / torch.square(flux_err)
    y = flux - (torch.sum(w * flux) / torch.sum(w))
    two_pi_t = 2.0 * math.pi * time
    n_freq = frequency.numel()
    if chunk_size is None:
        chunk_size = _choose_chunk_size(time.numel(), dtype=time.dtype)

    out = torch.empty(n_freq, dtype=time.dtype, device=time.device)
    one = torch.ones_like(time)
    for start in range(0, n_freq, chunk_size):
        stop = min(start + chunk_size, n_freq)
        f = frequency[start:stop]
        phase = torch.outer(f, two_pi_t)
        c = torch.cos(phase)
        s = torch.sin(phase)

        wc = c * w.unsqueeze(0)
        ws = s * w.unsqueeze(0)
        w1 = w.unsqueeze(0)

        cc = torch.sum(wc * c, dim=1)
        ss = torch.sum(ws * s, dim=1)
        cs = torch.sum(wc * s, dim=1)
        c1 = torch.sum(wc, dim=1)
        s1 = torch.sum(ws, dim=1)
        one1 = torch.full_like(cc, torch.sum(w))

        cy = torch.sum(wc * y.unsqueeze(0), dim=1)
        sy = torch.sum(ws * y.unsqueeze(0), dim=1)
        y1 = torch.full_like(cc, torch.sum(w * y))

        G = torch.stack(
            [
                torch.stack([cc, cs, c1], dim=-1),
                torch.stack([cs, ss, s1], dim=-1),
                torch.stack([c1, s1, one1], dim=-1),
            ],
            dim=-2,
        )  # (m, 3, 3)
        bvec = torch.stack([cy, sy, y1], dim=-1)  # (m, 3)

        diag_scale = torch.clamp(torch.mean(torch.diagonal(G, dim1=-2, dim2=-1), dim=-1), min=1e-16)
        eye = torch.eye(3, dtype=time.dtype, device=time.device).expand(G.shape[0], 3, 3)
        G = G + ridge_factor * diag_scale[:, None, None] * eye

        sign, logdet = torch.linalg.slogdet(G)
        bad = sign <= 0
        if torch.any(bad):
            jitter = 1e-8 * diag_scale[bad][:, None, None] * eye[: bad.sum()]
            G[bad] = G[bad] + jitter
            sign, logdet = torch.linalg.slogdet(G)

        sol = torch.linalg.solve(G, bvec.unsqueeze(-1)).squeeze(-1)
        quad = torch.sum(bvec * sol, dim=-1)
        out[start:stop] = 0.5 * quad - 0.5 * logdet

    return out


def _posterior_from_log_marginal(freq: torch.Tensor, log_marginal: torch.Tensor,
                                 *, prior: str | Mapping[str, Any] = "log_uniform") -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the chosen prior and normalize to obtain the posterior."""
    log_prior = _make_prior_log_density(freq, prior=prior)
    log_post = log_marginal + log_prior
    posterior = _normalize_log_weights(log_post)
    return log_prior, posterior


def _band_results_to_matrix(band_results: Mapping[Any, BGLSBandResult]) -> Tuple[List[Any], torch.Tensor]:
    """Stack per-band log marginal likelihoods into a matrix."""
    keys = list(band_results.keys())
    mat = torch.stack([band_results[k].log_marginal_likelihood for k in keys], dim=0)
    return keys, mat


def _default_weight_option_A(keys: Sequence[Any]) -> Dict[Any, float]:
    """Option A: single global temperature w=1 applied to all bands initially."""
    return {k: 1.0 for k in keys}


def _default_weight_option_B(band_dict: Mapping[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[Any, float]:
    """Option B: fast heteroskedastic information weights.

    Weight for band ``k`` is proportional to

    ``sqrt(effective_sample_size_k * sum_i sigma_i^{-2})``.

    This is cheap, fast, and intentionally conservative compared with raw point
    counts.  The weights are normalized to have mean 1 so they can be used as a
    tempering pattern rather than as absolute scaling.
    """
    raw = {}
    for k, (t, _, dy) in band_dict.items():
        info = _weighted_band_information(dy)
        n_eff = _band_effective_sample_size(t)
        raw[k] = max(math.sqrt(max(info * n_eff, 1e-12)), 1e-8)
    mean_raw = sum(raw.values()) / max(len(raw), 1)
    return {k: v / mean_raw for k, v in raw.items()}


def _default_weight_option_C(freq: torch.Tensor, band_results: Mapping[Any, BGLSBandResult]) -> Dict[Any, float]:
    """Option C: curvature-based weights.

    The local curvature of the log posterior around the dominant band-specific
    peak is used as a cheap proxy for how informative the band is about the
    common frequency.  This is slower than option B because it requires the
    per-band BGLS curves, but it is still much cheaper than repeated
    resampling-based calibration.
    """
    raw = {}
    for k, r in band_results.items():
        log_post = torch.log(torch.clamp(r.posterior, min=torch.finfo(r.posterior.dtype).tiny))
        curv = _estimate_curvature_from_log_posterior(freq, log_post)
        raw[k] = max(math.sqrt(curv + 1e-12), 1e-8)
    mean_raw = sum(raw.values()) / max(len(raw), 1)
    return {k: v / mean_raw for k, v in raw.items()}


def _calibrate_global_temperature_leave_one_band_out(
    freq: torch.Tensor,
    band_results: Mapping[Any, BGLSBandResult],
    weights: Mapping[Any, float],
) -> Dict[str, Any]:
    """Fast O(K) leave-one-band-out calibration.

    We compare the curvature of the full composite log posterior to the median
    curvature of leave-one-band-out composites and set a global temperature to
    match the two on average.  This follows the composite-likelihood spirit of
    reducing overconfidence when correlated evidence is aggregated
    (Varin et al. 2011; Ribatet et al. 2012).
    """
    keys, logL = _band_results_to_matrix(band_results)
    w = torch.tensor([weights[k] for k in keys], dtype=logL.dtype, device=logL.device)
    full = torch.sum(w[:, None] * logL, dim=0)
    full_curv = _estimate_curvature_from_log_posterior(freq, full)
    loo_curvs = []
    for i in range(len(keys)):
        comp = torch.sum(w[torch.arange(len(keys)) != i][:, None] * logL[torch.arange(len(keys)) != i], dim=0)
        loo_curvs.append(_estimate_curvature_from_log_posterior(freq, comp))
    target_curv = float(np.median(np.asarray(loo_curvs))) if loo_curvs else full_curv
    if full_curv <= 0:
        temp = 1.0
    else:
        temp = min(max(target_curv / full_curv, 0.05), 1.0)
    return {
        "method": "leave_one_band_out",
        "global_temperature": float(temp),
        "full_curvature": float(full_curv),
        "loo_curvatures": loo_curvs,
        "target_curvature": float(target_curv),
    }


def _calibrate_global_temperature_block_bootstrap(
    freq: torch.Tensor,
    band_dict: Mapping[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    weights: Mapping[Any, float],
    *,
    prior: str | Mapping[str, Any],
    n_bootstrap: int = 32,
    block_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Calibrate a global temperature using moving-block bootstrap within each band.

    This is slower than leave-one-band-out because each resample recomputes BGLS
    curves.  It is still implemented with the same shared frequency grid to keep
    the cost acceptable.
    """
    peak_freqs = []
    for _ in range(n_bootstrap):
        br = {}
        for k, (t, y, dy) in band_dict.items():
            bs = block_size or max(4, int(math.sqrt(t.numel())))
            tb, yb, dyb = _resample_blocks(t, y, dy, block_size=bs, generator=generator)
            br[k] = bgls_single_band(tb, yb, dyb, frequency=freq, prior=prior)
        comp = combine_bgls_bands(br, prior=prior, weight_method="custom", custom_weights=weights,
                                  calibration_method="none")
        peak_freqs.append(comp.peak_frequency)
    peak_freqs = np.asarray(peak_freqs, dtype=float)
    peak_std = float(np.std(peak_freqs)) if peak_freqs.size > 1 else 0.0

    # Convert empirical frequency spread to an approximate curvature target in log f.
    if peak_std <= 0:
        temp = 1.0
        target_curv = 0.0
    else:
        peak = float(np.median(peak_freqs))
        sigma_logf = peak_std / max(peak, 1e-15)
        target_curv = 1.0 / max(sigma_logf**2, 1e-12)

        keys = list(weights.keys())
        logL = torch.stack([band_dict_result.log_marginal_likelihood for band_dict_result in []]) if False else None  # no-op for static analyzers

        # Reconstruct the nominal full composite after the bootstrap loop.
        # This keeps the code simple and the cost negligible relative to the resamples.
        band_results_full = {
            k: bgls_single_band(*band_dict[k], frequency=freq, prior=prior)
            for k in band_dict
        }
        keys, logL = _band_results_to_matrix(band_results_full)
        w = torch.tensor([weights[k] for k in keys], dtype=logL.dtype, device=logL.device)
        full = torch.sum(w[:, None] * logL, dim=0)
        full_curv = _estimate_curvature_from_log_posterior(freq, full)
        temp = min(max(target_curv / max(full_curv, 1e-12), 0.05), 1.0)

    return {
        "method": "block_bootstrap",
        "global_temperature": float(temp),
        "bootstrap_peak_frequencies": peak_freqs.tolist(),
        "bootstrap_peak_std": peak_std,
    }


def _calibrate_global_temperature_nested_resampling(
    freq: torch.Tensor,
    band_dict: Mapping[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    weights: Mapping[Any, float],
    *,
    prior: str | Mapping[str, Any],
    n_outer: int = 16,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Nested band-level/time-level resampling.

    On each outer iteration, a subset of bands is sampled with replacement and
    each selected band is block-bootstrapped in time.  This is the slowest
    calibration option in this module, but it is also the most direct stress
    test of both cross-band redundancy and within-band temporal dependence.
    """
    band_keys = list(band_dict.keys())
    if not band_keys:
        raise ValueError("No bands available for nested calibration")
    peak_freqs = []
    for _ in range(n_outer):
        draw_idx = torch.randint(0, len(band_keys), size=(len(band_keys),), generator=generator)
        sampled_keys = [band_keys[int(i)] for i in draw_idx]
        br = {}
        w_sub = {}
        for k in sampled_keys:
            t, y, dy = band_dict[k]
            bs = max(4, int(math.sqrt(t.numel())))
            tb, yb, dyb = _resample_blocks(t, y, dy, block_size=bs, generator=generator)
            tag = f"{k}_draw{len(br)}"
            br[tag] = bgls_single_band(tb, yb, dyb, frequency=freq, prior=prior)
            w_sub[tag] = float(weights[k])
        comp = combine_bgls_bands(br, prior=prior, weight_method="custom", custom_weights=w_sub,
                                  calibration_method="none")
        peak_freqs.append(comp.peak_frequency)
    peak_freqs = np.asarray(peak_freqs, dtype=float)
    peak_std = float(np.std(peak_freqs)) if peak_freqs.size > 1 else 0.0
    if peak_std <= 0:
        temp = 1.0
    else:
        peak = float(np.median(peak_freqs))
        sigma_logf = peak_std / max(peak, 1e-15)
        # Conservative calibration: make the nominal posterior no sharper than the resampling spread.
        nominal_band_results = {
            k: bgls_single_band(*band_dict[k], frequency=freq, prior=prior)
            for k in band_dict
        }
        keys, logL = _band_results_to_matrix(nominal_band_results)
        w = torch.tensor([weights[k] for k in keys], dtype=logL.dtype, device=logL.device)
        full = torch.sum(w[:, None] * logL, dim=0)
        full_curv = _estimate_curvature_from_log_posterior(freq, full)
        target_curv = 1.0 / max(sigma_logf**2, 1e-12)
        temp = min(max(target_curv / max(full_curv, 1e-12), 0.05), 1.0)
    return {
        "method": "nested_band_time_resampling",
        "global_temperature": float(temp),
        "resampled_peak_frequencies": peak_freqs.tolist(),
        "resampled_peak_std": peak_std,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_lightcurve_arrays(
    lightcurve: Any,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Extract arrays from a PGMUVI ``Lightcurve`` without mutating it.

    Parameters
    ----------
    lightcurve : object
        Expected to expose ``xdata`` and ``ydata`` attributes, and optionally
        ``yerr``.
    dtype, device : optional
        Torch conversion settings.

    Returns
    -------
    dict
        Keys are ``xdata``, ``ydata``, and ``yerr`` (possibly ``None``).

    Notes
    -----
    This helper keeps the rest of the implementation insulated from
    ``pgmuvi.lightcurve.Lightcurve``.  The module never imports that class.
    """
    if not hasattr(lightcurve, "xdata") or not hasattr(lightcurve, "ydata"):
        raise TypeError("lightcurve must expose xdata and ydata attributes")
    x = lightcurve.xdata
    y = lightcurve.ydata
    dy = getattr(lightcurve, "yerr", None)
    if isinstance(x, torch.Tensor) and x.ndim == 1:
        x_out = _as_1d_tensor(x, dtype=dtype, device=device)
    else:
        x_out = _as_2d_tensor(x, dtype=dtype, device=device) if getattr(x, "ndim", None) == 2 else _as_1d_tensor(x, dtype=dtype, device=device)
    y_out = _as_1d_tensor(y, dtype=dtype, device=device)
    dy_out = None if dy is None else _as_1d_tensor(dy, dtype=dtype, device=device)
    return {"xdata": x_out, "ydata": y_out, "yerr": dy_out}


def build_log_frequency_grid(
    time: Any,
    *,
    samples_per_peak: int = 8,
    nyquist_factor: float = 5.0,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    n_frequency: Optional[int] = None,
    oversampling: float = 4.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Construct a shared logarithmic frequency grid.

    Parameters
    ----------
    time : array-like
        Time values.  Only a 1D time axis is needed; for 2D multiband input,
        pass the time column.
    samples_per_peak : int, default=8
        Target number of samples across a characteristic peak width in the
        *linear-frequency* sense.  This influences the default number of grid
        points when ``n_frequency`` is not supplied.
    nyquist_factor : float, default=5
        Irregular-sampling analogue of the Astropy ``nyquist_factor``.  The
        default upper frequency is
        ``0.5 * nyquist_factor / median_positive_dt``.
    minimum_frequency, maximum_frequency : float, optional
        Explicit grid bounds.  If omitted, they are estimated from the time
        baseline and median cadence.
    n_frequency : int, optional
        Explicit number of grid points.  If omitted, an automatic value is
        derived from the baseline, cadence, and requested oversampling.
    oversampling : float, default=4
        Multiplier on the automatic grid density.  Larger values increase
        resolution and runtime.

    Returns
    -------
    torch.Tensor
        Logarithmically spaced trial frequencies.

    Notes
    -----
    The grid is built in log frequency because the multiband composite method
    is primarily used to compare peak structure and posterior concentration
    across a wide dynamic range of periods.
    """
    t = _as_1d_tensor(time, dtype=dtype, device=device)
    finite = torch.isfinite(t)
    t = torch.sort(t[finite]).values
    if t.numel() < 4:
        raise ValueError("Need at least 4 finite time points to build a frequency grid")

    baseline = float((t[-1] - t[0]).item())
    if baseline <= 0:
        raise ValueError("Time baseline must be positive")

    diffs = torch.diff(t)
    diffs = diffs[diffs > 0]
    if diffs.numel() == 0:
        raise ValueError("Time axis must contain at least two distinct values")
    med_dt = float(torch.median(diffs).item())

    fmin = (1.0 / baseline) if minimum_frequency is None else float(minimum_frequency)
    fmax = (0.5 * nyquist_factor / med_dt) if maximum_frequency is None else float(maximum_frequency)

    if fmin <= 0 or fmax <= 0 or fmax <= fmin:
        raise ValueError(f"Invalid frequency bounds: fmin={fmin}, fmax={fmax}")

    if n_frequency is None:
        n_linear = max(int(oversampling * samples_per_peak * baseline * (fmax - fmin)), 512)
        n_log = max(int(oversampling * samples_per_peak * math.log10(fmax / fmin) * 128), 512)
        n_frequency = max(n_linear, n_log)
    return torch.logspace(math.log10(fmin), math.log10(fmax), steps=int(n_frequency), dtype=dtype, device=device)


def bgls_single_band(
    time: Any,
    flux: Any,
    flux_err: Any = None,
    *,
    frequency: Optional[Any] = None,
    prior: str | Mapping[str, Any] = "log_uniform",
    center_time: bool = True,
    ridge_factor: float = 1e-10,
    chunk_size: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    grid_kwargs: Optional[Mapping[str, Any]] = None,
) -> BGLSResult:
    """Compute a single-band BGLS curve on a supplied or auto-generated grid.

    References
    ----------
    Zechmeister & Kürster (2009) for the weighted floating-mean sinusoid model.
    Mortier et al. (2015) for the Bayesian GLS interpretation and relative
    probability comparison among peaks.

    Notes
    -----
    The implementation analytically marginalizes the linear coefficients of a
    weighted sinusoid-plus-offset model.  This is exact for Gaussian errors and
    flat priors on the linear coefficients.
    """
    t, y, dy = _prepare_1d_data(time, flux, flux_err, dtype=dtype, device=device)
    if frequency is None:
        frequency = build_log_frequency_grid(t, dtype=dtype, device=device, **(dict(grid_kwargs or {})))
    else:
        frequency = _as_1d_tensor(frequency, dtype=dtype, device=device)
    log_marginal = _evaluate_single_band_log_marginal(
        t, y, dy, frequency, center_time=center_time, ridge_factor=ridge_factor, chunk_size=chunk_size
    )
    log_prior, posterior = _posterior_from_log_marginal(frequency, log_marginal, prior=prior)
    idx = int(torch.argmax(posterior).item())
    peak_frequency = float(frequency[idx].item())
    return BGLSResult(
        frequency=frequency,
        log_marginal_likelihood=log_marginal,
        posterior=posterior,
        peak_frequency=peak_frequency,
        peak_period=float(1.0 / peak_frequency),
        metadata={
            "model": "sinusoid_plus_offset",
            "errors": "gaussian_independent",
            "prior": prior,
            "includes_prior_in_posterior": True,
            "center_time": center_time,
            "ridge_factor": ridge_factor,
            "log_prior": log_prior,
            "citation_note": (
                "Single-band model follows the weighted floating-mean sinusoid of "
                "Zechmeister & Kürster (2009) and the Bayesian GLS interpretation "
                "of Mortier et al. (2015)."
            ),
        },
    )


def bgls_per_band(
    xdata: Any,
    ydata: Any,
    yerr: Any = None,
    *,
    frequency: Optional[Any] = None,
    prior: str | Mapping[str, Any] = "log_uniform",
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    grid_kwargs: Optional[Mapping[str, Any]] = None,
    single_band_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[Any, BGLSBandResult]:
    """Compute per-band BGLS curves for a multiband light curve.

    Parameters
    ----------
    xdata : array-like, shape (N, 2)
        Column 0 is time and column 1 is band/wavelength identifier.
    ydata, yerr : array-like
        Flux and optional errors.
    frequency : array-like, optional
        Shared frequency grid.  If omitted, it is built from all times.
    prior : str or dict, default="log_uniform"
        Prior over frequency.
    grid_kwargs, single_band_kwargs : dict, optional
        Extra settings for grid construction and per-band evaluation.

    Returns
    -------
    dict
        Keys are band values, values are ``BGLSBandResult`` instances.

    Notes
    -----
    A shared grid is required so that the later multiband combination is
    meaningful.  The period is assumed to be common across bands, but
    amplitude/phase/offset are allowed to differ by band because each band is
    marginalized independently.
    """
    x, y, dy = _prepare_multiband_data(xdata, ydata, yerr, dtype=dtype, device=device)
    band_dict = _split_by_band(x, y, dy)
    if frequency is None:
        frequency = build_log_frequency_grid(x[:, 0], dtype=dtype, device=device, **(dict(grid_kwargs or {})))
    else:
        frequency = _as_1d_tensor(frequency, dtype=dtype, device=device)

    out: Dict[Any, BGLSBandResult] = {}
    kwargs = dict(single_band_kwargs or {})
    for band, (t, f, ferr) in band_dict.items():
        res = bgls_single_band(
            t, f, ferr,
            frequency=frequency,
            prior=prior,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        out[band] = BGLSBandResult(
            frequency=res.frequency,
            log_marginal_likelihood=res.log_marginal_likelihood,
            posterior=res.posterior,
            peak_frequency=res.peak_frequency,
            peak_period=res.peak_period,
            metadata=res.metadata,
            band_value=band,
        )
    return out


def combine_bgls_bands(
    band_results: Mapping[Any, BGLSBandResult],
    *,
    prior: str | Mapping[str, Any] = "log_uniform",
    weight_method: str = "B",
    custom_weights: Optional[Mapping[Any, float]] = None,
    calibration_method: str = "leave_one_band_out",
    calibration_kwargs: Optional[Mapping[str, Any]] = None,
    raw_band_data: Optional[Mapping[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
) -> BGLSMultibandResult:
    """Combine per-band BGLS curves with a tempered composite likelihood.

    Parameters
    ----------
    band_results : mapping
        Per-band results returned by :func:`bgls_per_band`.
    prior : str or dict, default="log_uniform"
        Prior applied *once* at the combination stage.
    weight_method : {"A", "B", "C", "custom"}, default="B"
        Weight/temperature pattern across bands.

        - ``A``: equal weights.
        - ``B``: fast information-based weights.  This is the recommended
          default because it is faster than ``C``.
        - ``C``: curvature-based weights using each band's posterior shape.
        - ``custom``: use ``custom_weights`` exactly as supplied.

    calibration_method : {"leave_one_band_out", "block_bootstrap",
                          "nested_band_time_resampling", "none"},
                          default="leave_one_band_out"
        Global temperature calibration.  The default is the fastest available
        calibration that still reacts to cross-band redundancy.

    raw_band_data : mapping, optional
        Required for the resampling-based calibration methods because they need
        access to the original per-band arrays.

    Returns
    -------
    BGLSMultibandResult
        Combined result.

    Notes
    -----
    The composite posterior is

    ``log p(f | D) = log p(f) + T * Σ_k w_k log L_k(f) + const``,

    where ``T`` is a calibrated global temperature and ``w_k`` are the band
    weights.  This is a tempered composite likelihood, not a full multiband
    generative model.  It is statistically defensible when documented as an
    approximation, particularly if the calibration step is retained
    (Varin et al. 2011; Ribatet et al. 2012).
    """
    if not band_results:
        raise ValueError("band_results must not be empty")
    keys, logL = _band_results_to_matrix(band_results)
    freq = next(iter(band_results.values())).frequency
    cal_kwargs = dict(calibration_kwargs or {})

    wm = weight_method.upper()
    if wm == "A":
        weights = _default_weight_option_A(keys)
    elif wm == "B":
        if raw_band_data is None:
            # fallback to curvature-free equal weights if raw arrays are missing
            warnings.warn("raw_band_data missing for weight method B; falling back to equal weights")
            weights = _default_weight_option_A(keys)
        else:
            weights = _default_weight_option_B(raw_band_data)
    elif wm == "C":
        weights = _default_weight_option_C(freq, band_results)
    elif wm == "CUSTOM":
        if custom_weights is None:
            raise ValueError("custom_weights must be supplied when weight_method='custom'")
        weights = {k: float(custom_weights[k]) for k in keys}
    else:
        raise ValueError(f"Unsupported weight_method: {weight_method!r}")

    cm = calibration_method.lower()
    if cm == "none":
        calibration = {"method": "none", "global_temperature": 1.0}
    elif cm == "leave_one_band_out":
        calibration = _calibrate_global_temperature_leave_one_band_out(freq, band_results, weights)
    elif cm == "block_bootstrap":
        if raw_band_data is None:
            raise ValueError("raw_band_data is required for block_bootstrap calibration")
        calibration = _calibrate_global_temperature_block_bootstrap(freq, raw_band_data, weights, prior=prior, **cal_kwargs)
    elif cm == "nested_band_time_resampling":
        if raw_band_data is None:
            raise ValueError("raw_band_data is required for nested_band_time_resampling calibration")
        calibration = _calibrate_global_temperature_nested_resampling(freq, raw_band_data, weights, prior=prior, **cal_kwargs)
    else:
        raise ValueError(f"Unsupported calibration_method: {calibration_method!r}")

    T = float(calibration["global_temperature"])
    wvec = torch.tensor([weights[k] for k in keys], dtype=logL.dtype, device=logL.device)
    log_prior = _make_prior_log_density(freq, prior=prior)
    log_comp = T * torch.sum(wvec[:, None] * logL, dim=0)
    posterior = _normalize_log_weights(log_comp + log_prior)
    idx = int(torch.argmax(posterior).item())
    peak_frequency = float(freq[idx].item())

    return BGLSMultibandResult(
        frequency=freq,
        log_composite_likelihood=log_comp,
        posterior=posterior,
        peak_frequency=peak_frequency,
        peak_period=float(1.0 / peak_frequency),
        weights=weights,
        band_results=dict(band_results),
        calibration=calibration,
        metadata={
            "weight_method": weight_method,
            "calibration_method": calibration_method,
            "prior": prior,
            "assumptions": [
                "shared_frequency_across_bands",
                "band_specific_amplitude_phase_offset",
                "gaussian_independent_errors_within_band",
                "cross_band_combination_via_tempered_composite_likelihood",
            ],
            "citation_note": (
                "Cross-band combination is motivated by the shared-period "
                "multiband logic of VanderPlas & Ivezić (2015), but it is "
                "implemented here as a tempered composite likelihood in the "
                "sense of Lindsay (1988), Varin et al. (2011), and "
                "Ribatet et al. (2012)."
            ),
        },
    )


def fit_bgls(
    data: Any,
    ydata: Any = None,
    yerr: Any = None,
    *,
    frequency: Optional[Any] = None,
    prior: str | Mapping[str, Any] = "log_uniform",
    weight_method: str = "B",
    calibration_method: str = "leave_one_band_out",
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    grid_kwargs: Optional[Mapping[str, Any]] = None,
    single_band_kwargs: Optional[Mapping[str, Any]] = None,
    calibration_kwargs: Optional[Mapping[str, Any]] = None,
) -> BGLSResult | BGLSMultibandResult:
    """High-level convenience interface for 1D or 2D input.

    Accepted input forms
    --------------------
    1. ``fit_bgls(lightcurve, ...)`` where ``lightcurve`` exposes ``xdata``,
       ``ydata``, and optional ``yerr``.
    2. ``fit_bgls(time, flux, flux_err, ...)`` for single-band data.
    3. ``fit_bgls(xdata_2d, ydata, yerr, ...)`` for multiband data with
       ``xdata[:, 0] = time`` and ``xdata[:, 1] = band``.

    Returns
    -------
    BGLSResult or BGLSMultibandResult
    """
    if ydata is None and hasattr(data, "xdata") and hasattr(data, "ydata"):
        arr = extract_lightcurve_arrays(data, dtype=dtype, device=device)
        x = arr["xdata"]
        y = arr["ydata"]
        dy = arr["yerr"]
    else:
        x = data
        y = ydata
        dy = yerr

    if isinstance(x, torch.Tensor) and x.ndim == 2:
        x2, y2, dy2 = _prepare_multiband_data(x, y, dy, dtype=dtype, device=device)
        raw_band_data = _split_by_band(x2, y2, dy2)
        per = bgls_per_band(
            x2, y2, dy2,
            frequency=frequency,
            prior=prior,
            dtype=dtype,
            device=device,
            grid_kwargs=grid_kwargs,
            single_band_kwargs=single_band_kwargs,
        )
        return combine_bgls_bands(
            per,
            prior=prior,
            weight_method=weight_method,
            calibration_method=calibration_method,
            calibration_kwargs=calibration_kwargs,
            raw_band_data=raw_band_data,
        )

    x1, y1, dy1 = _prepare_1d_data(x, y, dy, dtype=dtype, device=device)
    return bgls_single_band(
        x1, y1, dy1,
        frequency=frequency,
        prior=prior,
        dtype=dtype,
        device=device,
        grid_kwargs=grid_kwargs,
        **dict(single_band_kwargs or {}),
    )
